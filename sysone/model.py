import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .questions import LEAF, Query, build_messages

LM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LeafHead(nn.Module):
    def __init__(self, hidden, inner=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(), nn.Linear(inner, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def lm_lora_targets(backbone):
    names = []
    for name, mod in backbone.named_modules():
        if "visual" in name or not isinstance(mod, nn.Linear):
            continue
        if name.split(".")[-1] in LM_TARGETS:
            names.append(name)
    return names


class SystemOneVL(nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        min_pixels=224 * 224,
        max_pixels=336 * 336,
        dtype=torch.bfloat16,
        gradient_checkpointing=True,
        adapter_path=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.cfg = dict(model_name=model_name, lora_r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "right"
        self.leaf_id = self.processor.tokenizer.convert_tokens_to_ids(LEAF)
        assert isinstance(self.leaf_id, int) and self.leaf_id >= 0, f"missing leaf token {LEAF}"

        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        backbone.config.use_cache = False
        hidden = backbone.config.text_config.hidden_size if hasattr(backbone.config, "text_config") else backbone.config.hidden_size
        backbone.lm_head = nn.Identity()
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()

        if adapter_path is None:
            lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=lm_lora_targets(backbone), bias="none")
            self.backbone = get_peft_model(backbone, lcfg)
        else:
            self.backbone = PeftModel.from_pretrained(backbone, os.path.join(adapter_path, "adapter"), is_trainable=True)

        self.heads = nn.ModuleDict({k: LeafHead(hidden) for k in ("choice", "score", "noul")})
        self.log_temp = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in ("choice", "score", "noul")})
        if adapter_path is not None:
            sd = torch.load(os.path.join(adapter_path, "heads.pt"), map_location="cpu")
            self.heads.load_state_dict(sd["heads"])
            self.log_temp.load_state_dict(sd["log_temp"])

    @property
    def device(self):
        return next(self.heads.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encode(self, queries: List[Query]):
        texts, images = [], []
        for q in queries:
            texts.append(self.processor.apply_chat_template(build_messages(q), tokenize=False, add_generation_prompt=True))
            images.extend(q.images)
        inputs = self.processor(text=texts, images=images or None, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, queries: List[Query]):
        inputs = self.encode(queries)
        hidden = self.backbone(**inputs).logits
        out = []
        for b, q in enumerate(queries):
            pos = (inputs["input_ids"][b] == self.leaf_id).nonzero(as_tuple=True)[0]
            want = sum(x.n_leaves() for x in q.questions)
            assert pos.numel() == want, f"leaf count {pos.numel()} != {want}"
            feats = hidden[b, pos].float()
            per_q, i = [], 0
            for x in q.questions:
                n = x.n_leaves()
                logits = self.heads[x.type](feats[i:i + n]) / self.log_temp[x.type].exp()
                per_q.append(logits)
                i += n
            out.append(per_q)
        return out

    @torch.no_grad()
    def decide(self, queries: List[Query]):
        self.eval()
        results = []
        for per_q, q in zip(self.forward(queries), queries):
            res = []
            for logits, x in zip(per_q, q.questions):
                res.append(decision_from_logits(logits, x))
            results.append(res)
        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(os.path.join(path, "adapter"))
        torch.save({"heads": self.heads.state_dict(), "log_temp": self.log_temp.state_dict()}, os.path.join(path, "heads.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)

    @classmethod
    def load(cls, path, **kw):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        cfg.update(kw)
        return cls(adapter_path=path, **cfg)


def decision_from_logits(logits, x):
    if x.type == "noul":
        p = torch.sigmoid(logits[0]).item()
        return {"type": "noul", "p": p, "decision": p >= 0.5, "confidence": max(p, 1 - p)}
    probs = F.softmax(logits, dim=-1)
    idx = int(probs.argmax())
    if x.type == "choice":
        return {"type": "choice", "probs": probs.tolist(), "decision": idx,
                "option": x.options[idx], "confidence": float(probs[idx])}
    values = x.values or list(range(1, len(x.levels) + 1))
    expected = float((probs * torch.tensor(values, device=probs.device)).sum())
    return {"type": "score", "probs": probs.tolist(), "decision": idx, "level": x.levels[idx],
            "expected": expected, "confidence": float(probs[idx])}
