import argparse
import json
import re
import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .data import JsonlQueries
from .questions import letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default="data/tabletop/test.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=224 * 224, max_pixels=336 * 336)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    ds = JsonlQueries(a.data, a.limit)
    hits, times = [], []
    for i in range(len(ds)):
        q = ds[i]
        c = q.questions[0]
        text = ("STATE:\n" + q.state + "\n\n" + c.question + "\n"
                + "\n".join(f"{letter(k)}. {o}" for k, o in enumerate(c.options))
                + "\nAnswer with the letter only.")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=q.images, return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        ans = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        mt = re.search(r"[A-Z]", ans.strip().upper())
        pred = ord(mt.group(0)) - ord("A") if mt else -1
        hits.append(pred == c.answer)
    res = {"acc": float(np.mean(hits)), "median_ms": 1000 * float(np.median(times[3:] or times)), "n": len(hits)}
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
