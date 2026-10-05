import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .data import JsonlQueries, collate
from .losses import supervised_loss
from .model import SystemOneVL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tabletop/train.jsonl")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(0)
    model = SystemOneVL(a.model, lora_r=a.lora_r).cuda()
    model.backbone.print_trainable_parameters()
    ds = JsonlQueries(a.data, a.limit)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=2)
    head_params = list(model.heads.parameters())
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}], weight_decay=0.01)
    total_steps = int(len(dl) * a.epochs / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / max(total_steps, 1)))

    model.train()
    step, seen, t0, log = 0, 0, time.time(), []
    done = False
    while not done:
        for i, batch in enumerate(dl):
            loss = supervised_loss(model(batch), batch) / a.accum
            loss.backward()
            seen += len(batch)
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                log.append({"step": step, "loss": float(loss) * a.accum, "t": time.time() - t0})
                if step % a.log_every == 0:
                    print(f"step {step}/{total_steps} loss {float(loss) * a.accum:.4f} {seen / (time.time() - t0):.1f} samples/s", flush=True)
                if step >= total_steps:
                    done = True
                    break
    model.save(a.out)
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump(log, f)
    print("saved", a.out)


if __name__ == "__main__":
    main()
