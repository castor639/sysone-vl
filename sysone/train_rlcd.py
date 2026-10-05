import argparse
import json
import os
import random
import time

import torch

from .envs import tabletop
from .losses import rlcd_loss, sample_decisions
from .model import SystemOneVL


def rollout_batch(rng, bs):
    scenes = [tabletop.sample_scene(rng) for _ in range(bs)]
    queries = [tabletop.make_query(s, tabletop.render(s), with_answers=False) for s in scenes]
    return scenes, queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/sft")
    ap.add_argument("--out", default="runs/rlcd")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--explore", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    model = SystemOneVL.load(a.init).cuda()
    ref = None
    if a.beta > 0:
        ref = SystemOneVL.load(a.init, gradient_checkpointing=False).cuda().eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    head_params = list(model.heads.parameters()) + list(model.log_temp.parameters())
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}], weight_decay=0.0)

    model.train()
    log, t0 = [], time.time()
    hits, tot = 0, 0
    for step in range(1, a.steps + 1):
        scenes, queries = rollout_batch(rng, a.bs)
        outputs = model(queries)
        with torch.no_grad():
            ref_out = ref(queries) if ref is not None else None
        decisions = sample_decisions(outputs, queries, a.explore, gen)
        outs = [tabletop.outcomes(s, d, rng) for s, d in zip(scenes, decisions)]
        loss, stats = rlcd_loss(outputs, ref_out, queries, decisions, outs, a.beta, a.ent)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        hits += sum(o[0] for o in outs); tot += len(outs)
        stats.update(step=step, loss=float(loss), choice_hit=hits / tot, t=time.time() - t0)
        log.append(stats)
        if step % a.log_every == 0:
            print(f"step {step}/{a.steps} loss {float(loss):.4f} cal {stats['cal']:.4f} kl {stats['kl']:.4f} "
                  f"ent {stats['ent']:.3f} choice_hit {hits / tot:.3f} temps "
                  + " ".join(f"{k}={v.exp().item():.2f}" for k, v in model.log_temp.items()), flush=True)
            hits, tot = 0, 0
    model.save(a.out)
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump(log, f)
    print("saved", a.out)


if __name__ == "__main__":
    main()
