import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import JsonlQueries, collate
from .losses import brier, ece
from .model import SystemOneVL


@torch.no_grad()
def collect(model, ds, bs):
    dl = DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=2)
    logits, questions = [], []
    for batch in dl:
        for per_q, q in zip(model(batch), batch):
            logits.append([l.detach().cpu() for l in per_q])
            questions.append(q.questions)
    return logits, questions


def fit_temperatures(model, ds, bs):
    logits, questions = collect(model, ds, bs)
    for t in ("choice", "score", "noul"):
        ls, ys = [], []
        for per_q, qs in zip(logits, questions):
            for l, x in zip(per_q, qs):
                if x.type == t and x.answer is not None:
                    ls.append(l * model.log_temp[t].exp().item()); ys.append(x.answer)
        if not ls:
            continue
        lt = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            nll = 0.0
            for l, y in zip(ls, ys):
                z = l / lt.exp()
                if t == "noul":
                    nll = nll + F.binary_cross_entropy_with_logits(z[0], torch.tensor(float(y)))
                else:
                    nll = nll + F.cross_entropy(z.unsqueeze(0), torch.tensor([int(y)]))
            nll = nll / len(ls)
            nll.backward()
            return nll

        opt.step(closure)
        with torch.no_grad():
            model.log_temp[t].fill_(lt.item())
        print(f"temp {t} = {lt.exp().item():.3f}")


def evaluate(model, ds, bs, seed=0, dump=None):
    rng = random.Random(seed)
    logits, questions = collect(model, ds, bs)
    if dump:
        with open(dump, "w") as f:
            for row, per_q in zip(ds.rows, logits):
                f.write(json.dumps({"images": row["images"], "meta": row["meta"],
                                    "logits": [l.tolist() for l in per_q]}) + "\n")
    m = {"choice": {"conf": [], "correct": []}, "score": {"conf": [], "correct": [], "abs_err": []},
         "reach": {"conf": [], "correct": []}, "grasp": {"p": [], "outcome": [], "p_true": []}}
    for row, per_q, qs in zip(ds.rows, logits, questions):
        for j, (l, x) in enumerate(zip(per_q, qs)):
            if x.type == "choice":
                p = F.softmax(l, -1)
                m["choice"]["conf"].append(float(p.max())); m["choice"]["correct"].append(int(p.argmax()) == x.answer)
            elif x.type == "score":
                p = F.softmax(l, -1)
                m["score"]["conf"].append(float(p.max())); m["score"]["correct"].append(int(p.argmax()) == x.answer)
                vals = torch.tensor(x.values, dtype=torch.float)
                m["score"]["abs_err"].append(abs(float((p * vals).sum()) - x.values[x.answer]))
            elif j == 1:
                p = float(torch.sigmoid(l[0]))
                m["reach"]["conf"].append(max(p, 1 - p)); m["reach"]["correct"].append((p >= 0.5) == bool(x.answer))
            else:
                p = float(torch.sigmoid(l[0]))
                pt = row["meta"]["p_grasp"]
                m["grasp"]["p"].append(p); m["grasp"]["p_true"].append(pt)
                m["grasp"]["outcome"].append(int(rng.random() < pt))
    out = {}
    for k in ("choice", "score", "reach"):
        out[k] = {"acc": float(np.mean(m[k]["correct"])), "ece": ece(m[k]["conf"], m[k]["correct"]),
                  "brier": brier(m[k]["conf"], m[k]["correct"]), "n": len(m[k]["conf"])}
    out["score"]["mae_expected"] = float(np.mean(m["score"]["abs_err"]))
    g = m["grasp"]
    out["grasp"] = {"ece_vs_outcomes": ece(g["p"], g["outcome"]), "brier_vs_outcomes": brier(g["p"], g["outcome"]),
                    "mae_vs_true_p": float(np.mean(np.abs(np.array(g["p"]) - np.array(g["p_true"])))),
                    "acc_hard": float(np.mean((np.array(g["p"]) >= 0.5) == (np.array(g["p_true"]) >= 0.5))),
                    "n": len(g["p"])}
    return out


@torch.no_grad()
def latency(model, ds, n=30):
    times = []
    for i in range(min(n, len(ds))):
        q = ds[i]
        torch.cuda.synchronize(); t = time.perf_counter()
        model.decide([q])
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    times = times[3:] or times
    return {"median_ms": 1000 * float(np.median(times)), "p90_ms": 1000 * float(np.percentile(times, 90)),
            "questions_per_query": len(ds[0].questions)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rlcd")
    ap.add_argument("--data", default="data/tabletop/test.jsonl")
    ap.add_argument("--fit-temp", default=None)
    ap.add_argument("--save-calibrated", default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    model = SystemOneVL.load(a.ckpt, gradient_checkpointing=False).cuda().eval()
    if a.fit_temp:
        fit_temperatures(model, JsonlQueries(a.fit_temp), a.bs)
        if a.save_calibrated:
            model.save(a.save_calibrated)
    ds = JsonlQueries(a.data, a.limit)
    res = evaluate(model, ds, a.bs, dump=a.dump)
    res["latency"] = latency(model, ds)
    res["temps"] = {k: v.exp().item() for k, v in model.log_temp.items()}
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
