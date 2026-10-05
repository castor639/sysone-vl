import numpy as np
import torch
import torch.nn.functional as F


def supervised_loss(outputs, queries):
    total, n = 0.0, 0
    for per_q, q in zip(outputs, queries):
        for logits, x in zip(per_q, q.questions):
            if x.answer is None:
                continue
            if x.type == "noul":
                target = torch.tensor(float(x.answer), device=logits.device)
                total = total + F.binary_cross_entropy_with_logits(logits[0], target)
            else:
                target = torch.tensor(int(x.answer), device=logits.device)
                total = total + F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            n += 1
    return total / max(n, 1)


def sample_decisions(outputs, queries, explore=0.1, generator=None):
    decisions = []
    for per_q, q in zip(outputs, queries):
        row = []
        for logits, x in zip(per_q, q.questions):
            if x.type == "noul":
                row.append(None)
            elif torch.rand((), generator=generator).item() < explore:
                row.append(int(torch.randint(len(logits), (1,), generator=generator)))
            else:
                row.append(int(torch.multinomial(F.softmax(logits.detach(), -1).cpu(), 1, generator=generator)))
        decisions.append(row)
    return decisions


def _bernoulli_kl(logit_p, logit_q):
    p = torch.sigmoid(logit_p)
    return p * (F.logsigmoid(logit_p) - F.logsigmoid(logit_q)) + (1 - p) * (F.logsigmoid(-logit_p) - F.logsigmoid(-logit_q))


def rlcd_loss(outputs, ref_outputs, queries, decisions, outcomes, beta=0.05, ent_coef=0.01):
    cal, kl, ent, n = 0.0, 0.0, 0.0, 0
    for i, (per_q, q) in enumerate(zip(outputs, queries)):
        for j, (logits, x) in enumerate(zip(per_q, q.questions)):
            r = outcomes[i][j]
            if r is None:
                continue
            ref = ref_outputs[i][j] if ref_outputs is not None else None
            if x.type == "noul":
                target = torch.tensor(float(r), device=logits.device)
                cal = cal + F.binary_cross_entropy_with_logits(logits[0], target)
                if ref is not None:
                    kl = kl + _bernoulli_kl(logits[0], ref[0].detach())
                p = torch.sigmoid(logits[0])
                ent = ent - (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
            else:
                logp = F.log_softmax(logits, -1)
                if x.type == "choice":
                    a = decisions[i][j]
                    p_a = logp[a].exp().clamp(1e-6, 1 - 1e-6)
                    cal = cal - (float(r) * torch.log(p_a) + (1 - float(r)) * torch.log1p(-p_a))
                else:
                    cal = cal - logp[int(r)]
                if ref is not None:
                    ref_logp = F.log_softmax(ref.detach(), -1)
                    kl = kl + (logp.exp() * (logp - ref_logp)).sum()
                ent = ent - (logp.exp() * logp).sum()
            n += 1
    n = max(n, 1)
    loss = cal / n + beta * kl / n - ent_coef * ent / n
    return loss, {"cal": float(cal) / n, "kl": float(kl) / n, "ent": float(ent) / n}


def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def brier(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2)) if len(conf) else float("nan")
