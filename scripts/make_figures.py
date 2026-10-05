import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sysone.envs import tabletop  # noqa: E402
from sysone.envs.tabletop import CLUTTER, RADII, REACH, SIZE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

BG = "#ffffff"
PANEL = "#ffffff"
GRID = "#d9dde3"
TEXT = "#20242a"
MUTED = "#68717d"
C_SFT = "#d55e00"
C_RLCD = "#0072b2"
C_TRUE = "#4d5968"
C_SLOT = "#7b5ea7"
C_BAD = "#c43b3b"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
})


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_json(p):
    with open(p) as f:
        return json.load(f)


def softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def grasp_pred(preds):
    p = np.array([sigmoid(r["logits"][2][0]) for r in preds])
    t = np.array([r["meta"]["p_grasp"] for r in preds])
    return p, t


def equal_count_bins(p, t, n_bins=10):
    order = np.argsort(p)
    chunks = np.array_split(order, n_bins)
    xs = np.array([p[c].mean() for c in chunks])
    ys = np.array([t[c].mean() for c in chunks])
    ns = np.array([len(c) for c in chunks])
    return xs, ys, ns


def fig_reliability(sft, rlcd, sft_eval, rlcd_eval):
    fig = plt.figure(figsize=(13.8, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.28,
                          left=0.06, right=0.985, top=0.78, bottom=0.14)
    specs = ((sft, sft_eval, "SFT: trained on hard 0/1 labels", C_SFT),
             (rlcd, rlcd_eval, "SFT + RLCD: trained on outcomes", C_RLCD))
    for k, (preds, ev, name, color) in enumerate(specs):
        ax = fig.add_subplot(gs[k])
        p, t = grasp_pred(preds)
        xs, ys, ns = equal_count_bins(p, t)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
        ax.scatter(p, t, s=7, alpha=0.22, color=color, edgecolors="none")
        ax.plot(xs, ys, "o-", color=color, lw=2, ms=6, mec=BG)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{n}", (x, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5, color=MUTED)
        mae = np.abs(p - t).mean()
        ax.text(0.04, 0.96, f"mean |p - true p| = {mae:.3f}\nECE vs outcomes = {ev['grasp']['ece_vs_outcomes']:.3f}",
                transform=ax.transAxes, va="top", fontsize=10, color=TEXT,
                bbox=dict(boxstyle="square,pad=0.4", fc=BG, ec=GRID, lw=0.8))
        ax.set_title(name, fontsize=11.5, color=color, pad=8)
        ax.set_xlabel("predicted P(grasp succeeds)")
        if k == 0:
            ax.set_ylabel("true P(grasp succeeds)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.5, lw=0.5)
    ax = fig.add_subplot(gs[2])
    p_sft, _ = grasp_pred(sft)
    p_rlcd, t = grasp_pred(rlcd)
    ax.hist(p_sft, bins=20, range=(0, 1), alpha=0.6, color=C_SFT, label="SFT predictions")
    ax.hist(p_rlcd, bins=20, range=(0, 1), alpha=0.6, color=C_RLCD, label="SFT + RLCD predictions")
    ax.hist(t, bins=20, range=(0, 1), histtype="step", color=C_TRUE, lw=1.8, label="true p (environment)")
    ax.set_title("what each model believes vs what is true", fontsize=11.5, color=TEXT, pad=8)
    ax.set_xlabel("P(grasp succeeds)")
    ax.set_ylabel("test scenes")
    ax.grid(alpha=0.5, lw=0.5)
    ax.legend(fontsize=9, frameon=False, labelcolor=TEXT)
    fig.suptitle("RLCD makes the grasp probability mean what it says",
                 fontsize=16, color=TEXT, y=0.965)
    fig.text(0.5, 0.875, "500 unseen tabletop scenes, dots are equal-count bins of 50 with bin counts above them",
             ha="center", fontsize=10, color=MUTED)
    fig.savefig(os.path.join(DOCS, "reliability.png"), dpi=150)
    plt.close(fig)


def fig_before_after(sft_eval, rlcd_eval):
    rows = [
        ("choice accuracy", sft_eval["choice"]["acc"], rlcd_eval["choice"]["acc"], "higher is better"),
        ("reach accuracy", sft_eval["reach"]["acc"], rlcd_eval["reach"]["acc"], "higher is better"),
        ("grasp ECE vs outcomes", sft_eval["grasp"]["ece_vs_outcomes"], rlcd_eval["grasp"]["ece_vs_outcomes"], "lower is better"),
        ("grasp abs err vs true p", sft_eval["grasp"]["mae_vs_true_p"], rlcd_eval["grasp"]["mae_vs_true_p"], "lower is better"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    fig.subplots_adjust(left=0.20, right=0.86, top=0.76, bottom=0.06)
    ax.axis("off")
    for i, (name, a, b, direction) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = min(a, b), max(a, b)
        pad = (hi - lo) * 1.6 + 1e-9
        def mx(v):
            return 0.10 + 0.72 * (v - (lo - pad)) / (2 * pad)
        xa, xb = mx(a), mx(b)
        ax.plot([xa, xb], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.annotate("", xy=(xb, y), xytext=(xa, y),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
        ax.scatter([xa], [y], s=130, color=C_SFT, zorder=3)
        ax.scatter([xb], [y], s=130, color=C_RLCD, zorder=3)
        ax.text(xa, y + 0.34, f"{a:.3f}", ha="center", fontsize=11, color=C_SFT)
        ax.text(xb, y + 0.34, f"{b:.3f}", ha="center", fontsize=11, color=C_RLCD)
        ax.text(-0.02, y, name, ha="right", va="center", fontsize=12.5, color=TEXT)
        ax.text(1.0, y, direction, ha="left", va="center", fontsize=9.5, color=MUTED, style="italic")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.6, len(rows) + 0.9)
    ly = len(rows) + 0.35
    ax.scatter([0.30], [ly], s=105, color=C_SFT)
    ax.text(0.325, ly, "SFT", va="center", fontsize=11, color=TEXT)
    ax.scatter([0.44], [ly], s=105, color=C_RLCD)
    ax.text(0.465, ly, "SFT + RLCD", va="center", fontsize=11, color=TEXT)
    fig.suptitle("RLCD improves all four headline metrics", fontsize=16, color=TEXT, y=0.96)
    fig.text(0.5, 0.855, "500 held-out scenes, each row on its own scale", ha="center", fontsize=10, color=MUTED)
    fig.savefig(os.path.join(DOCS, "before_after.png"), dpi=150)
    plt.close(fig)


def smooth(x, k=9):
    x = np.asarray(x, float)
    sm = np.convolve(x, np.ones(k) / k, mode="valid")
    return np.arange(k - 1, len(x)), sm


def fig_training(test_rows):
    sft = load_json(os.path.join(ROOT, "runs/sft/train_log.json"))
    rl = load_json(os.path.join(ROOT, "runs/rlcd/train_log.json"))
    ps = np.array([r["meta"]["p_grasp"] for r in test_rows])
    floor = float(np.mean(-(ps * np.log(ps) + (1 - ps) * np.log(1 - ps)))) / 4.0
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))
    fig.subplots_adjust(left=0.055, right=0.94, top=0.74, bottom=0.14, wspace=0.3)

    s = np.array([r["loss"] for r in sft])
    xs, sm = smooth(s)
    axes[0].plot(xs, sm, color="#6f7782", lw=2)
    axes[0].set_title("stage 1: supervised loss", fontsize=11.5)
    axes[0].set_xlabel("optimizer step")

    c = np.array([r["cal"] for r in rl])
    kl = np.array([r["kl"] for r in rl])
    xs, cs = smooth(c)
    axes[1].plot(xs, cs, color=C_RLCD, lw=2, label="calibration loss (proper scoring rule)")
    axes[1].axhline(floor, ls="--", lw=1.4, color=C_TRUE)
    axes[1].text(len(rl) * 0.98, floor + 0.018, f"Bernoulli noise floor: {floor:.3f}",
                 ha="right", fontsize=8.5, color=C_TRUE)
    ax2 = axes[1].twinx()
    xs, kls = smooth(kl)
    ax2.plot(xs, kls, color=C_SLOT, lw=1.4, label="KL to frozen SFT policy")
    ax2.set_ylabel("KL", color=C_SLOT)
    ax2.tick_params(axis="y", colors=C_SLOT)
    ax2.set_facecolor("none")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right", labelcolor=TEXT)
    axes[1].set_title("stage 2: calibration loss on real outcomes", fontsize=11.5)
    axes[1].set_xlabel("rollout batch")

    h = np.array([r["choice_hit"] for r in rl])
    xs, hs = smooth(h)
    axes[2].plot(xs, hs, color=C_SFT, lw=2)
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("stage 2: choice hit rate (10% forced exploration)", fontsize=11.5)
    axes[2].set_xlabel("rollout batch")

    for ax in axes:
        ax.grid(alpha=0.5, lw=0.5)
    fig.suptitle("Stage 2 pulls the calibration loss down toward the Bernoulli noise floor",
                 fontsize=15, color=TEXT, y=0.97)
    fig.text(0.5, 0.02,
             "Noise floor = mean entropy of Bernoulli(true p), spread over the four questions in each query.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(os.path.join(DOCS, "training.png"), dpi=150)
    plt.close(fig)


FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def serif(size, italic=False):
    paths = ["/System/Library/Fonts/Supplemental/Arial Italic.ttf"] if italic else []
    paths += FONT_PATHS
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def lerp(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def draw_chip(d, xy, lines, font, accent, anchor="la"):
    pads, lh = 12, font.size + 8
    w = max(d.textlength(s, font=font) for s in lines) + 2 * pads
    h = lh * len(lines) - 8 + 2 * pads
    x, y = xy
    if anchor == "ra":
        x -= w
    x = min(max(x, 6), SIZE * 2 - w - 6)
    y = min(max(y, 6), SIZE * 2 - h - 6)
    d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255, 238),
                outline=(32, 36, 42, 255), width=2)
    d.rectangle([x, y + 8, x + 5, y + h - 8], fill=accent)
    for k, s in enumerate(lines):
        d.text((x + pads + 4, y + pads - 4 + k * lh), s, font=font, fill=hex_rgb(TEXT))
    return y + h


def render_panel(scene, pred, row, note=None):
    scale = 2
    img = tabletop.render(scene).resize((SIZE * scale, SIZE * scale), Image.NEAREST)
    img = img.convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    f = serif(27)
    f_small = serif(24)
    f_cap = serif(26, italic=True)

    qs = row["questions"]
    pc = softmax(pred["logits"][0])
    pick = int(pc.argmax())
    truth = row["questions"][0]["answer"]
    correct = pick == truth
    p_reach = sigmoid(pred["logits"][1][0])
    p_grasp = sigmoid(pred["logits"][2][0])
    true_p = row["meta"]["p_grasp"]
    ps = softmax(pred["logits"][3])
    lvl = int(ps.argmax())

    gx, gy = scene["gripper"]
    rcol = lerp(hex_rgb(C_BAD), hex_rgb(C_RLCD), p_reach)
    d.ellipse([(gx - REACH) * scale, (gy - REACH) * scale, (gx + REACH) * scale, (gy + REACH) * scale],
              outline=rcol + (255,), width=5)
    draw_chip(d, (SIZE * scale - 10, SIZE * scale - 70), [f"reach p={p_reach:.2f}"],
              f_small, rcol, anchor="ra")

    o = scene["objects"][pick]
    x, y = o["pos"]
    r = RADII[o["size"]] * scale + 10
    box_col = hex_rgb(C_RLCD) if correct else hex_rgb(C_BAD)
    cx, cy = x * scale, y * scale
    for wdt, col in ((9, (14, 16, 19, 255)), (5, box_col + (255,))):
        if o["shape"] == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=wdt)
        elif o["shape"] == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=col, width=wdt)
        else:
            d.polygon([(cx, cy - r - 4), (cx - r - 4, cy + r), (cx + r + 4, cy + r)], outline=col, width=wdt)

    grasp_lines = [f"grasp p={p_grasp:.2f}", f"true p={true_p:.2f}"]
    chip_x = cx + r + 14 if cx < SIZE else cx - r - 14
    anchor = "la" if cx < SIZE else "ra"
    draw_chip(d, (chip_x, cy - 40), grasp_lines, f, hex_rgb(C_SLOT), anchor=anchor)

    draw_chip(d, (10, 10), [f"clutter: {CLUTTER[lvl]}"], f, hex_rgb(C_TRUE))
    if not correct:
        draw_chip(d, (SIZE * 2 - 10, 10), ["wrong pick"], f, hex_rgb(C_BAD), anchor="ra")

    instruction = row["state"].split("\n")[0].replace("instruction: pick up the ", "")
    cap = instruction
    bar = Image.new("RGBA", (SIZE * scale, 64), (255, 255, 255, 255))
    bd = ImageDraw.Draw(bar)
    bd.text((14, 14), cap, font=f_cap, fill=hex_rgb(TEXT))
    out = Image.new("RGBA", (SIZE * scale, SIZE * scale + 64))
    out.paste(img, (0, 0))
    out.paste(bar, (0, SIZE * scale))
    return out.convert("RGB")


def fig_scenes(rlcd, test_rows):
    rng = random.Random(2)
    scenes = []
    for i, row in enumerate(test_rows):
        s = tabletop.sample_scene(rng)
        assert abs(s["p_grasp"] - row["meta"]["p_grasp"]) < 1e-9, f"scene {i} mismatch"
        scenes.append(s)

    picks = [(0, None), (2, None), (6, None), (7, None),
             (52, "poorly calibrated here"),
             (372, "the model picked the wrong object")]
    panels = [render_panel(scenes[i], rlcd[i], test_rows[i], note) for i, note in picks]
    w, h = panels[0].size
    gap = 18
    title_h = 110
    grid = Image.new("RGB", (3 * w + 4 * gap, 2 * h + 3 * gap + title_h), hex_rgb(BG))
    gd = ImageDraw.Draw(grid)
    f_title = serif(44)
    f_sub = serif(28)
    gd.text((gap, 22), "One forward pass per scene, four typed answers drawn on the image",
            font=f_title, fill=hex_rgb(TEXT))
    gd.text((gap, 74), "SFT + RLCD model, unseen test scenes. Bottom middle: poorly calibrated. Bottom right: wrong pick.",
            font=f_sub, fill=hex_rgb(MUTED))
    for k, p in enumerate(panels):
        r, c = divmod(k, 3)
        x = gap + c * (w + gap)
        y = title_h + gap + r * (h + gap)
        grid.paste(p, (x, y))
        gd.rectangle([x, y, x + w - 1, y + h - 1], outline=(0, 0, 0), width=1)
    grid.save(os.path.join(DOCS, "scenes.png"))


def main():
    sft = load_jsonl(os.path.join(ROOT, "runs/sft/preds.jsonl"))
    rlcd = load_jsonl(os.path.join(ROOT, "runs/rlcd/preds.jsonl"))
    test = load_jsonl(os.path.join(ROOT, "data/tabletop/test.jsonl"))
    sft_eval = load_json(os.path.join(ROOT, "runs/sft/eval.json"))
    rlcd_eval = load_json(os.path.join(ROOT, "runs/rlcd/eval.json"))
    fig_reliability(sft, rlcd, sft_eval, rlcd_eval)
    fig_before_after(sft_eval, rlcd_eval)
    fig_training(test)
    fig_scenes(rlcd, test)
    print("wrote", sorted(os.listdir(DOCS)))


if __name__ == "__main__":
    main()
