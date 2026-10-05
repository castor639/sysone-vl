import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw

from ..questions import Choice, Noul, Query, Score

SIZE = 336
COLORS = {"red": (220, 50, 40), "green": (40, 170, 70), "blue": (50, 90, 220), "yellow": (235, 200, 40)}
SHAPES = ("circle", "square", "triangle")
RADII = {"small": 11, "medium": 18, "large": 27}
REACH = 120
CLUTTER = ["clear", "light", "moderate", "heavy"]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def graspability(obj, scene):
    p = 0.92
    if obj["size"] == "small":
        p -= 0.22
    if obj["occluded"]:
        p -= 0.45
    if _dist(obj["pos"], scene["gripper"]) > REACH:
        p -= 0.5
    x, y = obj["pos"]
    if min(x, y, SIZE - x, SIZE - y) < 30:
        p -= 0.1
    return min(max(p, 0.03), 0.97)


def sample_scene(rng: random.Random):
    n = rng.randint(2, 6)
    combos = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(combos)
    objs = []
    tries = 0
    while len(objs) < n and tries < 200:
        tries += 1
        size = rng.choice(list(RADII))
        pos = (rng.randint(30, SIZE - 30), rng.randint(30, SIZE - 30))
        if all(_dist(pos, o["pos"]) > RADII[size] + RADII[o["size"]] + 8 for o in objs):
            color, shape = combos[len(objs)]
            objs.append({"color": color, "shape": shape, "size": size, "pos": pos, "occluded": False})
    gripper = (rng.randint(30, SIZE - 30), rng.randint(30, SIZE - 30))
    kind = rng.choice(["named", "nearest", "largest"])
    if kind == "named":
        t = rng.randrange(len(objs))
        instruction = f"pick up the {objs[t]['color']} {objs[t]['shape']}"
    elif kind == "nearest":
        t = min(range(len(objs)), key=lambda i: _dist(objs[i]["pos"], gripper))
        instruction = "pick up the object closest to the gripper"
    else:
        order = {"small": 0, "medium": 1, "large": 2}
        best = max(order[o["size"]] for o in objs)
        cands = [i for i, o in enumerate(objs) if order[o["size"]] == best]
        if len(cands) > 1:
            kind, t = "named", rng.choice(cands)
            instruction = f"pick up the {objs[t]['color']} {objs[t]['shape']}"
        else:
            t = cands[0]
            instruction = "pick up the largest object"
    occluder = None
    if rng.random() < 0.35:
        ox, oy = objs[t]["pos"]
        r = RADII[objs[t]["size"]]
        occluder = (ox + rng.choice([-1, 1]) * r * 0.7, oy + rng.choice([-1, 1]) * r * 0.5, 22)
        objs[t]["occluded"] = True
    scene = {"objects": objs, "gripper": gripper, "target": t, "instruction": instruction,
             "kind": kind, "occluder": occluder}
    scene["p_grasp"] = graspability(objs[t], scene)
    scene["in_reach"] = _dist(objs[t]["pos"], gripper) <= REACH
    clutter = len(objs) + (2 if occluder else 0)
    scene["clutter"] = 0 if clutter <= 3 else 1 if clutter == 4 else 2 if clutter == 5 else 3
    return scene


def render(scene):
    img = Image.new("RGB", (SIZE, SIZE), (92, 78, 62))
    d = ImageDraw.Draw(img)
    for x in range(0, SIZE, 48):
        d.line([(x, 0), (x, SIZE)], fill=(84, 70, 55), width=1)
        d.line([(0, x), (SIZE, x)], fill=(84, 70, 55), width=1)
    for o in scene["objects"]:
        x, y = o["pos"]
        r = RADII[o["size"]]
        c = COLORS[o["color"]]
        if o["shape"] == "circle":
            d.ellipse([x - r, y - r, x + r, y + r], fill=c, outline=(20, 20, 20))
        elif o["shape"] == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=c, outline=(20, 20, 20))
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=c, outline=(20, 20, 20))
    if scene["occluder"]:
        ox, oy, r = scene["occluder"]
        d.rectangle([ox - r, oy - r, ox + r, oy + r], fill=(110, 110, 110), outline=(30, 30, 30))
    gx, gy = scene["gripper"]
    d.ellipse([gx - REACH, gy - REACH, gx + REACH, gy + REACH], outline=(230, 230, 230), width=1)
    d.line([(gx - 12, gy), (gx + 12, gy)], fill=(255, 255, 255), width=3)
    d.line([(gx, gy - 12), (gx, gy + 12)], fill=(255, 255, 255), width=3)
    return img


def describe(o):
    return f"{o['color']} {o['shape']}"


def make_query(scene, image=None, with_answers=True):
    objs = scene["objects"]
    gx, gy = scene["gripper"]
    state = (f"instruction: {scene['instruction']}\n"
             f"gripper: white cross at pixel ({gx}, {gy}); reach radius {REACH} px shown as the thin white circle\n"
             f"a grey block, if present, is an obstacle resting on top of an object")
    qs = [
        Choice("Which object should the gripper pick?", [describe(o) for o in objs],
               scene["target"] if with_answers else None),
        Noul("The target object lies inside the gripper's reach circle.",
             scene["in_reach"] if with_answers else None),
        Noul("A grasp attempted on the target object right now will succeed.",
             (scene["p_grasp"] >= 0.5) if with_answers else None),
        Score("How cluttered is the workspace?", CLUTTER, [1, 2, 3, 4],
              scene["clutter"] if with_answers else None),
    ]
    return Query(state, [image] if image is not None else [], qs)


def outcome(scene, qi, decision, rng: random.Random):
    if qi == 0:
        return int(decision == scene["target"])
    if qi == 1:
        return int(scene["in_reach"])
    if qi == 2:
        return int(rng.random() < scene["p_grasp"])
    return scene["clutter"]


def outcomes(scene, decisions, rng):
    return [outcome(scene, i, d, rng) for i, d in enumerate(decisions)]


def export(n, out_dir, seed, split):
    rng = random.Random(seed)
    img_dir = os.path.join(out_dir, "images", split)
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i in range(n):
        scene = sample_scene(rng)
        rel = f"images/{split}/{i:05d}.png"
        render(scene).save(os.path.join(out_dir, rel))
        q = make_query(scene)
        rows.append({"images": [rel], "state": q.state,
                     "questions": [_qd(x) for x in q.questions],
                     "meta": {"p_grasp": scene["p_grasp"], "kind": scene["kind"], "n": len(scene["objects"])}})
    with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def _qd(x):
    from ..questions import question_to_dict
    return question_to_dict(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tabletop")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for split, n, s in (("train", a.train, a.seed), ("val", a.val, a.seed + 1), ("test", a.test, a.seed + 2)):
        print(split, export(n, a.out, s, split))
