import argparse
import json
import os
import time

from . import jevpix
from .jev import Jev


def main():
    ap = argparse.ArgumentParser(description="Generate an image with Jev alone")
    ap.add_argument("request")
    ap.add_argument("--grid", type=int, default=32, help="final canvas side in cells (16 to 128)")
    ap.add_argument("--render", choices=["soft", "hard"], default="soft")
    ap.add_argument("--rounds", type=int, default=2, help="commit rounds per level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/imagen")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jev = Jev()
    t0 = time.time()

    def on_plan(plan):
        print("plan:", json.dumps(plan.to_dict()), flush=True)

    def on_round(info, im):
        im.save(os.path.join(a.out, f"round_{info['round']:02d}.png"))
        print(f"level {info['level']}/{info['levels']} ({info['grid']}x{info['grid']}) round {info['round']}/{info['rounds']}: "
              f"{info['committed']}/{info['cells']} committed, {info['asked']} asked so far, {jev.usage['input_tokens']} tokens, {time.time() - t0:.1f}s", flush=True)

    im, res = jevpix.generate(jev, a.request, size=a.grid, rounds=a.rounds, seed=a.seed, render=a.render,
                              on_round=on_round, on_plan=on_plan)
    im.save(os.path.join(a.out, "best.png"))
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump({"request": a.request, "seconds": time.time() - t0, **res}, f, indent=2)
    print("\n".join(res["grid"]))
    print(f"done in {time.time() - t0:.1f}s, usage {res['usage']} -> {a.out}/best.png")


if __name__ == "__main__":
    main()
