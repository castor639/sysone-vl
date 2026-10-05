"""Chat UI: type a request, watch Jev decide the picture level by level, round by round.
Events stream as newline-delimited JSON over a single POST."""
import asyncio
import base64
import io
import json
import logging
import os
import queue
import threading
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import jevpix
from .jev import Jev, JevError

app = FastAPI(title="imagen")
components = {"jev": None}
lock = threading.Lock()
OUT_ROOT = os.environ.get("IMAGEN_OUT", "out/ui")
log = logging.getLogger("uvicorn.error")


class Cancelled(Exception):
    pass


class GenRequest(BaseModel):
    prompt: str
    grid: int = 32       # final canvas is grid x grid cells
    rounds: int = 2      # commit rounds per level
    seed: int = 0
    render: str = "soft"  # "soft": probability-blended colors, "hard": palette colors


def b64(im, fmt="PNG"):
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def get_jev():
    if components["jev"] is None:
        components["jev"] = Jev()
    return components["jev"]


def work(req: GenRequest, emit):
    t0 = time.time()
    out = os.path.join(OUT_ROOT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    jev = get_jev()
    log.info("generate %r grid=%d rounds=%d render=%s", req.prompt, req.grid, req.rounds, req.render)
    emit({"type": "status", "text": "asking Jev for the layout"})

    def on_plan(plan):
        emit({"type": "plan", **plan.to_dict()})
        emit({"type": "start", "grid": req.grid})

    def on_round(info, im):
        log.info("  level %d/%d %dx%d round %d/%d: %d/%d decided, %.1fs", info["level"], info["levels"], info["grid"],
                 info["grid"], info["round"], info["rounds"], info["committed"], info["cells"], time.time() - t0)
        emit({"type": "frame", "img": b64(im), **info})

    im, res = jevpix.generate(jev, req.prompt, size=req.grid, rounds=req.rounds, seed=req.seed, render=req.render,
                              on_round=on_round, on_plan=on_plan)
    im.save(os.path.join(out, "best.png"))
    emit({"type": "final", "img": b64(im)})
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump({"request": req.prompt, "seconds": time.time() - t0, **res}, f, indent=2)
    log.info("  done in %.1fs, %s", time.time() - t0, res["usage"])
    emit({"type": "done", "out": out, "seconds": round(time.time() - t0, 1), "usage": res["usage"]})


@app.post("/generate")
async def generate(req: GenRequest):
    q: "queue.Queue[Optional[dict]]" = queue.Queue()
    cancel = threading.Event()

    def emit(ev):
        if cancel.is_set():  # the browser went away: stop at the next event instead of finishing for nobody
            raise Cancelled()
        q.put(ev)

    def run():
        if not lock.acquire(blocking=False):
            q.put({"type": "status", "text": "waiting for the previous generation to finish"})
            lock.acquire()
        try:
            if not cancel.is_set():
                work(req, emit)
        except Cancelled:
            log.info("  cancelled: client disconnected")
        except JevError as e:
            q.put({"type": "error", "text": f"Jev: {e}"})
        except Exception as e:  # surfaced in the chat instead of a dead stream
            log.exception("generate failed")
            q.put({"type": "error", "text": f"{type(e).__name__}: {e}"})
        finally:
            lock.release()
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        try:
            while True:
                ev = await asyncio.to_thread(q.get)
                if ev is None:
                    break
                yield json.dumps(ev) + "\n"
        finally:
            cancel.set()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "ui.html")) as f:
        return f.read()


def main():
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
