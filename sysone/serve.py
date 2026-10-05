import argparse
import base64
import io
from typing import Any, Dict, List

from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from .model import SystemOneVL
from .questions import Query, question_from_dict

app = FastAPI(title="sysone-vl")
model = None


class DecideRequest(BaseModel):
    state: str
    images: List[str] = []
    questions: List[Dict[str, Any]]


def decode_image(s):
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


@app.post("/v1/decide")
def decide(req: DecideRequest):
    q = Query(req.state, [decode_image(s) for s in req.images], [question_from_dict(d) for d in req.questions])
    return {"decisions": model.decide([q])[0]}


def main():
    global model
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rlcd")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    model = SystemOneVL.load(a.ckpt, gradient_checkpointing=False).cuda().eval()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
