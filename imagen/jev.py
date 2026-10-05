import os
import time
from typing import Any, Dict, List, Optional

import httpx

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
RETRY_STATUS = (429, 529, 502, 503)


def noul(instructions: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: Dict[str, str]) -> Dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: List[str]) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(criteria)}


class JevError(RuntimeError):
    pass


class Jev:
    """Thin client for POST /v1/systemone. Uses httpx directly so it runs on Python 3.9."""

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL, endpoint: str = ENDPOINT,
                 timeout: float = 60.0, retries: int = 4, transport=None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("set TYPESAFE_API_KEY or pass api_key")
        self.model = model
        self.endpoint = endpoint
        self.retries = retries
        self.client = httpx.Client(timeout=timeout, transport=transport,
                                   headers={"Authorization": f"Bearer {self.api_key}",
                                            "Content-Type": "application/json"})
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def ask(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        body = {"state": state, "model": self.model, "questions": questions}
        delay = 1.0
        for attempt in range(self.retries + 1):
            r = self.client.post(self.endpoint, json=body)
            if r.status_code in RETRY_STATUS and attempt < self.retries:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                raise JevError(f"{r.status_code}: {r.text[:500]}")
            data = r.json()
            u = data.get("usage", {})
            self.usage["input_tokens"] += int(u.get("input_tokens", 0))
            self.usage["output_tokens"] += int(u.get("output_tokens", 0))
            self.usage["calls"] += 1
            return data["answers"]
        raise JevError("unreachable")
