from dataclasses import dataclass, field
from typing import List, Optional, Union
from PIL import Image

LEAF = "<|fim_pad|>"


@dataclass
class Choice:
    question: str
    options: List[str]
    answer: Optional[int] = None
    type: str = "choice"

    def n_leaves(self):
        return len(self.options)


@dataclass
class Score:
    question: str
    levels: List[str]
    values: Optional[List[float]] = None
    answer: Optional[int] = None
    type: str = "score"

    def n_leaves(self):
        return len(self.levels)


@dataclass
class Noul:
    proposition: str
    answer: Optional[bool] = None
    type: str = "noul"

    def n_leaves(self):
        return 1


Question = Union[Choice, Score, Noul]


@dataclass
class Query:
    state: str
    images: List[Image.Image] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)


SYSTEM = (
    "You are a decision model. You never write text. Read the images and the state, "
    "then judge every candidate marked with a slot token. Be honest about uncertainty."
)


def letter(i):
    return chr(ord("A") + i)


def render_question(q, idx):
    if q.type == "choice":
        lines = [f"Q{idx} choice: {q.question}"]
        for i, o in enumerate(q.options):
            lines.append(f"  {letter(i)}. {o}")
        lines.append("  candidates: " + " ".join(f"{letter(i)}{LEAF}" for i in range(len(q.options))))
        return "\n".join(lines)
    if q.type == "score":
        lines = [f"Q{idx} score: {q.question}"]
        for i, o in enumerate(q.levels):
            lines.append(f"  {i + 1}. {o}")
        lines.append("  levels: " + " ".join(f"{i + 1}{LEAF}" for i in range(len(q.levels))))
        return "\n".join(lines)
    if q.type == "noul":
        return f"Q{idx} noul: {q.proposition}\n  verdict: true{LEAF}"
    raise ValueError(q.type)


def build_messages(query: Query):
    content = [{"type": "image"} for _ in query.images]
    body = "STATE:\n" + query.state.strip() + "\n\n"
    body += "\n\n".join(render_question(q, i + 1) for i, q in enumerate(query.questions))
    content.append({"type": "text", "text": body})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def question_from_dict(d):
    t = d["type"]
    if t == "choice":
        return Choice(d["question"], list(d["options"]), d.get("answer"))
    if t == "score":
        return Score(d["question"], list(d["levels"]), d.get("values"), d.get("answer"))
    if t == "noul":
        return Noul(d["proposition"], d.get("answer"))
    raise ValueError(t)


def question_to_dict(q):
    if q.type == "choice":
        return {"type": "choice", "question": q.question, "options": q.options, "answer": q.answer}
    if q.type == "score":
        return {"type": "score", "question": q.question, "levels": q.levels, "values": q.values, "answer": q.answer}
    return {"type": "noul", "proposition": q.proposition, "answer": q.answer}
