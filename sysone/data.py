import json
import os

from PIL import Image
from torch.utils.data import Dataset

from .questions import Query, question_from_dict


class JsonlQueries(Dataset):
    def __init__(self, path, limit=None):
        self.root = os.path.dirname(os.path.abspath(path))
        with open(path) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        images = [Image.open(os.path.join(self.root, p)).convert("RGB") for p in r.get("images", [])]
        return Query(r["state"], images, [question_from_dict(q) for q in r["questions"]])


def collate(batch):
    return list(batch)
