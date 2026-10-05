import random

import pytest

from sysone.envs import tabletop
from sysone.questions import LEAF, build_messages


@pytest.mark.slow
def test_leaf_token_survives_tokenization():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    leaf_id = proc.tokenizer.convert_tokens_to_ids(LEAF)
    s = tabletop.sample_scene(random.Random(3))
    q = tabletop.make_query(s, tabletop.render(s))
    text = proc.apply_chat_template(build_messages(q), tokenize=False, add_generation_prompt=True)
    ids = proc(text=[text], images=q.images, return_tensors="pt")["input_ids"][0]
    assert int((ids == leaf_id).sum()) == sum(x.n_leaves() for x in q.questions)
