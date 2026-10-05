import random

from sysone.envs import tabletop
from sysone.questions import LEAF, build_messages


def test_scene_and_query():
    rng = random.Random(0)
    for _ in range(50):
        s = tabletop.sample_scene(rng)
        img = tabletop.render(s)
        assert img.size == (tabletop.SIZE, tabletop.SIZE)
        q = tabletop.make_query(s, img)
        text = build_messages(q)[1]["content"][-1]["text"]
        assert text.count(LEAF) == sum(x.n_leaves() for x in q.questions)
        assert q.questions[0].answer == s["target"]
        assert 0 <= s["p_grasp"] <= 1
