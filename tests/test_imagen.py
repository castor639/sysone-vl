import json

import httpx

from imagen import jevpix
from imagen.jev import Jev


def fake_transport(handler):
    def h(req: httpx.Request):
        body = json.loads(req.content)
        assert req.headers["authorization"] == "Bearer test-key"
        assert body["model"] == "jev-latest"
        return httpx.Response(200, json={"model": "jev-latest", "answers": handler(body),
                                         "usage": {"input_tokens": 1000, "output_tokens": 5}})
    return httpx.MockTransport(h)


def fake_jev(body, parts=(), shape="circle"):
    """A sun in the sky over grass. Layout answers are fixed; pixel answers follow the provisional color
    with a confidence that is high inside regions, lower at edges."""
    qs = body["questions"]
    ch = lambda v: {"type": "choice", "choice": v, "probabilities": {v: 0.9}, "confidence": 0.9}
    sc = lambda v: {"type": "score", "score": v, "probabilities": {}, "legend": {}, "confidence": 0.8}
    if "subject" in qs:                      # call 1
        assert "sun" in qs["subject"]["criteria"] and "none" in qs["subject"]["criteria"]
        return {"subject": ch("sun"), "split": {"type": "noul", "noul": 0.9}}
    if any(k.startswith("part_") for k in qs):   # call 2
        assert "part_sun" not in qs and "noun_sky" in qs
        return {k: {"type": "noul", "noul": 0.9 if k[5:] in parts else 0.1} for k in qs}
    if "bg_top" in qs:                       # call 3
        assert qs["shape"]["instructions"] == "Silhouette of the sun"
        out = {"bg_top": ch("sky blue"), "bg_bottom": ch("light green"), "horizon": sc(3.0), "shape": ch(shape),
               "tilt": sc(2.0), "aspect": sc(0.0), "x": sc(3.0), "y": sc(1.0), "size": sc(2.0), "color": ch("yellow"),
               "edge": ch("orange"), "glow": {"type": "noul", "noul": 0.8}}
        for i in range(12):
            out[f"reach_{i}"] = sc(4.0 if i in (0, 3, 6, 9) else 1.0)   # a plus sign outline
        i = 0
        while f"p{i}_rel" in qs:
            name = qs[f"p{i}_color"]["instructions"].replace("Dominant fill color of the ", "")
            assert name in parts and qs[f"p{i}_rel"]["instructions"] == f"Where the {name} sits relative to the sun"
            out.update({f"p{i}_rel": ch("below"), f"p{i}_side": sc(0.0 if i == 0 else 4.0), f"p{i}_shape": ch("rectangle"),
                        f"p{i}_tilt": sc(2.0), f"p{i}_aspect": sc(2.0), f"p{i}_size": sc(1.0), f"p{i}_color": ch("cream")})
            i += 1
        return out
    out = {}
    for qid, q in qs.items():
        assert q["type"] == "choice" and "Provisional color" in q["instructions"] and "Neighbors" in q["instructions"]
        prov = q["instructions"].split("Provisional color ")[1].split(".")[0]
        conf = 0.4 if "edge" in q["instructions"] else 0.9
        out[qid] = {"type": "choice", "choice": prov, "probabilities": {prov: 0.7, "gray": 0.3}, "confidence": conf}
    return out


def test_layout_plan():
    jev = Jev(api_key="test-key", transport=fake_transport(fake_jev))
    plan = jevpix.make_plan(jev, "a sun in the sky over grass")
    assert plan.subject == "sun" and plan.shape == "circle" and plan.color == "Y" and plan.edge == "O"
    assert plan.split and abs(plan.horizon - 0.75) < 1e-6 and plan.glow and plan.parts == []
    assert abs(plan.size - 0.28) < 1e-6 and plan.tilt == 0.0 and plan.aspect == 1.0
    assert abs(plan.x - 0.68) < 1e-6 and abs(plan.y - 0.32) < 1e-6   # 0.75, 0.25 nudged in so the disc fits
    assert jev.usage["calls"] == 3


def test_parts_are_picked_and_placed():
    jev = Jev(api_key="test-key", transport=fake_transport(lambda b: fake_jev(b, parts=("grass", "sky"))))
    plan = jevpix.make_plan(jev, "a sun in the sky over grass")
    assert [p.name for p in plan.parts] == ["sky", "grass"]        # candidate order, both above 0.5
    p = plan.parts[1]
    assert p.rel == "below" and abs(p.size - plan.size * 0.55) < 1e-6 and p.shape == "rectangle" and p.aspect == 2.0
    x, y, r = jevpix.part_geometry(plan, p)
    assert x > plan.x and y > plan.y + plan.size          # below, toward the right end
    assert jevpix.region(plan, x, y, 1 / 32)[1] == "C" and "grass" in jevpix.region(plan, x, y, 1 / 32)[0]


def test_part_hugs_the_real_edge():
    plan = jevpix.Plan("S", "D", True, 0.5, "dog", "ellipse", 0.5, 0.5, 0.25, "N", "N", False, {}, tilt=90.0, aspect=2.5)
    head = jevpix.Part("head", "above", 0.0, "circle", 0.0, 1.0, 0.08, "N")
    plan.parts.append(head)
    x, y, r = jevpix.part_geometry(plan, head)
    body_top = 0.5 - 0.25 / 2.5                        # lying ellipse: vertical half-extent is size / aspect
    assert abs((y + r) - body_top) < 0.03                # head sits on the body, not floating a full radius above
    legs = jevpix.Part("legs", "below", 0.6, "rectangle", 0.0, 2.0, 0.06, "N")
    x2, y2, r2 = jevpix.part_geometry(plan, legs)
    assert x2 > plan.x and y2 - r2 < 0.5 + 0.25 / 2.5 + 0.02


def test_shapes_and_outline():
    ins = jevpix.shape_inside
    assert ins("crescent", -0.5, 0) and not ins("crescent", 0.5, 0)            # horns open to the right
    assert ins("crescent", -0.5, 0, aspect=3.0)                                   # elongation does not squeeze it
    assert ins("crescent", 0, -0.5, tilt=90) and not ins("crescent", 0, 0.5, tilt=90)   # rotated: bite moves down
    assert ins("ellipse", 0, 0.9, aspect=3.0) and not ins("ellipse", 0.9, 0, aspect=3.0)   # long axis vertical
    assert ins("ellipse", 0.9, 0, aspect=3.0, tilt=90)                            # lying on its side
    assert ins("star", 0, -0.95) and not ins("star", 0.6, -0.6)                   # a point up, gap between points
    assert ins("heart", 0, 0.3) and ins("heart", 0.45, -0.45) and not ins("heart", 0, -0.7)   # notch at the top
    assert ins("ring", 0.8, 0) and not ins("ring", 0, 0)
    assert ins("arch", 0, -0.8) and not ins("arch", 0, 0) and ins("arch", 0.75, 0.5)
    assert ins("capsule", 0, 0.9, aspect=2.0) and not ins("capsule", 0.6, 0, aspect=2.0)
    plus = [1.0 if i in (0, 3, 6, 9) else 0.35 for i in range(12)]
    assert ins("outline", 0, -0.9, outline=plus) and ins("outline", 0.9, 0, outline=plus)
    assert not ins("outline", 0.6, -0.6, outline=plus)


def test_edge_band_from_sampling():
    jev = Jev(api_key="test-key", transport=fake_transport(lambda b: fake_jev(b, shape="star")))
    plan = jevpix.make_plan(jev, "a sun in the sky over grass")
    assert plan.shape == "star"
    assert jevpix.region(plan, plan.x, plan.y, 1 / 64)[0] == "inside the sun, well within it"
    tip = (plan.x, plan.y - plan.size * 0.97)
    assert jevpix.region(plan, tip[0], tip[1], 1 / 64)[0] == "inside the sun, at its edge"


def test_fit_composite_keeps_part_in_frame():
    for size in (0.33, 0.30, 0.45):
        plan = jevpix.Plan("K", "D", True, 0.5, "mushroom", "blob", 0.5, 0.75, size, "R", "C", False, {})
        plan.parts.append(jevpix.Part("stem", "below", 0.0, "rectangle", 0.0, 2.0, size * 0.55, "C"))
        jevpix.fit_composite(plan)
        x, y, r = jevpix.part_geometry(plan, plan.parts[0])
        assert y + r <= 0.97 and plan.y - plan.size >= 0.03, (size, plan.y, plan.size, y, r)
    assert jevpix.region(plan, x, y, 1 / 64)[1] == "C"


def test_regions_follow_geometry():
    jev = Jev(api_key="test-key", transport=fake_transport(fake_jev))
    plan = jevpix.make_plan(jev, "a sun in the sky over grass")
    assert jevpix.region(plan, plan.x, plan.y, 1 / 32)[1] == "Y"      # sun centre
    assert jevpix.region(plan, 0.1, 0.1, 1 / 32)[1] == "S"            # sky
    assert jevpix.region(plan, 0.1, 0.9, 1 / 32)[1] == "L"            # grass
    assert "glow" in jevpix.region(plan, plan.x, plan.y + plan.size * 1.1, 1 / 32)[0]


def test_generate_coarse_to_fine_and_spread():
    jev = Jev(api_key="test-key", transport=fake_transport(fake_jev))
    frames = []
    im, res = jevpix.generate(jev, "a sun in the sky over grass", size=32, rounds=2, chunk=100, workers=2,
                              on_round=lambda info, im: frames.append(info))
    assert [f["grid"] for f in frames] == [8, 8, 16, 16, 32, 32]
    assert frames[-1]["committed"] == 32 * 32 and frames[0]["committed"] < 64
    assert im.size == (512, 512) and len(res["grid"]) == 32
    g = res["grid"]
    assert g[8][24] == "Y" and g[2][2] == "S" and g[30][2] == "L"      # sun, sky, grass
    assert res["usage"]["calls"] >= 9 and res["usage"]["input_tokens"] == 1000 * res["usage"]["calls"]
    assert res["usage"]["cells_asked"] < 64 + 256 + 1024                # interior cells inherited, not asked
    # soft render blends: the sky cell is 70% sky blue + 30% gray in the fake, so it is darker than pure sky blue
    sky = im.getpixel((2 * 16, 2 * 16))
    assert sky != jevpix.RGB["S"] and abs(sky[0] - (0.7 * 135 + 0.3 * 128)) < 3
    hard, _ = jevpix.generate(jev, "a sun in the sky over grass", size=16, rounds=1, render="hard")
    assert hard.getpixel((8, 8)) == jevpix.RGB["S"]


def test_cropped_state_for_big_canvas():
    jev = Jev(api_key="test-key", transport=fake_transport(fake_jev))
    plan = jevpix.make_plan(jev, "a sun in the sky over grass")
    n = 64
    grid = [[None] * n for _ in range(n)]
    prov = jevpix.plan_grid(plan, n)
    full = jevpix.render_state("x", plan, grid, prov)
    band = jevpix.render_state("x", plan, grid, prov, rows=(20, 24))
    assert full.count("\n") > band.count("\n") and "rows 017 to 027" in band and "063 " in full


def test_spread_commit_avoids_adjacent():
    rng = __import__("random").Random(0)
    cells = [(r, c) for r in range(4) for c in range(4)]
    answers = {rc: {"confidence": 0.9} for rc in cells}
    picked = jevpix.spread_commit(cells, answers, 8, rng)
    assert len(picked) == 8
    s = set(picked)
    adjacent = sum(1 for r, c in picked if (r, c + 1) in s or (r + 1, c) in s)
    assert adjacent == 0


def test_retry_on_429(monkeypatch):
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {}})
    jev = Jev(api_key="test-key", transport=httpx.MockTransport(h), retries=1)
    monkeypatch.setattr("imagen.jev.time.sleep", lambda s: None)
    assert jev.ask("x", {"q": {"type": "noul", "instructions": "y"}})["q"]["noul"] == 0.5
    assert calls["n"] == 2


def test_app_streams_rounds(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from imagen import app as A
    monkeypatch.setattr(A, "OUT_ROOT", str(tmp_path))
    A.components["jev"] = Jev(api_key="test-key", transport=fake_transport(fake_jev))
    with TestClient(A.app) as c:
        assert "<title>Jev Imagen</title>" in c.get("/").text
        r = c.post("/generate", json={"prompt": "a sun in the sky over grass", "grid": 16, "rounds": 2})
    events = [json.loads(l) for l in r.text.splitlines()]
    types = [e["type"] for e in events]
    assert types[:3] == ["status", "plan", "start"] and types.count("frame") == 6 and types[-2:] == ["final", "done"]
    assert list(tmp_path.glob("*/best.png")) and list(tmp_path.glob("*/report.json"))


def test_app_reports_queueing_and_cancels_on_disconnect(tmp_path, monkeypatch):
    import threading
    from fastapi.testclient import TestClient
    from imagen import app as A
    monkeypatch.setattr(A, "OUT_ROOT", str(tmp_path))
    A.components["jev"] = Jev(api_key="test-key", transport=fake_transport(fake_jev))
    # hold the lock from outside: the request must say it is waiting, then run once released
    A.lock.acquire()
    with TestClient(A.app) as c:
        t = threading.Thread(target=lambda: (__import__("time").sleep(0.3), A.lock.release()))
        t.start()
        r = c.post("/generate", json={"prompt": "a sun in the sky over grass", "grid": 16, "rounds": 1})
        t.join()
    events = [json.loads(l) for l in r.text.splitlines()]
    assert events[0] == {"type": "status", "text": "waiting for the previous generation to finish"}
    assert events[-1]["type"] == "done"
    assert not A.lock.locked()
