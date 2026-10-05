"""Image generation with Jev alone: every pixel is a Choice answered by Jev.

Stage 1, layout: one request of typed questions decides the background, whether there is a
horizon, the main subject, its shape, position, size and colors. Code turns that into a plan.

Stage 2, pixels, coarse to fine: the canvas is decided at 1/4, then 1/2, then full resolution.
Each cell is a Choice over a 16-color palette whose question carries what code can compute and
Jev cannot: where the cell sits in words, which region of the plan it falls in, the provisional
color from the coarser level, and its neighbors. Every level runs a few rounds; each round asks
all undecided cells at once (they share one state), commits a confidence-weighted spread of
them, and re-asks the rest with the updated canvas. Undecided cells are drawn as the
probability blend of the palette, so the picture sharpens like a denoising process.
"""
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from .jev import Jev, JevError, choice, noul, score

PALETTE: List[Tuple[str, str, Tuple[int, int, int]]] = [
    ("K", "black", (20, 20, 24)),
    ("W", "white", (245, 245, 245)),
    ("G", "gray", (128, 128, 128)),
    ("R", "red", (210, 40, 40)),
    ("O", "orange", (240, 140, 30)),
    ("Y", "yellow", (245, 215, 50)),
    ("L", "light green", (120, 200, 90)),
    ("D", "dark green", (30, 110, 50)),
    ("S", "sky blue", (135, 195, 240)),
    ("B", "blue", (40, 80, 200)),
    ("N", "brown", (120, 75, 40)),
    ("P", "pink", (240, 150, 190)),
    ("C", "cream", (245, 230, 190)),
    ("M", "dark red", (130, 20, 30)),
    ("V", "navy", (15, 25, 70)),
    ("H", "light gray", (200, 200, 205)),
]
LETTERS = [p[0] for p in PALETTE]
NAMES = {p[0]: p[1] for p in PALETTE}
RGB = {p[0]: p[2] for p in PALETTE}
NAME_TO_LETTER = {p[1]: p[0] for p in PALETTE}
CRITERIA = {p[1]: f"cell is {p[1]}" for p in PALETTE}
SHAPES = {
    "circle": "round disc", "ellipse": "oval, longer in one direction", "capsule": "pill, a rounded bar",
    "rectangle": "box with straight sides", "triangle": "triangle pointing up", "crescent": "banana or moon shape, curved with two horns",
    "star": "five-pointed star", "heart": "heart shape", "leaf": "pointed oval, like a leaf or an eye",
    "ring": "hollow ring or donut", "arch": "upside-down U, an arch or a table", "blob": "irregular rounded lump",
    "outline": "none of these; the outline will be described direction by direction",
}
TILTS = ["lying flat, horizontal, its top end pointing left", "diagonal, leaning left", "standing vertical, upright",
         "diagonal, leaning right", "lying flat, horizontal, its top end pointing right"]
TILT_DEG = [-90, -45, 0, 45, 90]
ELONG = ["as wide as tall", "slightly longer than wide", "twice as long as wide", "three times as long as wide", "very long and thin"]
ELONG_RATIO = [1.0, 1.35, 2.0, 3.0, 4.5]
POS = ["far left", "left of center", "center", "right of center", "far right"]
POSY = ["top", "above center", "center", "below center", "bottom"]
SIZES = ["tiny", "small", "medium", "large", "fills most of the frame"]
SIZE_FRAC = [0.10, 0.18, 0.28, 0.40, 0.60]
HORIZON = ["very high, ground fills most of the frame", "high", "middle", "low", "very low, sky fills most of the frame"]
REL = {"below": "directly below it, touching it", "above": "directly above it, touching it",
       "left": "to its left, touching it", "right": "to its right, touching it",
       "inside": "on top of it, inside its outline", "around": "surrounding it"}
SIDE = ["at its far left end", "toward its left", "in its middle", "toward its right", "at its far right end"]
SIDE_OFF = [-0.8, -0.4, 0.0, 0.4, 0.8]
PART_SIZES = ["much smaller than it", "about half its size", "about the same size", "larger than it"]
PART_FRAC = [0.35, 0.55, 1.0, 1.4]
CLOCK = ["12 o'clock, straight up", "1 o'clock", "2 o'clock", "3 o'clock, straight right", "4 o'clock", "5 o'clock",
         "6 o'clock, straight down", "7 o'clock", "8 o'clock", "9 o'clock, straight left", "10 o'clock", "11 o'clock"]
REACH = ["not at all, the outline is close to the center there", "a quarter of its size", "half of its size",
         "three quarters of its size", "its full size"]
REACH_FRAC = [0.15, 0.35, 0.55, 0.8, 1.0]
MAX_PARTS = 3
STOP = set("a an the of in on at to with and or for from by is are was were be into onto over under image picture photo "
           "drawing painting pixel art style scene some very big small little colored color against".split())


# ----------------------------------------------------------------------------- stage 1: layout

@dataclass
class Part:
    name: str
    rel: str
    side: float         # lateral offset along the subject, -1 .. 1
    shape: str
    tilt: float         # degrees, clockwise positive
    aspect: float       # length / width
    size: float         # radius, fraction of the canvas side
    color: str


@dataclass
class Plan:
    bg_top: str
    bg_bottom: str
    split: bool
    horizon: float           # 0 top .. 1 bottom
    subject: Optional[str]
    shape: str
    x: float
    y: float
    size: float              # radius as a fraction of the canvas side
    color: str
    edge: str
    glow: bool
    answers: dict
    tilt: float = 0.0
    aspect: float = 1.0
    outline: Optional[List[float]] = None   # 12 radii, fraction of size, clockwise from 12 o'clock
    parts: List[Part] = field(default_factory=list)

    def to_dict(self):
        d = dict(self.__dict__)
        d.pop("answers")
        d["parts"] = [dict(p.__dict__) for p in self.parts]
        return d


def subject_candidates(request: str) -> List[str]:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z-]+", request.lower()) if w not in STOP]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(w)
    return out[:12] or [request.strip()]


def pick_questions(request: str):
    """Call 1: what the picture is about."""
    cands = {w: f"the {w}" for w in subject_candidates(request)}
    cands["none"] = "no single subject, the picture is a texture, pattern or landscape without one main object"
    return {
        "subject": choice("Which is the one main object the picture is about", cands),
        "split": noul("The picture has a distinct ground, floor or water below and a sky or wall above"),
    }


def part_questions(request: str, subject: str):
    """Call 2: two Nouls per candidate word. Sibling questions cannot see each other, so this is asked per word
    rather than as 'pick three', which would return the same word three times."""
    qs = {}
    for w in subject_candidates(request):
        if w == subject:
            continue
        qs[f"part_{w}"] = noul(f"The {w} is a distinct visible thing or body part attached to or right next to the {subject}, "
                               f"worth drawing as its own shape")
        qs[f"noun_{w}"] = noul(f"The word '{w}' names a physical object or a body part, not a color, material, texture, "
                               f"quality or action")
    return qs


def shape_questions(prefix: str, name: str):
    return {
        f"{prefix}shape": choice(f"Silhouette of the {name}", SHAPES),
        f"{prefix}tilt": score(f"Orientation of the {name}", TILTS),
        f"{prefix}aspect": score(f"How elongated the {name} is", ELONG),
        f"{prefix}color": choice(f"Dominant fill color of the {name}", CRITERIA),
    }


def property_questions(subject: Optional[str], parts: List[str]):
    """Call 3: properties of the named things."""
    qs = {
        "bg_top": choice("Background color of the upper part of the picture", CRITERIA),
        "bg_bottom": choice("Background color of the lower part of the picture", CRITERIA),
        "horizon": score("Where the boundary between the lower and upper background sits", HORIZON),
    }
    if subject:
        qs.update(shape_questions("", subject))
        qs.update({
            "x": score(f"Horizontal position of the {subject}", POS),
            "y": score(f"Vertical position of the {subject}", POSY),
            "size": score(f"Size of the {subject} relative to the picture", SIZES),
            "edge": choice(f"Color of the {subject}'s outline or its second most visible color", CRITERIA),
            "glow": noul(f"The {subject} glows, shines or has rays or a halo around it"),
        })
        for i, clock in enumerate(CLOCK):
            qs[f"reach_{i}"] = score(f"How far the {subject}'s outline extends from its center toward {clock}", REACH)
        for i, part in enumerate(parts):
            qs.update(shape_questions(f"p{i}_", part))
            qs.update({
                f"p{i}_rel": choice(f"Where the {part} sits relative to the {subject}", REL),
                f"p{i}_side": score(f"Where along the {subject}'s width the {part} is attached", SIDE),
                f"p{i}_size": score(f"Size of the {part} compared with the {subject}", PART_SIZES),
            })
    return qs


def layout_questions(request: str):
    """All three calls' questions for a request with placeholder picks; for inspection and tests."""
    return {**pick_questions(request), **part_questions(request, "subject"), **property_questions("subject", ["part"])}


def _frac(a, levels):
    """Jev's score is the expected level index, zero-based: 0 is the first level, len-1 the last."""
    return min(max(float(a["score"]) / (len(levels) - 1), 0.0), 1.0)


def _interp(f, table):
    i = min(int(f * (len(table) - 1)), len(table) - 2)
    t = f * (len(table) - 1) - i
    return table[i] * (1 - t) + table[i + 1] * t


def make_plan(jev: Jev, request: str) -> Plan:
    state = {"image_request": request}
    a = jev.ask(state, pick_questions(request))
    subj = a["subject"]["choice"]
    subj = None if subj == "none" else subj
    parts: List[str] = []
    if subj:
        pq = part_questions(request, subj)
        if pq:
            pa = jev.ask(state, pq)
            a.update(pa)
            words = [k[len("part_"):] for k in pq if k.startswith("part_")]
            ok = [w for w in words if float(pa[f"part_{w}"]["noul"]) >= 0.5 and float(pa[f"noun_{w}"]["noul"]) >= 0.5]
            parts = sorted(ok, key=lambda w: -float(pa[f"part_{w}"]["noul"]))[:MAX_PARTS]
    a.update(jev.ask(state, property_questions(subj, parts)))
    plan = Plan(
        bg_top=NAME_TO_LETTER.get(a["bg_top"]["choice"], "S"),
        bg_bottom=NAME_TO_LETTER.get(a["bg_bottom"]["choice"], "D"),
        split=float(a["split"]["noul"]) >= 0.5,
        horizon=_frac(a["horizon"], HORIZON),
        subject=subj,
        shape=a["shape"]["choice"] if subj else "circle",
        x=_frac(a["x"], POS) if subj else 0.5, y=_frac(a["y"], POSY) if subj else 0.5,
        size=_interp(_frac(a["size"], SIZES), SIZE_FRAC) if subj else 0.0,
        color=NAME_TO_LETTER.get(a["color"]["choice"], "Y") if subj else "Y",
        edge=NAME_TO_LETTER.get(a["edge"]["choice"], "O") if subj else "O",
        glow=float(a["glow"]["noul"]) >= 0.5 if subj else False,
        answers=a,
        tilt=_interp(_frac(a["tilt"], TILTS), TILT_DEG) if subj else 0.0,
        aspect=_interp(_frac(a["aspect"], ELONG), ELONG_RATIO) if subj else 1.0,
        outline=[_interp(_frac(a[f"reach_{i}"], REACH), REACH_FRAC) for i in range(len(CLOCK))] if subj else None,
    )
    for i, name in enumerate(parts):
        plan.parts.append(Part(
            name=name, rel=a[f"p{i}_rel"]["choice"], side=_interp(_frac(a[f"p{i}_side"], SIDE), SIDE_OFF),
            shape=a[f"p{i}_shape"]["choice"] if a[f"p{i}_shape"]["choice"] != "outline" else "ellipse",
            tilt=_interp(_frac(a[f"p{i}_tilt"], TILTS), TILT_DEG),
            aspect=_interp(_frac(a[f"p{i}_aspect"], ELONG), ELONG_RATIO),
            size=plan.size * _interp(_frac(a[f"p{i}_size"], PART_SIZES), PART_FRAC),
            color=NAME_TO_LETTER.get(a[f"p{i}_color"]["choice"], "C"),
        ))
    return fit_composite(plan)


# ----------------------------------------------------------------------------- geometry

def _rot(dx, dy, deg):
    t = math.radians(deg)
    c, s_ = math.cos(t), math.sin(t)
    return dx * c + dy * s_, -dx * s_ + dy * c


def shape_inside(shape: str, dx: float, dy: float, aspect: float = 1.0, tilt: float = 0.0,
                 outline: Optional[List[float]] = None) -> bool:
    """dx, dy in units of the radius, image coordinates (y down). Length axis is vertical when upright;
    elongation squeezes x. Tilt rotates clockwise."""
    dx, dy = _rot(dx, dy, tilt)
    if shape == "outline" and outline:
        ang = math.degrees(math.atan2(dx, -dy)) % 360        # 0 = up, clockwise
        k = ang / 30.0
        i, t = int(k) % 12, k - int(k)
        r = outline[i] * (1 - t) + outline[(i + 1) % 12] * t
        return math.hypot(dx, dy) <= r
    if shape == "crescent":
        aspect = 1.0
    x = dx * aspect
    y = dy
    if shape in ("circle", "ellipse"):
        return math.hypot(x, y) <= 1
    if shape == "blob":
        ang = math.atan2(y, x)
        return math.hypot(x, y) <= 1 + 0.08 * math.sin(3 * ang) + 0.05 * math.cos(5 * ang + 1)
    if shape == "capsule":
        half = max(1 - 1 / aspect, 0.0)
        yy = min(max(y, -half), half)
        return math.hypot(x, (y - yy) * aspect) <= 1
    if shape == "rectangle":
        return abs(x) <= 1 and abs(y) <= 1
    if shape == "triangle":
        return -1 <= y <= 1 and abs(x) <= (y + 1) / 2
    if shape == "crescent":
        return math.hypot(x, y) <= 1 and math.hypot(x - 0.55, y) >= 0.85
    if shape == "star":
        ang = math.atan2(x, -y)
        r = 0.45 + 0.55 * ((0.5 + 0.5 * math.cos(5 * ang)) ** 1.6)
        return math.hypot(x, y) <= r
    if shape == "heart":
        hx, hy = 1.25 * x, -1.25 * y + 0.15
        return (hx * hx + hy * hy - 1) ** 3 - hx * hx * hy ** 3 <= 0
    if shape == "leaf":
        return math.hypot(x - 0.6, y) <= 1.166 and math.hypot(x + 0.6, y) <= 1.166
    if shape == "ring":
        return 0.55 <= math.hypot(x, y) <= 1
    if shape == "arch":
        h = math.hypot(x, y)
        return (y <= 0 and 0.5 <= h <= 1) or (0 < y <= 1 and 0.5 <= abs(x) <= 1)
    return math.hypot(x, y) <= 1


def subject_inside(plan: Plan, u: float, v: float) -> bool:
    if plan.subject is None or plan.size <= 0:
        return False
    return shape_inside(plan.shape, (u - plan.x) / plan.size, (v - plan.y) / plan.size, plan.aspect, plan.tilt, plan.outline)


DIRS = {"below": (0.0, 1.0), "above": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def extent(inside: Callable[[float, float], bool], radius: float, dx: float, dy: float) -> float:
    """How far a shape reaches from its centre along (dx, dy): march outward until the point is outside."""
    step = radius / 40
    d, last = 0.0, 0.0
    while d <= radius * 2.2:
        if inside(dx * d, dy * d):
            last = d
        elif d > radius * 0.3:
            break
        d += step
    return max(last, radius * 0.25)


def part_geometry(plan: Plan, part: Part) -> Tuple[float, float, float]:
    """Centre and radius of a part, placed against the real edge of the subject in its direction."""
    r = part.size
    x, y = plan.x, plan.y
    lateral = part.side * plan.size
    if part.rel in DIRS:
        ddx, ddy = DIRS[part.rel]
        reach_s = extent(lambda ox, oy: shape_inside(plan.shape, (ox + lateral * ddy) / plan.size, (oy + lateral * ddx) / plan.size,
                                                    plan.aspect, plan.tilt, plan.outline), plan.size, ddx, ddy)
        reach_p = extent(lambda ox, oy: shape_inside(part.shape, ox / r, oy / r, part.aspect, part.tilt), r, -ddx, -ddy)
        gap = reach_s + reach_p * 0.85
        x, y = x + ddx * gap + lateral * abs(ddy), y + ddy * gap + lateral * abs(ddx)
    elif part.rel == "around":
        r = plan.size * 1.35
    else:  # inside
        x, y = x + lateral * 0.5, y
    return x, y, r


def part_inside(plan: Plan, part: Part, u: float, v: float) -> bool:
    x, y, r = part_geometry(plan, part)
    return r > 0 and shape_inside(part.shape, (u - x) / r, (v - y) / r, part.aspect, part.tilt)


def _composite_box(plan: Plan):
    boxes = [(plan.x - plan.size, plan.y - plan.size, plan.x + plan.size, plan.y + plan.size)]
    for part in plan.parts:
        x, y, r = part_geometry(plan, part)
        boxes.append((x - r, y - r, x + r, y + r))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def fit_composite(plan: Plan, margin: float = 0.04) -> Plan:
    """Jev places the subject without knowing where the parts will land. Shrink if the composite is bigger
    than the canvas, then shift so all of it is inside."""
    if plan.subject is None:
        return plan
    limit = 1 - 2 * margin
    x0, y0, x1, y1 = _composite_box(plan)
    if max(x1 - x0, y1 - y0) > limit:
        f = 0.98 * limit / max(x1 - x0, y1 - y0)
        plan.size *= f
        for part in plan.parts:
            part.size *= f
        x0, y0, x1, y1 = _composite_box(plan)
    plan.x += (margin - x0) if x0 < margin else (1 - margin - x1) if x1 > 1 - margin else 0.0
    plan.y += (margin - y0) if y0 < margin else (1 - margin - y1) if y1 > 1 - margin else 0.0
    return plan


def owner(plan: Plan, u: float, v: float) -> Tuple[Optional[str], str]:
    """Which drawn thing covers this point: ("subject"|part name|None, letter). Parts drawn inside the subject
    win over it; other parts sit behind the subject."""
    for part in plan.parts:
        if part.rel == "inside" and part_inside(plan, part, u, v):
            return part.name, part.color
    if subject_inside(plan, u, v):
        return "subject", plan.color
    for part in plan.parts:
        if part.rel != "inside" and part_inside(plan, part, u, v):
            return part.name, part.color
    return None, ""


def region(plan: Plan, u: float, v: float, cell: float) -> Tuple[str, str]:
    """(region text, provisional letter) for a cell centred at (u, v); cell is one cell's size in [0,1] units.
    Edges are found by sampling the neighbouring cell centres, so every silhouette gets an edge band."""
    who, letter = owner(plan, u, v)
    name = plan.subject or "subject"
    nb = [owner(plan, u + du, v + dv)[0] for du, dv in ((cell, 0), (-cell, 0), (0, cell), (0, -cell))]
    if who is not None:
        label = name if who == "subject" else who
        if all(n == who for n in nb):
            return f"inside the {label}, well within it", letter
        edge_letter = plan.edge if who == "subject" else letter
        return f"inside the {label}, at its edge", edge_letter
    bg = "background"
    if plan.split:
        if abs(v - plan.horizon) <= cell:
            bg_letter = plan.bg_bottom if v >= plan.horizon else plan.bg_top
            bg = "on the horizon line between upper and lower background"
        elif v < plan.horizon:
            bg, bg_letter = "upper background (sky or wall)", plan.bg_top
        else:
            bg, bg_letter = "lower background (ground, floor or water)", plan.bg_bottom
    else:
        bg_letter = plan.bg_top if v < 0.5 else plan.bg_bottom
    near = [n for n in nb if n is not None]
    if not near:
        ring = [owner(plan, u + du, v + dv)[0] for du, dv in ((2 * cell, 0), (-2 * cell, 0), (0, 2 * cell), (0, -2 * cell),
                                                             (cell, cell), (-cell, cell), (cell, -cell), (-cell, -cell))]
        near = [n for n in ring if n is not None]
    if near:
        label = name if near[0] == "subject" else near[0]
        if plan.glow and near[0] == "subject":
            return f"just outside the {name}, in its glow", bg_letter
        return f"{bg}, right next to the {label}", bg_letter
    return bg, bg_letter


def where(r: int, c: int, n: int) -> str:
    v, h = r / (n - 1), c / (n - 1)
    vy = "top" if v < 0.3 else "upper middle" if v < 0.45 else "middle" if v <= 0.55 else "lower middle" if v <= 0.7 else "bottom"
    hx = "left" if h < 0.3 else "center-left" if h < 0.45 else "center" if h <= 0.55 else "center-right" if h <= 0.7 else "right"
    return f"{vy} {hx}"


# ----------------------------------------------------------------------------- stage 2: pixels

Grid = List[List[Optional[str]]]     # committed letters, None if undecided
Prov = List[List[str]]               # provisional letters (from the plan, then from the coarser level)


def render_state(request: str, plan: Plan, grid: Grid, prov: Prov, rows: Optional[Tuple[int, int]] = None) -> str:
    """The canvas as text. For big canvases only the band of rows around the cells being asked is sent."""
    n = len(grid)
    lo, hi = rows if rows else (0, n - 1)
    lo, hi = max(0, lo - 3), min(n - 1, hi + 3)
    legend = " ".join(f"{k}={NAMES[k]}" for k in LETTERS)
    body = "\n".join(f"{r:03d} " + "".join(grid[r][c] or prov[r][c].lower() for c in range(n)) for r in range(lo, hi + 1))
    subj = f"The main object is the {plan.subject}, drawn as a {plan.shape}." if plan.subject else "There is no single main object."
    for part in plan.parts:
        subj += f" The {part.name} is a {part.shape} {REL[part.rel]}."
    window = "" if (lo, hi) == (0, n - 1) else f" Only rows {lo:03d} to {hi:03d} are shown.\n"
    return (f"Image request: {request}\n{subj}\n"
            f"Canvas: {n}x{n} cells, row 000 at the top, column 000 at the left, seen as a tiny pixel-art picture.{window}\n"
            f"UPPERCASE letters are decided cells. lowercase letters are provisional guesses that may be changed.\n"
            f"Palette: {legend}\n{body}")


def cell_id(r, c):
    return f"r{r:03d}c{c:03d}"


def pixel_question(plan: Plan, grid: Grid, prov: Prov, r: int, c: int):
    n = len(grid)
    reg, _ = region(plan, (c + 0.5) / n, (r + 0.5) / n, 1 / n)

    def nb(dr, dc):
        rr, cc = r + dr, c + dc
        if not (0 <= rr < n and 0 <= cc < n):
            return "edge of canvas"
        return NAMES[grid[rr][cc]] if grid[rr][cc] else NAMES[prov[rr][cc]] + " (provisional)"
    text = (f"Color of the cell at row {r:03d} column {c:03d}: {where(r, c, n)} of the picture, {reg}. "
            f"Provisional color {NAMES[prov[r][c]]}. Neighbors: above {nb(-1, 0)}, below {nb(1, 0)}, "
            f"left {nb(0, -1)}, right {nb(0, 1)}. Pick the color that makes the whole canvas depict the request.")
    return choice(text, CRITERIA)


def ask_cells(jev: Jev, request: str, plan: Plan, grid: Grid, prov: Prov, cells, chunk: int, workers: int):
    chunks = [cells[i:i + chunk] for i in range(0, len(cells), chunk)]
    crop = len(grid) > 32

    def one(cs):
        qs = {cell_id(r, c): pixel_question(plan, grid, prov, r, c) for r, c in cs}
        rows = (min(r for r, _ in cs), max(r for r, _ in cs)) if crop else None
        state = render_state(request, plan, grid, prov, rows)
        try:
            a = jev.ask(state, qs)
        except JevError as e:
            if "422" in str(e) and len(cs) > 8:  # over the request limit: split
                mid = len(cs) // 2
                a = dict(one(cs[:mid])); a.update(one(cs[mid:]))
                return a
            raise
        return {(r, c): a[cell_id(r, c)] for r, c in cs}

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(one, chunks):
            out.update(part)
    return out


def soft_rgb(probs: Dict[str, float]) -> Tuple[int, int, int]:
    tot = sum(probs.values()) or 1.0
    out = [0.0, 0.0, 0.0]
    for name, p in probs.items():
        k = NAME_TO_LETTER.get(name)
        if k:
            for j in range(3):
                out[j] += RGB[k][j] * p / tot
    return tuple(int(v) for v in out)


def to_image(grid: Grid, prov: Prov, soft: Dict[Tuple[int, int], Dict[str, float]], out_px: int,
             render: str = "soft") -> Image.Image:
    """render="soft": every cell is the palette blended by Jev's probabilities, decided or not.
    render="hard": decided cells take their palette color; only undecided cells are blended."""
    n = len(grid)
    im = Image.new("RGB", (n, n))
    px = im.load()
    for r in range(n):
        for c in range(n):
            probs = soft.get((r, c))
            if probs and (render == "soft" or not grid[r][c]):
                px[c, r] = soft_rgb(probs)
            elif grid[r][c]:
                px[c, r] = RGB[grid[r][c]]
            else:
                px[c, r] = RGB[prov[r][c]]
    return im.resize((out_px, out_px), Image.NEAREST)


def plan_grid(plan: Plan, n: int) -> Prov:
    return [[region(plan, (c + 0.5) / n, (r + 0.5) / n, 1 / n)[1] for c in range(n)] for r in range(n)]


def upsample(grid: Grid) -> Prov:
    return [[grid[r // 2][c // 2] for c in range(2 * len(grid))] for r in range(2 * len(grid))]


def uniform_interior(prov: Prov, r: int, c: int) -> bool:
    """True when the cell and its 8 neighbours all carry the same provisional color."""
    n = len(prov)
    k = prov[r][c]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and prov[rr][cc] != k:
                return False
    return True


def spread_commit(cells, answers, k: int, rng: random.Random):
    """Pick k cells to commit: confidence-weighted, but preferring cells whose neighbors are not
    already picked, so decided cells end up spread out and every undecided cell gets context."""
    conf = {rc: max(float(answers[rc].get("confidence", 0.0)), 1e-3) for rc in cells}
    order = sorted(cells, key=lambda rc: conf[rc] * (0.5 + rng.random()), reverse=True)
    picked, taken = [], set()
    for rc in order:
        if len(picked) >= k:
            break
        r, c = rc
        if any((r + dr, c + dc) in taken for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))):
            continue
        picked.append(rc); taken.add(rc)
    for rc in order:  # fill up if the checkerboard constraint left us short
        if len(picked) >= k:
            break
        if rc not in taken:
            picked.append(rc); taken.add(rc)
    return picked


RoundCallback = Callable[[dict, Image.Image], None]


def generate(jev: Jev, request: str, size: int = 32, rounds: int = 2, chunk: Optional[int] = None, workers: int = 4,
             seed: int = 0, out_px: Optional[int] = None, render: str = "soft", inherit: bool = True,
             on_round: Optional[RoundCallback] = None,
             on_plan: Optional[Callable[[Plan], None]] = None) -> Tuple[Image.Image, dict]:
    rng = random.Random(seed)
    u0 = dict(jev.usage)
    chunk = chunk or (120 if size <= 64 else 100)
    out_px = out_px or max(512, size * 4)
    plan = make_plan(jev, request)
    if on_plan:
        on_plan(plan)
    levels = [size // 4, size // 2, size] if size >= 16 else [size]
    grid: Optional[Grid] = None
    prov: Prov = plan_grid(plan, levels[0])
    soft: Dict[Tuple[int, int], Dict[str, float]] = {}
    total_rounds = rounds * len(levels)
    done_rounds = 0
    asked = 0
    for li, n in enumerate(levels):
        if grid is not None:
            prov = upsample(grid)
            soft = {(r, c): soft[(r // 2, c // 2)] for r in range(n) for c in range(n) if (r // 2, c // 2) in soft}
        grid = [[None] * n for _ in range(n)]
        if inherit and li > 0:
            # Deep inside a uniform region the answer will not change: inherit the parent's color and
            # probabilities instead of asking. Cells near any color boundary are still asked.
            for r in range(n):
                for c in range(n):
                    if uniform_interior(prov, r, c):
                        grid[r][c] = prov[r][c]
        for step in range(rounds):
            cells = [(r, c) for r in range(n) for c in range(n) if grid[r][c] is None]
            if not cells:
                break
            answers = ask_cells(jev, request, plan, grid, prov, cells, chunk, workers)
            asked += len(cells)
            for rc, a in answers.items():
                soft[rc] = a.get("probabilities", {})
            remaining_after = 0 if step == rounds - 1 else int(len(cells) * (1 - (step + 1) / rounds))
            k = len(cells) - remaining_after
            for r, c in spread_commit(cells, answers, k, rng):
                grid[r][c] = NAME_TO_LETTER.get(answers[(r, c)]["choice"], prov[r][c])
            done_rounds += 1
            if on_round:
                committed = sum(1 for row in grid for x in row if x)
                on_round({"level": li + 1, "levels": len(levels), "grid": n, "round": done_rounds, "rounds": total_rounds,
                          "committed": committed, "cells": n * n, "asked": asked},
                         to_image(grid, prov, soft, out_px, render))
    im = to_image(grid, prov, soft, out_px, render)
    usage = {k: jev.usage[k] - u0[k] for k in u0}
    usage["cells_asked"] = asked
    return im, {"plan": plan.to_dict(), "grid": ["".join(x or "." for x in row) for row in grid], "usage": usage}
