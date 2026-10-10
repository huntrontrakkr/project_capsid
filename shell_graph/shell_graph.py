"""Shell state graph for project_capsid.

A hexagonal cell's shell is a 6-node binary ring (bit i = the node at the corner
between edges i and i+1, edges counted clockwise from the top), so a shell has 64 states.

Graph structure
    unit edges      add or remove one unit: the 6-cube, 192 edges, 6 per state
    rotation edges  rotate the ring one position: joins states into 14 necklaces
    complement      swap every open and filled node: maps fill level w to 6 - w

Layouts
    Every layout is authored for a 380 px wide canvas with the vertical axis at
    x = 190, all-open shells at the top and all-filled shells at the bottom.

        orbit_rows         necklaces as polygons in rows by fill level
        twelve_gon         standard 12-gon (Petrie) projection of the 6-cube
        layered            rows by fill level, ordered by 12-gon x
        necklace_rows      rows grouped into necklace blocks in rotation order
        necklace_columns   each necklace in its own vertical column
        stratified         12-gon with every necklace flattened onto one line
        stratified_arcs    amber split around the alternating pair, strata bent into arcs
        arc_rings          rotation-ordered strata on arcs, rings wrapping off the edges
        compact_rings      thin, tall rows with the fewest crossings found
        codex              closed necklace rings stacked between the poles

Usage
    python shell_graph.py                   rebuild layouts.json from the stored results
    python shell_graph.py --search compact  rerun one search (rows, arcs, compact, codex)

The searches are seeded. The stored constants below are the results that were
shown on 2026-10-10.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

CX = 190.0
WIDTH = 380.0


# ---------------------------------------------------------------- core graph
def rot(s, k=1):
    k %= 6
    return ((s << k) | (s >> (6 - k))) & 63 if k else s


def comp(s):
    return 63 ^ s


def weight(s):
    return bin(s).count("1")


def canon(s):
    return min(rot(s, k) for k in range(6))


def pattern(s):
    return "".join("●" if (s >> i) & 1 else "○" for i in range(6))


def orbit(c):
    """Rotation order of a necklace, starting at c and stepping one position clockwise."""
    m, t = [c], rot(c, 1)
    while t != c:
        m.append(t)
        t = rot(t, 1)
    return m


CLASS = [canon(s) for s in range(64)]
NECKLACES = sorted(set(CLASS), key=lambda c: (weight(c), c))
MEMBERS = {c: [s for s in range(64) if CLASS[s] == c] for c in NECKLACES}
UNIT_EDGES = [(s, s ^ (1 << i), i) for s in range(64) for i in range(6) if s < s ^ (1 << i)]
ROTATION_EDGES = sorted({(min(s, rot(s, 1)), max(s, rot(s, 1))) for s in range(64) if rot(s, 1) != s})

# colors used in the necklace-colored views: complementary necklaces share a hue
NECKLACE_COLOR = {0: "#B4B2A9", 63: "#5F5E5A", 1: "#F0997B", 31: "#993C1D", 3: "#85B7EB", 15: "#185FA5",
                  5: "#5DCAA5", 23: "#0F6E56", 9: "#AFA9EC", 27: "#534AB7", 7: "#BA7517", 21: "#639922",
                  11: "#ED93B1", 13: "#993556"}


def ring_orders(c):
    """Every way to lay a necklace out in rotation order: each start, both directions."""
    m = orbit(c)
    n = len(m)
    out = []
    for st in range(n):
        for d in (1, -1):
            t = [m[(st + d * k) % n] for k in range(n)]
            if t not in out:
                out.append(t)
    return out


def amber_triples():
    """Three rotation-consecutive three-in-a-row states; their complements are the other three."""
    a = orbit(7)
    return [[a[(st + d * k) % 6] for k in range(3)] for st in range(6) for d in (1, -1)]


# ------------------------------------------------------------- measurement
_E = np.array([(s, t) for s, t, _ in UNIT_EDGES])
_SHARE = np.zeros((len(_E), len(_E)), bool)
for _i in (0, 1):
    for _j in (0, 1):
        _SHARE |= _E[:, _i][:, None] == _E[:, _j][None, :]


def _orient(p, q, r):
    return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))


def crossings(pos, bend=0.0, samples=6):
    """Pairs of unit edges that cross. With bend > 0 each edge is the quadratic curve
    whose control point is the chord midpoint pushed away from the axis by bend * offset."""
    pos = np.asarray(pos, float)
    A, B = pos[_E[:, 0]], pos[_E[:, 1]]
    if bend == 0:
        a, b, c, d = A[:, None], B[:, None], A[None], B[None]
        X = ((_orient(a, b, c) * _orient(a, b, d)) < 0) & ((_orient(c, d, a) * _orient(c, d, b)) < 0) & ~_SHARE
        return int(X.sum()) // 2
    M = (A + B) / 2
    C = M.copy()
    C[:, 0] += bend * (M[:, 0] - CX)
    u = np.linspace(0, 1, samples + 1)[:, None, None]
    pts = (1 - u) ** 2 * A + 2 * (1 - u) * u * C + u * u * B          # [samples+1, edges, 2]
    S0 = pts[:-1].transpose(1, 0, 2).reshape(-1, 2)
    S1 = pts[1:].transpose(1, 0, 2).reshape(-1, 2)
    own = np.repeat(np.arange(len(_E)), samples)
    a, b, c, d = S0[:, None], S1[:, None], S0[None], S1[None]
    X = ((_orient(a, b, c) * _orient(a, b, d)) < 0) & ((_orient(c, d, a) * _orient(c, d, b)) < 0)
    X &= ~_SHARE[own][:, own]
    i, j = np.nonzero(np.triu(X))
    keep = own[i] != own[j]
    codes = np.unique(np.minimum(own[i], own[j])[keep] * 1000 + np.maximum(own[i], own[j])[keep])
    return len(codes)


def edge_length(pos):
    pos = np.asarray(pos, float)
    return float(np.linalg.norm(pos[_E[:, 0]] - pos[_E[:, 1]], axis=1).sum())


# ------------------------------------------------------------------ layouts
def twelve_gon_raw(s):
    """6-cube projected onto its Coxeter plane, turned so all-open is up and all-filled is down."""
    x = y = 0.0
    for i in range(6):
        c = ((s >> i) & 1) - 0.5
        a = math.radians(i * 30 + 15)
        x += c * math.cos(a)
        y += c * math.sin(a)
    return x, y


def layout_twelve_gon(cy=250.0, scale=88.0, fan=8.5):
    """The true projection stacks some states on one point; those are fanned apart slightly."""
    raw = [twelve_gon_raw(s) for s in range(64)]
    pos = [[CX + scale * x, cy + scale * y] for x, y in raw]
    groups = {}
    for s, (x, y) in enumerate(raw):
        groups.setdefault((round(x * 1000), round(y * 1000)), []).append(s)
    for g in groups.values():
        if len(g) < 2:
            continue
        r = 11 if len(g) > 2 else fan
        for j, s in enumerate(g):
            a = math.pi / 4 + 2 * math.pi * j / len(g)
            pos[s] = [pos[s][0] + r * math.cos(a), pos[s][1] + r * math.sin(a)]
    return pos


def layout_layered(row_y=lambda w: 30 + w * 73.3, spacing=17.9):
    tg = layout_twelve_gon()
    pos = [None] * 64
    for w in range(7):
        row = sorted([s for s in range(64) if weight(s) == w], key=lambda s: (tg[s][0], tg[s][1]))
        for j, s in enumerate(row):
            pos[s] = [CX + (j - (len(row) - 1) / 2) * spacing, row_y(w)]
    return pos


def layout_orbit_rows():
    """First sketch: necklaces as small polygons (hexagon, triangle, pair) in rows by fill level."""
    ys = [34, 128, 232, 336, 440, 544, 638]
    wd = {6: 70, 3: 52, 2: 20, 1: 20}
    pos = [None] * 64
    for w in range(7):
        orbs = sorted([orbit(c) for c in NECKLACES if weight(c) == w], key=lambda o: (-len(o), o[0]))
        tot = sum(wd[len(o)] for o in orbs) + 24 * (len(orbs) - 1)
        x = (WIDTH - tot) / 2
        for o in orbs:
            cx, cy = x + wd[len(o)] / 2, ys[w]
            for k, s in enumerate(o):
                px, py = cx, cy
                if len(o) == 6:
                    a = math.radians(-90 + k * 60)
                    px, py = cx + 28 * math.cos(a), cy + 28 * math.sin(a)
                elif len(o) == 3:
                    a = math.radians(-90 + k * 120)
                    px, py = cx + 18 * math.cos(a), cy + 18 * math.sin(a)
                elif len(o) == 2:
                    py = cy + (14 if k else -14)
                pos[s] = [px, py]
            x += wd[len(o)] + 24
    return pos


# necklace_rows: rows grouped into necklace blocks, each block in rotation order, the
# 2-filled and 4-filled rows aligned so every state sits above its complement (1,516 crossings)
NECKLACE_ROWS = [[0], [32, 1, 2, 4, 8, 16], [34, 5, 10, 20, 40, 17, 18, 36, 9, 48, 33, 3, 6, 12, 24],
                 [42, 21, 50, 37, 11, 22, 44, 25, 13, 26, 52, 41, 19, 38, 7, 14, 28, 56, 49, 35],
                 [29, 58, 53, 43, 23, 46, 45, 27, 54, 15, 30, 60, 57, 51, 39], [31, 62, 61, 59, 55, 47], [63]]


def layout_necklace_rows(rows=NECKLACE_ROWS, row_y=lambda w: 30 + w * 73.3):
    pos = [None] * 64
    for w, row in enumerate(rows):
        xs = [0.0]
        for j in range(1, len(row)):
            xs.append(xs[-1] + 15.5 + (12 if CLASS[row[j]] != CLASS[row[j - 1]] else 0))
        off = CX - xs[-1] / 2
        for s, x in zip(row, xs):
            pos[s] = [off + x, row_y(w)]
    return pos


def search_necklace_rows(restarts=30, seed=0):
    """Pick block order, rotation start and direction to minimize crossings (rows layout)."""
    W2, W3 = [3, 5, 9], [7, 11, 13, 21]
    orients = {c: ring_orders(c) for c in [1] + W2 + W3}

    def build(st):
        w1 = orients[1][st["o"][1]]
        w2 = [s for c in st["b2"] for s in orients[c][st["o"][c]]]
        w3 = [s for c in st["b3"] for s in orients[c][st["o"][c]]]
        return [[0], w1, w2, w3, [comp(s) for s in w2], [comp(s) for s in w1], [63]]

    def score(rows):
        pos = np.zeros(64)
        for r in rows:
            for j, s in enumerate(r):
                pos[s] = j
        tot = 0
        for w in range(6):
            E = np.array([(s, s | (1 << i)) for s in rows[w] for i in range(6) if not (s >> i) & 1])
            a, b = pos[E[:, 0]], pos[E[:, 1]]
            tot += int(((a[:, None] - a[None, :]) * (b[:, None] - b[None, :]) < 0).sum()) // 2
        return tot

    best = None
    for r in range(restarts):
        rnd = random.Random(seed + r)
        st = {"b2": rnd.sample(W2, 3), "b3": rnd.sample(W3, 4), "o": {c: rnd.randrange(len(orients[c])) for c in orients}}
        cur = score(build(st))
        for it in range(6000):
            T = 20 * (1 - it / 6000) + 0.01
            new = {"b2": st["b2"][:], "b3": st["b3"][:], "o": dict(st["o"])}
            m = rnd.random()
            if m < 0.15:
                i, j = rnd.sample(range(3), 2)
                new["b2"][i], new["b2"][j] = new["b2"][j], new["b2"][i]
            elif m < 0.3:
                i, j = rnd.sample(range(4), 2)
                new["b3"][i], new["b3"][j] = new["b3"][j], new["b3"][i]
            else:
                c = rnd.choice(list(orients))
                new["o"][c] = rnd.randrange(len(orients[c]))
            v = score(build(new))
            if v <= cur or rnd.random() < math.exp((cur - v) / T):
                st, cur = new, v
        if best is None or cur < best[0]:
            best = (cur, build(st))
    return best[1]


# necklace_columns: each necklace stacked vertically in its own column, alternating pair at
# the center, complements mirrored top to bottom. Stored output of the column search.
NECKLACE_COLUMNS = [[190, 18], [190, 85], [190, 71], [240, 193], [190, 57], [140, 179], [240, 179], [340, 283], [190, 127],
                    [40, 214], [140, 193], [290, 297], [240, 165], [90, 283], [340, 269], [240, 387], [190, 113], [140, 235],
                    [40, 200], [90, 339], [140, 207], [190, 311], [290, 283], [140, 387], [240, 235], [290, 339], [90, 297],
                    [40, 422], [340, 339], [140, 443], [240, 401], [190, 509], [190, 99], [240, 207], [140, 165], [340, 297],
                    [40, 186], [290, 311], [90, 269], [240, 373], [140, 221], [90, 325], [190, 297], [140, 401], [290, 269],
                    [40, 408], [140, 373], [190, 495], [240, 221], [340, 311], [290, 325], [240, 443], [90, 311], [140, 415],
                    [40, 394], [190, 481], [340, 325], [240, 429], [140, 429], [190, 551], [240, 415], [190, 537], [190, 523],
                    [190, 590]]


def _symmetric_slots(xs, gap):
    """Pair the k-th from the left with the k-th from the right, mirror about the axis, then push apart."""
    xs = sorted(xs)
    n = len(xs)
    xs = [CX + (xs[k] - xs[n - 1 - k]) / 2 for k in range(n)]
    for _ in range(200):
        for k in range(n - 1):
            d = xs[k + 1] - xs[k]
            if d < gap:
                xs[k] -= (gap - d) / 2
                xs[k + 1] += (gap - d) / 2
        xs = [CX + (xs[k] - xs[n - 1 - k]) / 2 for k in range(n)]
    return xs


def _flatten(groups, gap, row_y):
    """Stratify: each group onto its own line, keeping 12-gon x order, symmetrized about the axis."""
    tg = [(CX + 88 * x, 250 + 88 * y) for x, y in map(twelve_gon_raw, range(64))]
    pos = [None] * 64
    for k, g in enumerate(groups):
        if g is None:
            continue
        m = sorted(g, key=lambda s: (round(tg[s][0], 6), tg[s][1]))
        for s, x in zip(m, _symmetric_slots([tg[s][0] for s in m], gap)):
            pos[s] = [x, row_y(k)]
    return pos


def layout_stratified():
    """Every necklace on one line, lines pushed apart in 12-gon height order (1,001 crossings)."""
    order = [0, 1, 3, 9, 5, 11, 7, 21, None, 13, 23, 27, 15, 31, 63]
    groups = [MEMBERS[c] if c is not None else None for c in order]
    return _flatten(groups, 19, lambda k: 40 + k * 28)


def _split_amber():
    tg = [twelve_gon_raw(s) for s in range(64)]
    amber = sorted(MEMBERS[7], key=lambda s: (tg[s][1], tg[s][0]))
    upper = amber[:3]
    return upper, [comp(s) for s in upper]


def _arc_groups():
    up, down = _split_amber()
    return [[0], MEMBERS[1], MEMBERS[3], MEMBERS[9], MEMBERS[5], MEMBERS[11], up, MEMBERS[21], down,
            MEMBERS[13], MEMBERS[23], MEMBERS[27], MEMBERS[15], MEMBERS[31], [63]]


# bends (per stratum, y += k * (x - 190)^2) that minimize total edge length with strata >= 14 px apart
ARC_BENDS = [0.0027829901216353163, 0.003125655190789025, 0.002603206339015475, 0.0020807628763207337,
             0.001567338880017284, 0.0010448901124078995, 0.0005224478983701067, 0.0]
ARC_BENDS = ARC_BENDS + [-b for b in reversed(ARC_BENDS[:-1])]


def layout_stratified_arcs(bends=ARC_BENDS):
    groups = _arc_groups()
    pos = _flatten(groups, 17, lambda k: 40 + k * 30)
    for k, g in enumerate(groups):
        for s in g:
            pos[s][1] += bends[k] * (pos[s][0] - CX) ** 2
    return pos


def search_arc_bends():
    from scipy.optimize import minimize
    groups = _arc_groups()
    flat = np.array(_flatten(groups, 17, lambda k: 40 + k * 30))
    stratum = {s: k for k, g in enumerate(groups) for s in g}
    xg = np.linspace(15, 365, 71)

    def bends(kap):
        kk = np.zeros(15)
        for k in range(7):
            kk[k], kk[14 - k] = kap[k], -kap[k]
        return kk

    def objective(kap):
        kk = bends(kap)
        pos = flat.copy()
        for s in range(64):
            pos[s, 1] += kk[stratum[s]] * (pos[s, 0] - CX) ** 2
        curves = np.array([[40 + k * 30 + kk[k] * (x - CX) ** 2 for x in xg] for k in range(15)])
        return edge_length(pos) + np.clip(14 - (curves[1:] - curves[:-1]), 0, None).sum() * 500

    best = min((minimize(objective, s0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-3})
                for s0 in [np.zeros(7), np.full(7, 0.002), np.full(7, -0.002)]), key=lambda r: r.fun)
    return bends(best.x).tolist()


# arc_rings: rotation-ordered strata on arcs; the rotation ring of each row wraps off both edges
ARC_RING_STRATA = [[0], [8, 16, 32, 1, 2, 4], [12, 24, 48, 33, 3, 6], [9, 18, 36], [17, 40, 20, 10, 5, 34],
                   [25, 44, 22, 11, 37, 50], [56, 49, 35], [21, 42], [28, 14, 7], [13, 26, 52, 41, 19, 38],
                   [29, 58, 53, 43, 23, 46], [27, 45, 54], [57, 60, 30, 15, 39, 51], [59, 61, 62, 31, 47, 55], [63]]
ARC_RING_BENDS = [14.51, 15.797, 14.416, 11.556, 8.665, 5.781, 2.89, 0.0, -2.89, -5.781, -8.665, -11.556, -14.416, -15.797, -14.51]


def layout_arc_rings(strata=ARC_RING_STRATA, bends=ARC_RING_BENDS, y0=34.0, sp=36.0):
    tg = [CX + 88 * twelve_gon_raw(s)[0] for s in range(64)]
    H = 2 * (y0 + 7 * sp)
    pos = [None] * 64
    for k, order in enumerate(strata[:8]):
        if k == 0:
            xs = [CX]
        elif k == 6:
            up, _ = _split_amber_top()
            xs = _symmetric_slots([tg[s] for s in up], 22)
        else:
            xs = _symmetric_slots([tg[s] for s in MEMBERS[CLASS[order[0]]]], 22)
        for s, x in zip(order, xs):
            pos[s] = [x, y0 + k * sp]
    for order in strata[:7]:
        for s in order:
            pos[comp(s)] = [WIDTH - pos[s][0], H - pos[s][1]]
    stratum = {s: k for k, o in enumerate(strata) for s in o}
    for s in range(64):
        pos[s][1] += bends[stratum[s]] * 1e-4 * (pos[s][0] - CX) ** 2
    return pos


def _split_amber_top():
    tg = [twelve_gon_raw(s) for s in range(64)]
    up = sorted(MEMBERS[7], key=lambda s: tg[s][1])[:3]
    return up, [comp(s) for s in up]


# compact_rings: evenly spaced rows one glyph apart, rotation order, the open/filled swap
# turns the picture 180 degrees (988 crossings)
COMPACT_STRATA = [[0], [4, 8, 16, 32, 1, 2], [36, 9, 18], [40, 20, 10, 5, 34, 17], [12, 24, 48, 33, 3, 6],
                  [44, 25, 50, 37, 11, 22], [56, 49, 35], [42, 21], [28, 14, 7], [41, 52, 26, 13, 38, 19],
                  [57, 60, 30, 15, 39, 51], [46, 29, 58, 53, 43, 23], [45, 54, 27], [61, 62, 31, 47, 55, 59], [63]]


def layout_compact_rings(strata=COMPACT_STRATA, d=24.0, y0=30.0, sp=40.0):
    pos = [None] * 64
    for k, order in enumerate(strata):
        n = len(order)
        for j, s in enumerate(order):
            pos[s] = [CX + (j - (n - 1) / 2) * d, y0 + k * sp]
    return pos


def search_compact_rings(restarts=10, iters=2500, seed=0, d=24.0, y0=30.0, sp=40.0):
    RO = {c: ring_orders(c) for c in [1, 3, 9, 5, 11, 21]}
    AMB = amber_triples()
    H = 2 * (y0 + 7 * sp)

    def build(st):
        top = [[0], RO[1][st["o"][1]]] + [RO[c][st["o"][c]] for c in st["t2"]] + [RO[11][st["o"][11]], AMB[st["a"]]]
        strata = top + [RO[21][st["o"][21]]] + [[comp(s) for s in reversed(o)] for o in reversed(top)]
        return strata

    def score(st):
        return crossings(layout_compact_rings(build(st), d, y0, sp))

    rnd = random.Random(seed)
    best = None
    for _ in range(restarts):
        st = {"t2": rnd.sample([3, 9, 5], 3), "o": {c: rnd.randrange(len(RO[c])) for c in RO}, "a": rnd.randrange(12)}
        cur = score(st)
        for _ in range(iters):
            new = {"t2": st["t2"][:], "o": dict(st["o"]), "a": st["a"]}
            x = rnd.random()
            if x < 0.1:
                i, j = rnd.sample(range(3), 2)
                new["t2"][i], new["t2"][j] = new["t2"][j], new["t2"][i]
            elif x < 0.22:
                new["a"] = rnd.randrange(12)
            else:
                c = rnd.choice(list(RO))
                new["o"][c] = rnd.randrange(len(RO[c]))
            v = score(new)
            if v <= cur:
                st, cur = new, v
        if best is None or cur < best[0]:
            best = (cur, build(st))
    return best[1]


# codex: every necklace a closed ring (ellipse) stacked between the poles; the three-in-a-row
# ring encircles the alternating pair. Unit edges are drawn bowed outward with CODEX_BEND.
# Each ring: center y, radii, members in rotation order, and each member's angle in degrees
# (0 = right, 90 = front/down, increasing clockwise on screen).
CODEX_BEND = 0.3
CODEX_POLES = {0: [190.0, 40.0], 63: [190.0, 658.0]}
CODEX_RINGS = [
    {"cy": 75.0, "rx": 52, "ry": 13, "o": [4, 8, 16, 32, 1, 2], "t": [0, 60, 120, 180, 240, 300]},
    {"cy": 623.0, "rx": 52, "ry": 13, "o": [59, 55, 47, 31, 62, 61], "t": [180, 240, 300, 360, 420, 480]},
    {"cy": 124.0, "rx": 74, "ry": 14, "o": [9, 18, 36], "t": [90, 210, 330]},
    {"cy": 574.0, "rx": 74, "ry": 14, "o": [54, 45, 27], "t": [270, 390, 510]},
    {"cy": 175.0, "rx": 92, "ry": 15, "o": [10, 20, 40, 17, 34, 5], "t": [0, 60, 120, 180, 240, 300]},
    {"cy": 523.0, "rx": 92, "ry": 15, "o": [53, 43, 23, 46, 29, 58], "t": [180, 240, 300, 360, 420, 480]},
    {"cy": 228.0, "rx": 106, "ry": 16, "o": [6, 3, 33, 48, 24, 12], "t": [0, 60, 120, 180, 240, 300]},
    {"cy": 470.0, "rx": 106, "ry": 16, "o": [57, 60, 30, 15, 39, 51], "t": [180, 240, 300, 360, 420, 480]},
    {"cy": 283.0, "rx": 118, "ry": 17, "o": [22, 11, 37, 50, 25, 44], "t": [0, 60, 120, 180, 240, 300]},
    {"cy": 415.0, "rx": 118, "ry": 17, "o": [41, 52, 26, 13, 38, 19], "t": [180, 240, 300, 360, 420, 480]},
    {"cy": 349.0, "rx": 128, "ry": 27, "o": [7, 35, 49, 56, 28, 14], "t": [30, 90, 150, 210, 270, 330]},
    {"cy": 349.0, "rx": 16, "ry": 14, "o": [21, 42], "t": [180, 360]},
]


def layout_codex(rings=CODEX_RINGS, poles=CODEX_POLES):
    pos = [None] * 64
    for s, p in poles.items():
        pos[s] = list(p)
    for r in rings:
        for s, t in zip(r["o"], r["t"]):
            a = math.radians(t)
            pos[s] = [CX + r["rx"] * math.cos(a), r["cy"] + r["ry"] * math.sin(a)]
    return pos


def search_codex(restarts=10, iters=2000, seed=3,
                 rxs=(0, 52, 74, 92, 106, 118, 128), rys=(0, 13, 14, 15, 16, 17, 27), gap=22, bend=CODEX_BEND):
    """Ring order, starting angle and direction per necklace, chosen for fewest crossings."""
    RO = {c: ring_orders(c) for c in [1, 3, 9, 5, 11, 21]}
    amb = orbit(7)
    ys = [0.0]
    for L in range(1, 7):
        ys.append(ys[-1] + rys[L - 1] + rys[L] + gap)
    ys = [y + 40 for y in ys]
    c = ys[6]

    def rings_of(st):
        out = []
        tops = [(1, RO[1][st["o"][1]])] + [(k, RO[k][st["o"][k]]) for k in st["t2"]] + [(11, RO[11][st["o"][11]])]
        for L, (k, order) in enumerate(tops, start=1):
            n = len(order)
            th0 = (90 if st["b"][k] else 270) if n == 3 else 0
            ts = [th0 + j * 360 / n for j in range(n)]
            out.append({"cy": ys[L], "rx": rxs[L], "ry": rys[L], "o": order, "t": ts})
            out.append({"cy": 2 * c - ys[L], "rx": rxs[L], "ry": rys[L], "o": [comp(s) for s in order], "t": [t + 180 for t in ts]})
        a = [amb[(st["a0"] + st["ad"] * k) % 6] for k in range(6)]
        th0 = 30 if st["b"][7] else 0
        out.append({"cy": c, "rx": rxs[6], "ry": rys[6], "o": a, "t": [th0 + 60 * j for j in range(6)]})
        g = RO[21][st["o"][21]]
        out.append({"cy": c, "rx": 16, "ry": 14, "o": g, "t": [180, 360]})
        return out, {0: [CX, ys[0]], 63: [CX, 2 * c - ys[0]]}

    def score(st):
        rings, poles = rings_of(st)
        pos = np.array(layout_codex(rings, poles))
        dmin = (np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(64) * 99).min()
        return crossings(pos) + (0 if dmin >= 20.5 else 10000)

    rnd = random.Random(seed)
    best = None
    for _ in range(restarts):
        st = {"t2": rnd.sample([3, 9, 5], 3), "o": {k: rnd.randrange(len(RO[k])) for k in RO},
              "b": {k: rnd.randrange(2) for k in [1, 3, 9, 5, 11, 7]}, "a0": rnd.randrange(6), "ad": rnd.choice([1, -1])}
        cur = score(st)
        for _ in range(iters):
            new = {"t2": st["t2"][:], "o": dict(st["o"]), "b": dict(st["b"]), "a0": st["a0"], "ad": st["ad"]}
            x = rnd.random()
            if x < 0.08:
                i, j = rnd.sample(range(3), 2)
                new["t2"][i], new["t2"][j] = new["t2"][j], new["t2"][i]
            elif x < 0.25:
                k = rnd.choice([9, 7])
                new["b"][k] ^= 1
            elif x < 0.35:
                new["a0"], new["ad"] = rnd.randrange(6), rnd.choice([1, -1])
            else:
                k = rnd.choice(list(RO))
                new["o"][k] = rnd.randrange(len(RO[k]))
            v = score(new)
            if v <= cur:
                st, cur = new, v
        if best is None or cur < best[0]:
            best = (cur, st)
    rings, poles = rings_of(best[1])
    return rings, poles


# -------------------------------------------------------------- pathfinding
def cheapest_path(start, goal, rotation_cost=0.0, unit_cost=1.0):
    """Dijkstra over the shell graph. Rotation steps are one position at a time.
    Ties on cost go to the path with fewer moves. Returns (cost, [(state, move), ...])."""
    import heapq
    best = {start: (0.0, 0)}
    prev = {}
    heap = [(0.0, 0, start)]
    while heap:
        c, n, u = heapq.heappop(heap)
        if (c, n) != best.get(u):
            continue
        if u == goal:
            break
        steps = [(u ^ (1 << i), unit_cost, ("remove" if (u >> i) & 1 else "add", i)) for i in range(6)]
        steps += [(rot(u, k), rotation_cost, ("rotate", k)) for k in (1, -1) if rot(u, k) != u]
        for v, w, mv in steps:
            cand = (c + w, n + 1)
            if v not in best or cand < best[v]:
                best[v] = cand
                prev[v] = (u, mv)
                heapq.heappush(heap, (cand[0], cand[1], v))
    path = []
    v = goal
    while v != start:
        u, mv = prev[v]
        path.append((v, mv))
        v = u
    return best[goal][0], list(reversed(path))


# ------------------------------------------------------------------- output
def build_all():
    layouts = {
        "orbit_rows": {"pos": layout_orbit_rows(), "height": 670, "rings": "polygon"},
        "twelve_gon": {"pos": layout_twelve_gon(), "height": 500, "rings": None},
        "layered": {"pos": layout_layered(), "height": 500, "rings": None},
        "necklace_rows": {"pos": layout_necklace_rows(), "height": 500, "rings": None},
        "necklace_columns": {"pos": NECKLACE_COLUMNS, "height": 610, "rings": None},
        "stratified": {"pos": layout_stratified(), "height": 456, "rings": None},
        "stratified_arcs": {"pos": layout_stratified_arcs(), "height": 500, "rings": None},
        "arc_rings": {"pos": layout_arc_rings(), "height": 572, "rings": "wrap", "strata": ARC_RING_STRATA},
        "compact_rings": {"pos": layout_compact_rings(), "height": 620, "rings": "wrap", "strata": COMPACT_STRATA},
        "codex": {"pos": layout_codex(), "height": 698, "rings": "ellipse", "ellipses": CODEX_RINGS, "bend": CODEX_BEND},
    }
    for name, L in layouts.items():
        L["pos"] = [[round(float(x), 2), round(float(y), 2)] for x, y in L["pos"]]
        L["crossings"] = crossings(L["pos"], L.get("bend", 0.0))
        L["edge_length"] = round(edge_length(L["pos"]))
    graph = {
        "states": [{"id": s, "pattern": pattern(s), "filled": weight(s), "necklace": CLASS[s]} for s in range(64)],
        "necklaces": [{"id": c, "pattern": pattern(c), "size": len(MEMBERS[c]), "rotation_order": orbit(c),
                       "complement": canon(comp(c)), "color": NECKLACE_COLOR[c]} for c in NECKLACES],
        "unit_edges": [[s, t, i] for s, t, i in UNIT_EDGES],
        "rotation_edges": [list(e) for e in ROTATION_EDGES],
    }
    return {"graph": graph, "layouts": layouts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", choices=["rows", "arcs", "compact", "codex"], help="rerun one layout search and print the result")
    ap.add_argument("--out", default=str(Path(__file__).with_name("layouts.json")))
    args = ap.parse_args()
    if args.search == "rows":
        rows = search_necklace_rows()
        print(json.dumps(rows), crossings(layout_necklace_rows(rows)))
    elif args.search == "arcs":
        b = search_arc_bends()
        print(json.dumps(b), crossings(layout_stratified_arcs(b)))
    elif args.search == "compact":
        strata = search_compact_rings()
        print(json.dumps(strata), crossings(layout_compact_rings(strata)))
    elif args.search == "codex":
        rings, poles = search_codex()
        pos = layout_codex(rings, poles)
        print(json.dumps(rings), crossings(pos, CODEX_BEND))
    else:
        data = build_all()
        Path(args.out).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        for name, L in data["layouts"].items():
            print(f"{name:18s} crossings {L['crossings']:5d}  edge length {L['edge_length']}")


if __name__ == "__main__":
    main()
