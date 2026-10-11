"""Boards: which cells exist, who borders whom, beacon positions, starting shells, and starts.

`build` turns a set of cells into a Board and checks it against the rules. `hex_board` makes a
plain hexagon. `generate` makes a 2-player board with gaps and passages whose start areas are
mirror images of each other, with randomness that grows with distance from the starts
(development_plan.md, Board generator).
"""
import math
import random
from collections import deque
from dataclasses import dataclass, field

from . import hexgrid as hg
from .shell import rot

SYMMETRIC_SHELLS = [s for s in range(64) if rot(s, 3) == s]   # shells unchanged by a half turn


@dataclass(frozen=True)
class Board:
    coords: tuple          # (q, r) per cell
    nbr: tuple             # [cell][edge] -> cell across that edge, -1 if none
    vertices: tuple        # [vertex] -> cells around that beacon position (2 or 3)
    corner_vertex: tuple   # [cell][corner] -> vertex at that corner, -1 if not a beacon position
    shells: tuple          # starting shell per cell
    starts: tuple          # start cells per player: a tuple of cells for each
    name: str = ""
    info: dict = field(default_factory=dict, compare=False)

    @property
    def n_cells(self):
        return len(self.coords)

    @property
    def n_vertices(self):
        return len(self.vertices)


def build(coords, shells, starts, name="", info=None):
    """Board from cell coordinates, a shell per cell, and per player either one start coordinate or
    a list of start coordinates."""
    coords = list(coords)
    starts = [[s] if isinstance(s[0], int) else list(s) for s in starts]
    index = {c: i for i, c in enumerate(coords)}
    nbr = tuple(tuple(index.get(hg.neighbor(c, k), -1) for k in range(6)) for c in coords)
    keys, vertices, corner_vertex = {}, [], []
    for c in coords:
        row = []
        for k in range(6):
            key = hg.corner_key(c, k)
            cells = tuple(sorted(index[x] for x in key if x in index))
            if len(cells) < 2:          # a corner touching one cell is not shared: no beacon
                row.append(-1)
                continue
            if key not in keys:
                keys[key] = len(vertices)
                vertices.append(cells)
            row.append(keys[key])
        corner_vertex.append(tuple(row))
    board = Board(tuple(coords), nbr, tuple(vertices), tuple(corner_vertex), tuple(shells),
                  tuple(tuple(index[c] for c in group) for group in starts), name, info or {})
    validate(board)
    return board


def components(board, removed=()):
    """Connected groups of cells, ignoring the cells in `removed`."""
    removed, seen, groups = set(removed), set(removed), []
    for s in range(board.n_cells):
        if s in seen:
            continue
        group, todo = [], [s]
        seen.add(s)
        while todo:
            u = todo.pop()
            group.append(u)
            for v in board.nbr[u]:
                if v >= 0 and v not in seen:
                    seen.add(v)
                    todo.append(v)
        groups.append(group)
    return groups


def validate(board):
    assert board.n_cells >= 2, "a board needs at least two cells"
    assert len(components(board)) == 1, "the board must be one connected group of cells"
    for i, row in enumerate(board.corner_vertex):
        assert any(v >= 0 for v in row), f"cell {board.coords[i]} touches no beacon position"
    assert all(0 <= s < 64 for s in board.shells)
    cells = [c for group in board.starts for c in group]
    assert len(set(cells)) == len(cells), "start cells must differ"


def path_distances(board, src):
    """Steps from the nearest of the source cells (one cell index or several)."""
    src = [src] if isinstance(src, int) else list(src)
    d = [-1] * board.n_cells
    for s in src:
        d[s] = 0
    q = deque(src)
    while q:
        u = q.popleft()
        for v in board.nbr[u]:
            if v >= 0 and d[v] < 0:
                d[v] = d[u] + 1
                q.append(v)
    return d


def chokepoints(board):
    """Cells whose removal splits the board."""
    return [i for i in range(board.n_cells) if len(components(board, [i])) > 1]


def start_entrances(board, player, radius=1):
    """Edges leading out of a start area (the cells within `radius` steps of the start cells)."""
    d = path_distances(board, board.starts[player])
    area = {i for i in range(board.n_cells) if 0 <= d[i] <= radius}
    return sum(1 for u in area for v in board.nbr[u] if v >= 0 and v not in area)


def balance(board):
    """Share of cells closer to start 0 minus share closer to start 1 (by path)."""
    d0, d1 = path_distances(board, board.starts[0]), path_distances(board, board.starts[1])
    a = sum(x < y for x, y in zip(d0, d1))
    b = sum(y < x for x, y in zip(d0, d1))
    return (a - b) / board.n_cells


def hex_board(radius, starts=None, shells=None, seed=0):
    """Plain hexagon. Default starts sit one cell in from the top and bottom edges, and the
    default shells are random but point-symmetric, so both starts see the same board."""
    coords = hg.disc(radius)
    if starts is None:
        starts = [(0, -(radius - 1)), (0, radius - 1)] if radius > 1 else [(0, -1), (0, 1)]
    if shells is None:
        rng, shells_by = random.Random(seed), {}
        for c in sorted(coords):
            if c in shells_by:
                continue
            m = hg.mirror(c)
            s = rng.choice(SYMMETRIC_SHELLS) if m == c else rng.randrange(64)
            shells_by[c], shells_by[m] = s, rot(s, 3)
        shells = [shells_by[c] for c in coords]
    return build(coords, shells, starts, name=f"hex-r{radius}")


@dataclass(frozen=True)
class GenParams:
    radius: int = 4             # the field is cut from a hex disc of this radius
    gap: float = 0.25           # share of the disc removed as gaps, walls, and ragged edges
    blobs: int = 4              # round gaps
    walls: int = 2              # long thin gaps, which make narrow passages around their ends
    ragged: float = 0.9         # how strongly the outline is eaten away (0 keeps the disc's outline)
    start_depth: int = 1        # starts sit this many cells in from the edge of the disc
    start_cells: int = 3        # 1, or 3 mutually adjacent cells meeting at one corner
    keep_radius: int = 1        # cells this close to a start are never removed
    sym_radius: int = 2         # within this distance of a start, everything is mirrored
    falloff: float = 1.5        # beyond it, the chance of mirroring decays over this many cells
    balance_tol: float = 0.08   # largest allowed |balance(board)|
    detour: float = 1.75        # largest allowed path distance between starts / straight distance
    max_tries: int = 500


def mirror_weight(d, p):
    """How strongly a cell at distance d from the nearer start is mirrored: 1 near the starts,
    decaying toward 0 farther out."""
    return 1.0 if d <= p.sym_radius else math.exp(-(d - p.sym_radius) / p.falloff)


def _gap_field(cells, p, rng):
    """Random smooth field over the disc; the highest values become gaps."""
    pos = {c: hg.pixel(c, 1 / math.sqrt(3)) for c in cells}     # neighbors one unit apart
    span = p.radius
    f = {c: 0.12 * rng.random() + p.ragged * rng.random() * (hg.distance(c, (0, 0)) / p.radius) ** 3 for c in cells}
    for _ in range(p.blobs):
        cx, cy = rng.uniform(-span, span), rng.uniform(-span, span)
        sig = rng.uniform(0.7, 1.6)
        for c, (x, y) in pos.items():
            f[c] += math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig * sig))
    for _ in range(p.walls):
        ang, length = rng.uniform(0, math.pi), rng.uniform(2.0, max(2.5, p.radius))
        cx, cy = rng.uniform(-span, span) * 0.7, rng.uniform(-span, span) * 0.7
        ax, ay = cx - length / 2 * math.cos(ang), cy - length / 2 * math.sin(ang)
        bx, by = cx + length / 2 * math.cos(ang), cy + length / 2 * math.sin(ang)
        for c, (x, y) in pos.items():
            t = max(0.0, min(1.0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / (length * length)))
            dx, dy = x - (ax + t * (bx - ax)), y - (ay + t * (by - ay))
            f[c] += 1.2 * math.exp(-(dx * dx + dy * dy) / (2 * 0.35 ** 2))
    return f


def _try_generate(p, rng):
    disc = hg.disc(p.radius)
    ring = [c for c in disc if hg.distance(c, (0, 0)) == p.radius - p.start_depth]
    s0 = rng.choice(ring)
    s1 = hg.mirror(s0)
    group0 = [s0]
    if p.start_cells == 3:      # the start cell and the two neighbors that meet it at one corner
        disc_set = set(disc)
        corners = [k for k in range(6) if hg.neighbor(s0, k) in disc_set and hg.neighbor(s0, k + 1) in disc_set]
        k = rng.choice(corners)
        group0 = [s0, hg.neighbor(s0, k), hg.neighbor(s0, k + 1)]
    group1 = [hg.mirror(c) for c in group0]
    home = group0 + group1
    near = {c: min(hg.distance(c, h) for h in home) for c in disc}
    w = {c: mirror_weight(near[c], p) for c in disc}

    f = _gap_field(disc, p, rng)
    f = {c: w[c] * (f[c] + f[hg.mirror(c)]) / 2 + (1 - w[c]) * f[c] for c in disc}
    removable = sorted((c for c in disc if near[c] > p.keep_radius), key=lambda c: -f[c])
    n_gap = int(round(p.gap * len(disc)))
    if n_gap:
        cut = f[removable[min(n_gap, len(removable) - 1)]]
        gone = {c for c in removable if f[c] > cut}     # mirrored pairs share a value: both go or both stay
    else:
        gone = set()
    kept = [c for c in disc if c not in gone]

    # keep the group of cells connected to the starts; anything cut off becomes a gap too
    kept_set, seen, todo = set(kept), {s0}, [s0]
    while todo:
        u = todo.pop()
        for k in range(6):
            v = hg.neighbor(u, k)
            if v in kept_set and v not in seen:
                seen.add(v)
                todo.append(v)
    if s1 not in seen:
        return None
    cells = [c for c in kept if c in seen]

    shells = {}
    for c in sorted(cells):
        if c in shells:
            continue
        m = hg.mirror(c)
        mirrored = rng.random() < w[c]
        if m == c:
            shells[c] = rng.choice(SYMMETRIC_SHELLS) if mirrored else rng.randrange(64)
            continue
        shells[c] = rng.randrange(64)
        if m in seen:
            shells[m] = rot(shells[c], 3) if mirrored else rng.randrange(64)

    board = build(cells, [shells[c] for c in cells], [group0, group1])
    straight = min(hg.distance(a, b) for a in group0 for b in group1)
    d0 = path_distances(board, board.starts[0])
    d = min(d0[c] for c in board.starts[1])
    bal = balance(board)
    if d > p.detour * straight or abs(bal) > p.balance_tol:
        return None
    return board, {"cells": board.n_cells, "gaps": len(disc) - board.n_cells, "start_path": d,
                   "start_straight": straight, "balance": round(bal, 3),
                   "chokepoints": len(chokepoints(board)),
                   "entrances": [start_entrances(board, 0), start_entrances(board, 1)]}


def generate(seed, params=GenParams()):
    """A 2-player board. The same seed and params always give the same board."""
    rng = random.Random(seed)
    for tries in range(1, params.max_tries + 1):
        out = _try_generate(params, rng)
        if out:
            board, info = out
            info.update(seed=seed, tries=tries)
            return Board(board.coords, board.nbr, board.vertices, board.corner_vertex, board.shells,
                         board.starts, f"gen-{seed}", info)
    raise RuntimeError(f"no board met the constraints in {params.max_tries} tries")
