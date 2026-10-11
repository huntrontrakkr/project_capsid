"""SVG drawings of boards and game states."""
import math

from . import hexgrid as hg

PLAYER_COLORS = ["#2F7DD1", "#D85A30", "#1D9E75", "#BA7517", "#7F5AB7", "#C2477A"]
INK, MUTED, CELL, PAGE = "#1f1e1d", "#8b8a85", "#f3f2ec", "#faf9f5"


def board_svg(board, state=None, size=22.0, title=None, show_disc=None):
    """Cells as flat-topped hexes, shell nodes as dots on the inner ring pointing at the cell's
    corners, beacon positions as small diamonds. Start cells are outlined in the player's color.
    With a state, stores are written in the cells and beacons are filled with the owner's color.
    `show_disc` (a radius) draws the missing cells of that disc as faint outlines."""
    centers = [hg.pixel(c, size) for c in board.coords]
    ghost = []
    if show_disc is not None:
        present = set(board.coords)
        ghost = [hg.pixel(c, size) for c in hg.disc(show_disc) if c not in present]
    pts = centers + ghost
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = size * 1.3
    x0, y0 = min(xs) - pad, min(ys) - pad
    w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad + (18 if title else 0)
    ty = 18 if title else 0

    def P(x, y):
        return f"{x - x0:.1f},{y - y0 + ty:.1f}"

    def hexagon(cx, cy, r):
        return " ".join(P(cx + r * math.cos(math.radians(60 * k)), cy + r * math.sin(math.radians(60 * k)))
                        for k in range(6))

    def corner(cx, cy, k, r):        # corner k sits between edge k and k+1: 30 + 60k degrees clockwise from up
        a = math.radians(30 + 60 * k)
        return cx + r * math.sin(a), cy - r * math.cos(a)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           f'font-family="system-ui,-apple-system,Segoe UI,sans-serif"><rect width="100%" height="100%" fill="{PAGE}"/>']
    if title:
        out.append(f'<text x="8" y="14" font-size="12" fill="{INK}">{title}</text>')
    for cx, cy in ghost:
        out.append(f'<polygon points="{hexagon(cx, cy, size * 0.96)}" fill="none" stroke="{MUTED}" '
                   f'stroke-width=".6" stroke-dasharray="2 3"/>')
    starts = {s: p for p, s in enumerate(board.starts)}
    for i, (cx, cy) in enumerate(centers):
        stroke, sw = ("#6b6a65", 1.0)
        if i in starts:
            stroke, sw = PLAYER_COLORS[starts[i]], 2.6
        out.append(f'<polygon points="{hexagon(cx, cy, size * 0.97)}" fill="{CELL}" stroke="{stroke}" stroke-width="{sw}"/>')
        shell = state.shells[i] if state is not None else board.shells[i]
        for k in range(6):
            x, y = corner(cx, cy, k, size * 0.55)
            if (shell >> k) & 1:
                out.append(f'<circle cx="{x - x0:.1f}" cy="{y - y0 + ty:.1f}" r="{size * 0.13:.1f}" fill="{INK}"/>')
            else:
                out.append(f'<circle cx="{x - x0:.1f}" cy="{y - y0 + ty:.1f}" r="{size * 0.11:.1f}" fill="none" '
                           f'stroke="{MUTED}" stroke-width=".7"/>')
        if state is not None:
            units = [(p, n) for p, n in enumerate(state.stores[i]) if n]
            for j, (p, n) in enumerate(units):
                dy = (j - (len(units) - 1) / 2) * size * 0.42
                out.append(f'<text x="{cx - x0:.1f}" y="{cy - y0 + ty + dy + size * 0.16:.1f}" text-anchor="middle" '
                           f'font-size="{size * 0.42:.1f}" font-weight="600" fill="{PLAYER_COLORS[p]}">{n}</text>')
    for v, cells in enumerate(board.vertices):
        i = cells[0]
        k = board.corner_vertex[i].index(v)
        x, y = corner(*centers[i], k, size)
        owner = state.beacons[v] if state is not None else -1
        r = size * (0.2 if owner >= 0 else 0.11)
        fill = PLAYER_COLORS[owner] if owner >= 0 else PAGE
        out.append(f'<polygon points="{P(x, y - r)} {P(x + r, y)} {P(x, y + r)} {P(x - r, y)}" fill="{fill}" '
                   f'stroke="{MUTED if owner < 0 else INK}" stroke-width=".7"/>')
    out.append("</svg>")
    return "".join(out)
