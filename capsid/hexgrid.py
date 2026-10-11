"""Flat-topped hex grid in axial coordinates (q, r).

Edges are numbered clockwise from the top: 0 top, 1 upper right, 2 lower right, 3 bottom,
4 lower left, 5 upper left. Corner k is the corner between edge k and edge k+1, and shell
node k points at corner k (game_rules.md, Shell). The cell across edge k shares edge k+3.
"""
import math

DIRS = ((0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0))


def neighbor(c, k):
    dq, dr = DIRS[k % 6]
    return (c[0] + dq, c[1] + dr)


def distance(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def disc(radius):
    """All cells within `radius` of the origin (1, 7, 19, 37, 61, 91, ... cells)."""
    return [(q, r) for q in range(-radius, radius + 1)
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1)]


def corner_key(c, k):
    """The three cells that meet at corner k of cell c. Identifies the vertex uniquely."""
    return frozenset((c, neighbor(c, k), neighbor(c, k + 1)))


def mirror(c):
    """Point reflection through the origin. Edge k maps to edge k+3, corner k to corner k+3."""
    return (-c[0], -c[1])


def pixel(c, size=1.0):
    """Center of cell c for drawing, y pointing down."""
    q, r = c
    return (1.5 * size * q, math.sqrt(3) * size * (r + q / 2))
