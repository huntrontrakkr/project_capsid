"""Shells and crossing costs (game_rules.md, Shell and Calculating Costs).

A shell is a 6-bit ring: bit k is the node at corner k. Rotating clockwise by one moves the
unit at node k to node k+1.
"""
from math import gcd


def rot(s, k=1):
    """Rotate shell s clockwise by k positions (negative k is counterclockwise)."""
    k %= 6
    return ((s << k) | (s >> (6 - k))) & 63 if k else s


def node(s, k):
    return (s >> (k % 6)) & 1


def weight(s):
    return bin(s).count("1")


def hamming_across(a, b, d):
    """Pairs that differ when cell A's edge d is shared with cell B.

    Reflection across the shared edge pairs node d+k of A with node d+2-k of B."""
    return sum(node(a, d + k) != node(b, d + 2 - k) for k in range(6))


def reduce_cost(h, total=6):
    """H:~H reduced by the gcd. Free is (0, 1) and blocked is (1, 0)."""
    g = gcd(h, total - h)
    return h // g, (total - h) // g


def crossing_cost(a, b, d):
    """Cost p:m for moving from a cell with shell a, across its edge d, into a cell with shell b."""
    return reduce_cost(hamming_across(a, b, d))


def transfer_payment(p, m, n):
    """Units paid to move n units at cost p:m, or None if the crossing is blocked."""
    if m == 0:
        return None
    if p == 0:
        return 0
    return -(-n // m) * p


# lookup table COST[a][b][d], 64 x 64 x 6
COST = [[[crossing_cost(a, b, d) for d in range(6)] for b in range(64)] for a in range(64)]
