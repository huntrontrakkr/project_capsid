"""Tests for the reference engine: the worked examples in game_rules.md, the decisions in
open_questions.md, and invariants checked over random games."""
import math
import random

import pytest

from capsid import hexgrid as hg
from capsid.board import GenParams, balance, build, components, generate, hex_board
from capsid.rules import (BEACON, Beacon, Config, Destroy, EndTurn, Modify, Pay, Place, Shift, Transfer, apply,
                          dominant, legal_actions, new_game, outcome)
from capsid.shell import COST, hamming_across, rot, transfer_payment

A, B = (0, 0), (0, -1)          # B sits across A's top edge (edge 0)
FREE = 0                         # two empty shells match on every pair: a free crossing


def pair_game(sa=FREE, sb=FREE, a=(0, 0), b=(0, 0), cfg=Config()):
    """Two cells, A below B. a and b are the (player 0, player 1) units in each store."""
    s = new_game(build([A, B], [sa, sb], [A, B]), cfg)
    s.stores = [list(a), list(b)]
    return s


def play(s, *actions):
    for a in actions:
        s = apply(s, a)
    return s


# ------------------------------------------------------------------ crossing costs
def test_only_these_costs_occur_on_hex_cells():
    seen = {COST[a][b][d] for a in range(64) for b in range(64) for d in range(6)}
    assert seen == {(0, 1), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (1, 0)}


def test_cost_is_the_same_in_both_directions():
    for a in range(64):
        for b in range(64):
            for d in range(6):
                assert COST[a][b][d] == COST[b][a][(d + 3) % 6]


def test_pairing_is_the_reflection_across_the_shared_edge():
    """Node d+k of A pairs with node d+2-k of B: check against the geometry directly."""
    def node_pos(cell, k):
        x, y = hg.pixel(cell)
        a = math.radians(30 + 60 * k)
        return x + 0.55 * math.sin(a), y - 0.55 * math.cos(a)
    for d in range(6):
        nb = hg.neighbor(A, d)
        (mx, my), (bx, by) = hg.pixel(A), hg.pixel(nb)
        ux, uy = (bx - mx) / math.dist((mx, my), (bx, by)), (by - my) / math.dist((mx, my), (bx, by))
        midx, midy = (mx + bx) / 2, (my + by) / 2
        for k in range(6):
            px, py = node_pos(A, k)
            t = (px - midx) * ux + (py - midy) * uy
            rx, ry = px - 2 * t * ux, py - 2 * t * uy
            j = min(range(6), key=lambda j: math.dist((rx, ry), node_pos(nb, j)))
            assert j == (2 * d + 2 - k) % 6       # k = d + k' pairs with d + 2 - k'


def test_figure_example():
    """figures/crossing_pairing.svg: A's top edge shared, two pairs differ, cost 1:2."""
    a, b = 0b000011, 0b010100
    assert hamming_across(a, b, 0) == 2
    assert COST[a][b][0] == (1, 2)


def test_transfer_is_a_rate():
    """The 13-unit example in game_rules.md at 2:5."""
    assert 5 + transfer_payment(2, 5, 5) == 7
    assert 9 + transfer_payment(2, 5, 9) == 13
    assert 10 + transfer_payment(2, 5, 10) == 14
    assert transfer_payment(2, 5, 3) == 2          # a partial block still costs the full p
    assert transfer_payment(0, 1, 50) == 0         # free
    assert transfer_payment(1, 0, 1) is None       # blocked


def test_transfer_in_the_engine():
    s = pair_game(0b000011, 0b010100, a=(13, 0))    # cost 1:2 across A's top edge
    moves = {a.n for a in legal_actions(s) if isinstance(a, Transfer) and a.cell == 0}
    assert max(moves) == 8                          # 8 + ceil(8/2) = 12; 9 would need 14
    s = play(s, Transfer(0, 0, 5))
    assert s.stores == [[5, 0], [5, 0]] and s.spent  # 5 moved, 3 paid


def test_blocked_crossing_allows_no_transfer():
    s = pair_game(0b111111, 0b000000, a=(5, 0))
    assert not any(isinstance(a, Transfer) for a in legal_actions(s))


# ------------------------------------------------------------------ moves
def test_destroy_example():
    s = pair_game(a=(5, 3))
    s = play(s, Destroy(0, 1, 2))
    assert s.stores[0] == [3, 1]


def kinds(s):
    return {type(a) for a in legal_actions(s)} - {EndTurn}


def test_two_groups_per_turn_in_order():
    """Shells, then units (move and attack together), and no going back to the shells."""
    s = pair_game(a=(5, 3), b=(0, 2))
    s = play(s, Shift(0, 1))                       # group 1: shells
    assert kinds(s) == {Shift, Modify, Transfer, Destroy, Pay}   # (no beacon: player 1 holds B)
    s = play(s, Transfer(0, 0, 1), Destroy(0, 1, 1), Transfer(0, 0, 1))   # group 2: units, in any mix
    assert kinds(s) == {Transfer, Destroy}         # shells are closed now, and no third group


def test_units_then_shells_is_allowed():
    """Move in, become dominant, then rotate for free; but then no more moving."""
    s = pair_game(a=(5, 0), b=(0, 2))
    s = play(s, Transfer(0, 0, 3), Shift(1, 2))
    assert not s.spent
    assert kinds(s) == {Shift, Modify}


def test_unordered_groups_option_restores_the_old_rule():
    cfg = Config(move_groups=(0, 1, 2, 3, 4), ordered_groups=False)
    s = pair_game(a=(5, 3), b=(0, 2), cfg=cfg)
    s = play(s, Transfer(0, 0, 1), Destroy(0, 1, 1), Transfer(0, 0, 1))
    assert kinds(s) == {Transfer, Destroy}


def test_dominance_is_checked_live():
    """Move in, become dominant, then rotate for free in the same turn."""
    s = pair_game(a=(5, 0), b=(0, 2))
    assert dominant(s, 1) == 1
    s = play(s, Transfer(0, 0, 3))
    assert dominant(s, 1) == 0
    s = play(s, Shift(1, 2))
    assert s.stores[1] == [3, 2] and not s.spent    # the move and the rotation were both free


def test_shift_costs_one_without_dominance():
    s = pair_game(sb=0b000001, a=(0, 0), b=(1, 2))
    s = play(s, Shift(1, 1))
    assert s.shells[1] == 0b000010 and s.stores[1] == [0, 2] and s.spent


def test_anyone_with_units_can_modify():
    s = pair_game(sa=0b000001, a=(1, 4))
    assert dominant(s, 0) == 1
    s = play(s, Modify(0, 3))
    assert s.shells[0] == 0b001001 and s.stores[0] == [0, 4]     # neutral unit added, the paid unit discarded
    s2 = pair_game(sa=0b000001, a=(0, 4))
    assert not any(isinstance(a, Modify) and a.cell == 0 for a in legal_actions(s2))


def test_beacon_build_needs_dominance_in_every_adjacent_cell():
    s = pair_game(a=(3, 0), b=(2, 3))
    assert not any(isinstance(a, Beacon) for a in legal_actions(s))
    s = pair_game(a=(3, 0), b=(4, 3))
    s = play(s, Beacon(0))
    assert s.beacons[0] == 0
    assert s.stores == [[1, 0], [2, 3]]             # 2 from each adjacent store; the marker is free


def test_beacon_capture_costs_one_per_store():
    s = pair_game(a=(0, 3), b=(0, 3))
    s.beacons[0] = 0
    s = play(s, EndTurn(), Beacon(0))             # player 0 passes; player 1 captures
    assert s.beacons[0] == 1 and s.stores == [[0, 2], [0, 2]]


def test_payment_counts_as_spent():
    s = pair_game(a=(4, 0), b=(0, 1))
    s = play(s, Pay(0, 1, 2))
    assert s.coffers == [0, 2] and s.stores[0] == [2, 0] and s.spent


def test_multi_transfer_off_stops_units_moving_twice():
    s = pair_game(a=(4, 0), cfg=Config(multi_transfer=False))
    s = play(s, Transfer(0, 0, 4))
    assert not any(isinstance(a, Transfer) for a in legal_actions(s))
    s = pair_game(a=(4, 0))
    s = play(s, Transfer(0, 0, 4))
    assert any(isinstance(a, Transfer) for a in legal_actions(s))


# ------------------------------------------------------------------ phases
def test_active_phase_ends_after_a_cost_free_turn_from_each_player():
    s = pair_game(a=(4, 0), b=(0, 4))
    s = play(s, Pay(0, 1, 1), EndTurn())           # costly: the count stays at 0
    assert s.phase == "active" and s.player == 1
    s = play(s, EndTurn())                         # pass: 1
    assert s.phase == "active" and s.player == 0
    s = play(s, EndTurn())                         # pass: 2, so harvest
    assert s.phase == "place" and s.player == 1    # player 1 has a coffer to place


def test_harvest_income_is_computed_all_at_once():
    s = pair_game(a=(3, 0), b=(1, 2))
    s = play(s, EndTurn(), EndTurn())
    # A: player 0 dominant, +1, +1 for occupying B. B: player 1 dominant, +1, nothing in A.
    assert s.stores == [[5, 0], [1, 3]]
    assert s.round == 1 and s.phase == "active"


def test_turn_order_goes_from_least_to_most_in_the_coffer():
    s = pair_game(a=(4, 0), b=(0, 4))
    s = play(s, EndTurn(), Pay(1, 0, 2), EndTurn(), EndTurn(), EndTurn())
    assert s.phase == "place" and s.order == [1, 0]
    s = play(s, Place(0, 2), EndTurn())
    assert s.round == 1 and s.player == 1


def test_inactive_player_keeps_scoring_with_their_beacons():
    s = pair_game(a=(3, 0), b=(0, 0))
    s.beacons[0] = 1                               # player 1 has a beacon but no units
    s = play(s, EndTurn())
    assert s.player == 1                           # player 1 is still active until harvest decides
    s = play(s, EndTurn())
    assert s.round == 1 and s.active == [True, False] and s.tally == [0, 1]
    s = play(s, EndTurn())                         # only player 0 takes turns now
    assert s.round == 2 and s.tally == [0, 2]


def test_early_end_option():
    s = pair_game(a=(3, 0), cfg=Config(early_end=True))
    s = play(s, EndTurn(), EndTurn())
    assert s.phase == "over" and outcome(s) == 1   # tie on tally, player 0 has more dominant cells


# ------------------------------------------------------------------ random games
def random_policy(state, rng, allow_pay=True):
    acts = [a for a in legal_actions(state) if allow_pay or not isinstance(a, Pay)]
    if rng.random() < 0.2:
        return acts[0]                             # EndTurn
    kinds = {}
    for a in acts:
        kinds.setdefault(type(a), []).append(a)
    return rng.choice(rng.choice(list(kinds.values())))


@pytest.mark.parametrize("seed,allow_pay", [(k, k % 2 == 0) for k in range(8)])
def test_random_games_keep_the_invariants(seed, allow_pay):
    rng = random.Random(seed)
    s = new_game(generate(seed, GenParams(radius=3)), Config())
    start_total = sum(map(sum, s.stores))
    steps = 0
    while s.phase != "over":
        before_round, before_tally = s.round, list(s.tally)
        s = apply(s, random_policy(s, rng, allow_pay))
        steps += 1
        assert steps < 50_000, "the game should end"
        on_board = sum(map(sum, s.stores)) + sum(s.coffers)
        assert on_board + s.units_discarded == start_total + s.units_created
        assert min(min(r) for r in s.stores) >= 0 and min(s.coffers) >= 0
        assert all(0 <= x < 64 for x in s.shells)
        if s.tally != before_tally:                # the tally only moves at harvest
            assert s.phase in ("place", "over") or s.round != before_round
    assert s.round == 6 or not any(s.active)       # six rounds, or nobody has units left
    assert outcome(s) in (-1, 0, 1)


# ------------------------------------------------------------------ boards
def test_generated_boards_are_connected_and_balanced():
    for seed in range(20):
        b = generate(seed)
        assert len(components(b)) == 1
        assert abs(balance(b)) <= GenParams().balance_tol
        assert generate(seed).coords == b.coords and generate(seed).shells == b.shells


def test_start_areas_are_mirror_images():
    p = GenParams()
    for seed in range(20):
        b = generate(seed, p)
        cells = dict(zip(b.coords, b.shells))
        assert [hg.mirror(b.coords[c]) for c in b.starts[0]] == [b.coords[c] for c in b.starts[1]]
        s0, s1 = b.coords[b.starts[0][0]], b.coords[b.starts[1][0]]
        assert s1 == hg.mirror(s0)
        for c, sh in cells.items():
            if min(hg.distance(c, s0), hg.distance(c, s1)) <= p.sym_radius:
                assert cells.get(hg.mirror(c)) == rot(sh, 3)


def test_fully_mirrored_boards():
    p = GenParams(sym_radius=99)
    for seed in range(10):
        b = generate(seed, p)
        cells = dict(zip(b.coords, b.shells))
        assert all(cells.get(hg.mirror(c)) == rot(sh, 3) for c, sh in cells.items())


def test_hex_board_sizes():
    assert [hex_board(r).n_cells for r in (2, 3, 4)] == [19, 37, 61]


def test_moves_of_different_kinds_never_compare_equal():
    assert Shift(3, 1) != Modify(3, 1)
    s = pair_game(a=(1, 4), cfg=Config(shift_cost=2))   # player 0 can afford to modify cell 0 but not to shift it
    assert not any(isinstance(a, Shift) and a.cell == 0 for a in legal_actions(s))
    with pytest.raises(ValueError):
        apply(s, Shift(0, 1))


def test_thirteen_units_at_one_per_five():
    """Pay 1 for every 5 units moved, paid up front: moving 13 costs 3, so 16 leave the store."""
    assert transfer_payment(1, 5, 13) == 3
    a, b = 0b000000, 0b000001       # one pair differs across A's top edge: cost 1:5
    assert COST[a][b][0] == (1, 5)
    s = pair_game(a, b, a=(16, 0))
    s = play(s, Transfer(0, 0, 13))
    assert s.stores == [[0, 0], [13, 0]]
    s = pair_game(a, b, a=(15, 0))                  # with 15, moving 13 is not affordable
    assert max(x.n for x in legal_actions(s) if isinstance(x, Transfer) and x.cell == 0) == 12


def test_harvest_adjacency_on_a_line_of_three_cells():
    """Dominant: +1, and +1 for each adjacent cell the player occupies."""
    cells = [(0, 1), (0, 0), (0, -1)]               # a vertical line: bottom, middle, top
    s = new_game(build(cells, [0, 0, 0], [cells[0], cells[2]]), Config())
    s.stores = [[2, 0], [3, 1], [1, 2]]
    s = play(s, EndTurn(), EndTurn())
    # bottom: player 0 dominant, occupies the middle: +2. middle: player 0, occupies bottom and top: +3.
    # top: player 1 dominant, occupies the middle: +2.
    assert s.stores == [[4, 0], [6, 1], [1, 4]]
