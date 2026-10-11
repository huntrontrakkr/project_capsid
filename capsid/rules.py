"""The rules of game_rules.md as plain Python: the reference engine.

Readable first, fast never: the batched JAX engine is checked against this one. A game is a
State; `legal_actions` lists what the player to move may do, and `apply` returns the next
State. Rules that are still open (open_questions.md) are settings in Config.

Turn structure. In the active phase the player to move makes moves one at a time and finishes
with EndTurn (an EndTurn with no moves is a pass). When the active phase ends, harvest runs on
its own; then each player with units in their coffer places them with Place moves and EndTurn.
"""
import copy
from dataclasses import dataclass, field

from .shell import COST, rot, transfer_payment

TRANSFER, DESTROY, SHELL_OPS, BEACON, PAY = range(5)
MOVE_TYPE_NAMES = ("transfer", "destroy", "shell ops", "beacon", "pay")


@dataclass(frozen=True)
class Config:
    players: int = 2
    rounds: int = 6
    start_units: int = 5               # units on each of a player's start cells (no bid in training v1)
    shift_cost: int = 1                # a shift by a player who is not dominant
    modify_cost: int = 1
    beacon_build_cost: int = 2         # per adjacent store (open question 2)
    beacon_capture_cost: int = 1       # per adjacent store
    multi_transfer: bool = True        # units may cross more than once per turn (open question 1)
    allow_pay: bool = True             # coffer payments; off for training v1 (2 players, no placement step)
    move_groups: tuple = (0, 0, 1, 2, 3)   # group of each move type: transfer, destroy, shell ops, beacon, pay
    groups_per_turn: int = 2
    ordered_groups: bool = True        # finish one group before starting the next, with no going back
    early_end: bool = False            # end when one active player remains (open question 5)
    max_actions_per_turn: int = 64     # safety cap: after this only EndTurn is legal
    max_turns_per_round: int = 200     # safety cap: the round goes to harvest and is flagged


# Actions are frozen dataclasses, not tuples, so moves of different kinds never compare equal
# (as tuples, Shift(3, 1) would equal Modify(3, 1)).
@dataclass(frozen=True)
class Transfer:
    cell: int
    edge: int
    n: int


@dataclass(frozen=True)
class Destroy:
    cell: int
    target: int
    n: int


@dataclass(frozen=True)
class Shift:
    cell: int
    k: int          # 1 to 5 steps clockwise


@dataclass(frozen=True)
class Modify:
    cell: int
    node: int       # removes the unit there, or adds a neutral unit if the node is vacant


@dataclass(frozen=True)
class Beacon:
    vertex: int     # build if empty, capture if another player owns it


@dataclass(frozen=True)
class Pay:
    cell: int
    target: int
    n: int


@dataclass(frozen=True)
class Place:
    cell: int
    n: int


@dataclass(frozen=True)
class EndTurn:
    pass


MOVE_TYPE = {Transfer: TRANSFER, Destroy: DESTROY, Shift: SHELL_OPS, Modify: SHELL_OPS, Beacon: BEACON, Pay: PAY}


@dataclass
class State:
    board: object
    config: Config
    stores: list            # [cell][player] units
    shells: list            # [cell] 6-bit shell
    beacons: list           # [vertex] owner, -1 if empty
    coffers: list           # [player]
    tally: list             # [player]
    active: list            # [player] still taking turns
    round: int = 0          # 0-based
    phase: str = "active"   # "active", "place", or "over"
    order: list = field(default_factory=list)       # turn order this round
    seat: int = 0                                   # index into order of the player to move
    groups_used: list = field(default_factory=list)  # move groups used this turn, in order
    spent: bool = False                             # units expended this turn
    moves_this_turn: int = 0
    arrived: list = field(default_factory=list)     # [cell] units moved in this turn (for multi_transfer off)
    streak: int = 0                                 # cost-free turns or passes in a row
    turns_this_round: int = 0
    place_queue: list = field(default_factory=list)  # players still to place coffer units
    capped_rounds: int = 0                           # rounds that hit max_turns_per_round
    units_created: int = 0                           # harvest income so far
    units_discarded: int = 0                         # units paid or destroyed so far

    def copy(self):
        s = copy.copy(self)          # the board and config are shared; everything that changes is copied
        s.stores = [row[:] for row in self.stores]
        for name in ("shells", "beacons", "coffers", "tally", "active", "order", "arrived", "place_queue"):
            setattr(s, name, getattr(self, name)[:])
        s.groups_used = list(self.groups_used)
        return s

    @property
    def player(self):
        """The player to move, or None when the game is over."""
        if self.phase == "active":
            return self.order[self.seat]
        if self.phase == "place":
            return self.place_queue[0]
        return None


def new_game(board, config=Config()):
    P, N = config.players, board.n_cells
    assert len(board.starts) == P, "the board needs start cells for every player"
    stores = [[0] * P for _ in range(N)]
    for p, group in enumerate(board.starts):
        for cell in group:
            stores[cell][p] = config.start_units
    return State(board, config, stores, list(board.shells), [-1] * board.n_vertices, [0] * P, [0] * P,
                 [True] * P, order=list(range(P)), arrived=[0] * N)


# ------------------------------------------------------------------ queries
def dominant(state, cell):
    """The player with strictly more units in the cell than any other, or -1."""
    row = state.stores[cell]
    top = max(row)
    if top == 0 or row.count(top) > 1:
        return -1
    return row.index(top)


def crossing(state, cell, edge):
    """(p, m) for moving from `cell` across `edge`, or None if there is no cell there."""
    nb = state.board.nbr[cell][edge]
    if nb < 0:
        return None
    return COST[state.shells[cell]][state.shells[nb]][edge]


def type_allowed(state, t):
    """A turn uses at most `groups_per_turn` move groups. With ordered groups, the player finishes
    one group before starting the next and cannot go back to an earlier one."""
    cfg, used = state.config, state.groups_used
    g = cfg.move_groups[t]
    if not used:
        return True
    if cfg.ordered_groups:
        return g == used[-1] or (g not in used and len(used) < cfg.groups_per_turn)
    return g in used or len(used) < cfg.groups_per_turn


def is_legal(state, a):
    return a in legal_actions(state)


def legal_actions(state):
    """Every legal action for the player to move, with every exact amount."""
    if state.phase == "over":
        return []
    i, b, cfg = state.player, state.board, state.config
    acts = [EndTurn()]
    if state.phase == "place":
        for c in range(b.n_cells):
            if state.stores[c][i] > 0:
                acts += [Place(c, n) for n in range(1, state.coffers[i] + 1)]
        return acts
    if state.moves_this_turn >= cfg.max_actions_per_turn:
        return acts
    P = cfg.players
    for c in range(b.n_cells):
        mine = state.stores[c][i]
        if type_allowed(state, TRANSFER) and mine > 0:
            movable = mine if cfg.multi_transfer else mine - state.arrived[c]
            for e in range(6):
                pm = crossing(state, c, e)
                if pm is None or pm[1] == 0:
                    continue
                for n in range(1, movable + 1):
                    if n + transfer_payment(*pm, n) <= mine:
                        acts.append(Transfer(c, e, n))
        if type_allowed(state, DESTROY) and mine > 0:
            for j in range(P):
                if j != i and state.stores[c][j] > 0:
                    acts += [Destroy(c, j, n) for n in range(1, min(mine, state.stores[c][j]) + 1)]
        if type_allowed(state, SHELL_OPS):
            if dominant(state, c) == i or (mine > 0 and mine >= cfg.shift_cost):
                acts += [Shift(c, k) for k in range(1, 6)]
            if mine >= max(1, cfg.modify_cost):
                acts += [Modify(c, k) for k in range(6)]
        if cfg.allow_pay and type_allowed(state, PAY) and mine > 0:
            for j in range(P):
                if j != i and state.active[j]:
                    acts += [Pay(c, j, n) for n in range(1, mine + 1)]
    if type_allowed(state, BEACON):
        for v, cells in enumerate(b.vertices):
            owner = state.beacons[v]
            if owner == i or any(dominant(state, c) != i for c in cells):
                continue
            cost = cfg.beacon_build_cost if owner < 0 else cfg.beacon_capture_cost
            if all(state.stores[c][i] >= cost for c in cells):
                acts.append(Beacon(v))
    return acts


# ------------------------------------------------------------------ moves
def apply(state, a):
    """The state after action `a`. Raises ValueError if `a` is not legal."""
    if not is_legal(state, a):
        raise ValueError(f"illegal action {a!r} for player {state.player}")
    s = state.copy()
    i, cfg = s.player, s.config
    if isinstance(a, EndTurn):
        _end_turn(s) if s.phase == "active" else _end_placement(s)
        return s
    if isinstance(a, Place):
        s.stores[a.cell][i] += a.n
        s.coffers[i] -= a.n
        return s

    def spend(cell, n):
        s.stores[cell][i] -= n
        s.units_discarded += n
        if n:
            s.spent = True

    if isinstance(a, Transfer):
        nb = s.board.nbr[a.cell][a.edge]
        spend(a.cell, transfer_payment(*crossing(s, a.cell, a.edge), a.n))
        s.stores[a.cell][i] -= a.n
        s.stores[nb][i] += a.n
        s.arrived[nb] += a.n
    elif isinstance(a, Destroy):
        spend(a.cell, a.n)
        s.stores[a.cell][a.target] -= a.n
        s.units_discarded += a.n
    elif isinstance(a, Shift):
        if dominant(s, a.cell) != i:
            spend(a.cell, cfg.shift_cost)
        s.shells[a.cell] = rot(s.shells[a.cell], a.k)
    elif isinstance(a, Modify):
        spend(a.cell, cfg.modify_cost)
        s.shells[a.cell] ^= 1 << a.node
    elif isinstance(a, Beacon):
        cost = cfg.beacon_build_cost if s.beacons[a.vertex] < 0 else cfg.beacon_capture_cost
        for c in s.board.vertices[a.vertex]:
            spend(c, cost)
        s.beacons[a.vertex] = i
    elif isinstance(a, Pay):
        s.stores[a.cell][i] -= a.n
        s.coffers[a.target] += a.n
        s.spent = True
    g = cfg.move_groups[MOVE_TYPE[type(a)]]
    if g not in s.groups_used:
        s.groups_used.append(g)
    s.moves_this_turn += 1
    return s


def _n_active(s):
    return sum(s.active)


def _start_turn(s):
    s.groups_used, s.spent, s.moves_this_turn = [], False, 0
    s.arrived = [0] * s.board.n_cells


def _end_turn(s):
    s.streak = 0 if s.spent else s.streak + 1
    s.turns_this_round += 1
    capped = s.turns_this_round >= s.config.max_turns_per_round
    if s.streak >= _n_active(s) or capped:
        s.capped_rounds += capped and s.streak < _n_active(s)
        _harvest(s)
        return
    seat = s.seat
    while True:
        seat = (seat + 1) % len(s.order)
        if s.active[s.order[seat]]:
            break
    s.seat = seat
    _start_turn(s)


def _harvest(s):
    b, P = s.board, s.config.players
    for owner in s.beacons:
        if owner >= 0:
            s.tally[owner] += 1
    # all income comes from the board as it stood when harvest began
    income = [[0] * P for _ in range(b.n_cells)]
    for c in range(b.n_cells):
        d = dominant(s, c)
        if d < 0:
            continue
        income[c][d] = 1 + sum(1 for nb in b.nbr[c] if nb >= 0 and s.stores[nb][d] > 0)
    for c in range(b.n_cells):
        for p in range(P):
            s.stores[c][p] += income[c][p]
            s.units_created += income[c][p]
    # next round's order: least to most in the coffer, ties to the earlier player this round
    s.order = sorted(s.order, key=lambda p: s.coffers[p])
    s.place_queue = []
    for p in s.order:
        if s.coffers[p] == 0:
            continue
        if s.active[p] and any(s.stores[c][p] > 0 for c in range(b.n_cells)):
            s.place_queue.append(p)
        else:
            s.units_discarded += s.coffers[p]
            s.coffers[p] = 0
    s.phase = "place"
    if not s.place_queue:
        _finish_round(s)


def _end_placement(s):
    p = s.place_queue.pop(0)
    s.units_discarded += s.coffers[p]
    s.coffers[p] = 0
    if not s.place_queue:
        _finish_round(s)


def _finish_round(s):
    b = s.board
    for p in range(s.config.players):
        if s.active[p] and not any(s.stores[c][p] > 0 for c in range(b.n_cells)):
            s.active[p] = False
    s.round += 1
    if s.round >= s.config.rounds or _n_active(s) == 0 or (s.config.early_end and _n_active(s) <= 1):
        s.phase = "over"
        return
    s.phase = "active"
    s.streak, s.turns_this_round = 0, 0
    s.seat = next(k for k, p in enumerate(s.order) if s.active[p])
    _start_turn(s)


# ------------------------------------------------------------------ results
def dominant_cells(state, p):
    return sum(1 for c in range(state.board.n_cells) if dominant(state, c) == p)


def ranking_key(state, p):
    """Highest tally wins; ties go to the most dominant cells (End of Game)."""
    return (state.tally[p], dominant_cells(state, p))


def outcome(state):
    """For 2 players: +1 if player 0 won, -1 if player 1 won, 0 for a draw."""
    k0, k1 = ranking_key(state, 0), ranking_key(state, 1)
    return (k0 > k1) - (k0 < k1)
