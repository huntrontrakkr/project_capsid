"""The JAX engine must agree with the reference engine: same legal moves (at the bucketed
amounts) and the same state after every move, over random games on generated boards."""
import random

import jax
import numpy as np
import pytest

from capsid import rules as R
from capsid.board import GenParams, generate
from capsid.jax_engine import KINDS, Engine, Layout

CFG = R.Config(allow_pay=False, max_actions_per_turn=10, max_turns_per_round=12)


class Pair:
    """A game in both engines, with the index maps between them."""

    def __init__(self, seed, radius=3):
        self.layout = L = Layout(radius)
        self.board = generate(seed, GenParams(radius=radius))
        self.eng = Engine(L, CFG)
        self.legal = jax.jit(self.eng.legal)
        self.step = jax.jit(self.eng.step)
        self.js = self.eng.init(*L.board_arrays(self.board))
        self.rs = R.new_game(self.board, CFG)
        self.slot = [L.index[c] for c in self.board.coords]               # ref cell -> slot
        self.cell = {s: i for i, s in enumerate(self.slot)}                 # slot -> ref cell
        self.vertex = {}                                                    # layout vertex -> ref vertex
        for i, row in enumerate(self.board.corner_vertex):
            for k, v in enumerate(row):
                if v >= 0:
                    self.vertex[int(L.corner_vertex[self.slot[i], k])] = v

    def to_ref(self, a):
        L, s = self.layout, self.js
        kind = KINDS[int(L.kind(a))]
        me = int(s.player)
        stores = np.asarray(s.stores)
        if kind == "transfer":
            i = a - L.T0
            c, e, b = i // 18, (i // 3) % 6, i % 3
            nbr, _, _ = self.eng.view(s.mask)
            nmax = int(self.eng._transfer_amounts(s, nbr)[0][c, e])
            return R.Transfer(self.cell[c], e, [1, nmax // 2, nmax][b])
        if kind == "destroy":
            i = a - L.D0
            c, b = i // 2, i % 2
            return R.Destroy(self.cell[c], 1 - me, 1 if b == 0 else int(min(stores[c, me], stores[c, 1 - me])))
        if kind == "shift":
            i = a - L.S0
            return R.Shift(self.cell[i // 5], i % 5 + 1)
        if kind == "modify":
            i = a - L.M0
            return R.Modify(self.cell[i // 6], i % 6)
        if kind == "beacon":
            return R.Beacon(self.vertex[a - L.B0])
        return R.EndTurn()

    def bucketed_ref_legal(self):
        """Reference legal moves, keeping only the amounts the bucketed actions can express."""
        acts = R.legal_actions(self.rs)
        tmax, dmax = {}, {}
        for a in acts:
            if isinstance(a, R.Transfer):
                tmax[a.cell, a.edge] = max(tmax.get((a.cell, a.edge), 0), a.n)
            if isinstance(a, R.Destroy):
                dmax[a.cell] = max(dmax.get(a.cell, 0), a.n)
        keep = set()
        for a in acts:
            if isinstance(a, R.Transfer):
                m = tmax[a.cell, a.edge]
                if a.n not in (1, m) and not (a.n == m // 2 and a.n > 1):
                    continue
            if isinstance(a, R.Destroy) and a.n not in (1, dmax[a.cell]):
                continue
            keep.add(a)
        return keep

    def check_same_state(self):
        s, r = self.js, self.rs
        stores = np.asarray(s.stores)
        assert [list(stores[self.slot[i]]) for i in range(self.board.n_cells)] == r.stores
        assert [int(np.asarray(s.shells)[self.slot[i]]) for i in range(self.board.n_cells)] == r.shells
        beacons = np.asarray(s.beacons)
        assert all(int(beacons[lv]) == r.beacons[rv] for lv, rv in self.vertex.items())
        assert list(np.asarray(s.tally)) == r.tally
        assert list(np.asarray(s.active)) == r.active
        assert bool(s.done) == (r.phase == "over")
        if not bool(s.done):
            assert int(s.round) == r.round and int(s.player) == r.player


@pytest.mark.parametrize("seed", range(3))
def test_jax_engine_matches_reference(seed):
    g = Pair(seed)
    rng = random.Random(seed)
    steps = 0
    while not bool(g.js.done):
        legal = np.flatnonzero(np.asarray(g.legal(g.js)))
        conv = {g.to_ref(int(a)) for a in legal}
        assert len(conv) == len(legal), "two bucketed actions mapped to the same move"
        assert conv == g.bucketed_ref_legal()
        ends = [a for a in legal if a == g.layout.END]
        a = int(ends[0]) if rng.random() < 0.15 else int(rng.choice(list(legal)))
        g.rs = R.apply(g.rs, g.to_ref(a))
        g.js = g.step(g.js, a)
        g.check_same_state()
        steps += 1
        assert steps < 5000
    assert g.rs.phase == "over"
