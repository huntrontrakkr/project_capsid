"""Batched rules engine in JAX for self-play: 2 players, coffer payments off.

It follows capsid/rules.py move for move (tests/test_jax_engine.py replays random games through
both). Boards are padded to a fixed hex disc, the Layout: every generated board of that radius
is a mask over the disc's cells, so one compiled engine and one network serve every board.

Actions are human-sized choices rather than exact amounts:

    transfer  cell x edge x {1 unit, half the most affordable, the most affordable}
    destroy   cell x {1 unit, as many as possible}
    shift     cell x {1..5 steps clockwise}
    modify    cell x node
    beacon    vertex (build or capture)
    end turn
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import hexgrid as hg
from .rules import BEACON, DESTROY, SHELL_OPS, TRANSFER, Config
from .shell import COST, hamming_across

H_TAB = np.array([[[hamming_across(a, b, d) for d in range(6)] for b in range(64)] for a in range(64)], np.int32)
P_TAB = np.array([[[COST[a][b][d][0] for d in range(6)] for b in range(64)] for a in range(64)], np.int32)
M_TAB = np.array([[[COST[a][b][d][1] for d in range(6)] for b in range(64)] for a in range(64)], np.int32)
KINDS = ("transfer", "destroy", "shift", "modify", "beacon", "end")


class Layout:
    """A hex disc: slot indices for cells, neighbors, vertices, and the action layout."""

    def __init__(self, radius):
        self.radius = radius
        self.coords = hg.disc(radius)
        self.index = {c: i for i, c in enumerate(self.coords)}
        n = self.n = len(self.coords)
        self.nbr = np.array([[self.index.get(hg.neighbor(c, k), -1) for k in range(6)] for c in self.coords], np.int32)
        keys, vcells = {}, []
        self.corner_vertex = np.full((n, 6), -1, np.int32)
        for i, c in enumerate(self.coords):
            for k in range(6):
                key = hg.corner_key(c, k)
                if key not in keys:
                    keys[key] = len(vcells)
                    cells = sorted(self.index[x] for x in key if x in self.index)
                    vcells.append(cells + [-1] * (3 - len(cells)))
                self.corner_vertex[i, k] = keys[key]
        self.vcells = np.array(vcells, np.int32)
        self.v = len(vcells)
        # which corner of each of those cells the vertex is, so the network can tell corners apart
        self.vcorner = np.zeros((self.v, 3), np.int32)
        for v, cells in enumerate(self.vcells):
            for j, c in enumerate(cells):
                if c >= 0:
                    self.vcorner[v, j] = list(self.corner_vertex[c]).index(v)
        self.T0, self.D0 = 0, 18 * n
        self.S0 = self.D0 + 2 * n
        self.M0 = self.S0 + 5 * n
        self.B0 = self.M0 + 6 * n
        self.END = self.B0 + self.v
        self.n_actions = self.END + 1

    def kind(self, a):
        """Index into KINDS for an action index (works on arrays)."""
        a = jnp.asarray(a)
        return sum((a >= t).astype(jnp.int32) for t in (self.D0, self.S0, self.M0, self.B0, self.END))

    def board_arrays(self, board):
        """mask, shells, starts for a capsid.board.Board whose cells lie in this disc."""
        mask = np.zeros(self.n, bool)
        shells = np.zeros(self.n, np.int32)
        for c, s in zip(board.coords, board.shells):
            mask[self.index[c]] = True
            shells[self.index[c]] = s
        starts = np.full((2, 3), -1, np.int32)          # up to 3 start cells per player, -1 padded
        for p, group in enumerate(board.starts):
            for j, c in enumerate(group):
                starts[p, j] = self.index[board.coords[c]]
        return mask, shells, starts


class State(NamedTuple):
    mask: jax.Array          # [N] bool: which disc cells are on the board
    stores: jax.Array        # [N, 2]
    shells: jax.Array        # [N]
    beacons: jax.Array       # [V] owner, -1 empty
    tally: jax.Array         # [2]
    active: jax.Array        # [2] bool
    round: jax.Array
    player: jax.Array        # player to move
    groups: jax.Array        # [2] move groups used this turn, in order, -1 unused
    spent: jax.Array         # bool: units expended this turn
    moves: jax.Array         # moves this turn
    streak: jax.Array        # cost-free turns in a row
    turns: jax.Array         # turns this round
    arrived: jax.Array       # [N] units moved in this turn
    steps: jax.Array         # decisions so far this game
    n_turns: jax.Array       # turns so far this game
    capped_turns: jax.Array  # turns that hit max_actions_per_turn
    capped_rounds: jax.Array # rounds that hit max_turns_per_round
    done: jax.Array
    truncated: jax.Array     # ended by max_steps rather than by the rules


def dominance(stores):
    s0, s1 = stores[:, 0], stores[:, 1]
    return jnp.where(s0 > s1, 0, jnp.where(s1 > s0, 1, -1))


def max_affordable(units, p, m):
    """Largest n with n + ceil(n/m) * p <= units (all units if the crossing is free)."""
    m1 = jnp.maximum(m, 1)
    q = units // (m1 + p)
    r = units - q * (m1 + p)
    n = q * m1 + jnp.where(r > p, jnp.minimum(m1, r - p), 0)
    return jnp.where(p == 0, units, n)


def rot(s, k):
    k = k % 6
    return jnp.where(k == 0, s, ((s << k) | (s >> (6 - k))) & 63)


class Engine:
    def __init__(self, layout, cfg=Config(allow_pay=False), max_steps=2000):
        assert cfg.players == 2 and not cfg.allow_pay and cfg.groups_per_turn == 2
        self.L, self.cfg, self.max_steps = layout, cfg, max_steps
        self.nbr0 = jnp.asarray(layout.nbr)
        self.vcells0 = jnp.asarray(layout.vcells)
        self.corner_vertex = jnp.asarray(layout.corner_vertex)
        self.H, self.P, self.M = jnp.asarray(H_TAB), jnp.asarray(P_TAB), jnp.asarray(M_TAB)
        g = cfg.move_groups
        self.group_of = (g[TRANSFER], g[DESTROY], g[SHELL_OPS], g[BEACON])

    # ------------------------------------------------------------------ board
    def view(self, mask):
        """Neighbors and vertex cells with missing cells removed, and which vertices are beacon positions."""
        nbr = jnp.where((self.nbr0 >= 0) & mask[jnp.maximum(self.nbr0, 0)], self.nbr0, -1)
        vc = jnp.where((self.vcells0 >= 0) & mask[jnp.maximum(self.vcells0, 0)], self.vcells0, -1)
        return nbr, vc, (vc >= 0).sum(-1) >= 2

    def init(self, mask, shells, starts):
        n, z = self.L.n, jnp.int32(0)
        stores = jnp.zeros((n, 2), jnp.int32)
        for p in (0, 1):
            st = jnp.asarray(starts)[p]
            stores = stores.at[jnp.maximum(st, 0), p].add(jnp.where(st >= 0, self.cfg.start_units, 0))
        return State(jnp.asarray(mask), stores, jnp.asarray(shells, jnp.int32), jnp.full(self.L.v, -1, jnp.int32),
                     jnp.zeros(2, jnp.int32), jnp.ones(2, bool), z, z, jnp.full(2, -1, jnp.int32), jnp.bool_(False),
                     z, z, z, jnp.zeros(n, jnp.int32), z, z, z, z, jnp.bool_(False), jnp.bool_(False))

    # ------------------------------------------------------------------ legal moves
    def _groups_ok(self, s):
        used0, used1 = s.groups[0], s.groups[1]
        last = jnp.where(used1 >= 0, used1, used0)
        out = []
        for g in self.group_of:
            if self.cfg.ordered_groups:
                out.append((used0 < 0) | (g == last) | ((g != used0) & (used1 < 0)))
            else:
                out.append((used0 < 0) | (g == used0) | (g == used1) | (used1 < 0))
        return out

    def _transfer_amounts(self, s, nbr):
        """[N, 6] most units movable across each edge, and the crossing costs p, m."""
        me = s.player
        nb = jnp.maximum(nbr, 0)
        e = jnp.arange(6)[None, :]
        p = self.P[s.shells[:, None], s.shells[nb], e]
        m = self.M[s.shells[:, None], s.shells[nb], e]
        mine = s.stores[:, me]
        movable = mine if self.cfg.multi_transfer else mine - s.arrived
        nmax = jnp.minimum(max_affordable(mine[:, None], p, m), movable[:, None])
        nmax = jnp.where((nbr >= 0) & (m > 0), nmax, 0)
        return nmax, p, m

    def legal(self, s):
        cfg = self.cfg
        nbr, vc, vvalid = self.view(s.mask)
        me = s.player
        mine, theirs = s.stores[:, me], s.stores[:, 1 - me]
        dom = dominance(s.stores)
        go = (~s.done) & (s.moves < cfg.max_actions_per_turn)
        ok_t, ok_d, ok_s, ok_b = [x & go for x in self._groups_ok(s)]
        nmax, _, _ = self._transfer_amounts(s, nbr)
        t = jnp.stack([nmax >= 1, nmax // 2 > 1, nmax > 1], -1) & ok_t
        dmax = jnp.minimum(mine, theirs)
        d = jnp.stack([dmax >= 1, dmax > 1], -1) & ok_d
        can_shift = (dom == me) | ((mine > 0) & (mine >= cfg.shift_cost))
        sh = jnp.broadcast_to(can_shift[:, None], (self.L.n, 5)) & ok_s
        mo = jnp.broadcast_to((mine >= max(1, cfg.modify_cost))[:, None], (self.L.n, 6)) & ok_s
        cost = jnp.where(s.beacons < 0, cfg.beacon_build_cost, cfg.beacon_capture_cost)
        vcs = jnp.maximum(vc, 0)
        cell_ok = (vc < 0) | ((dom[vcs] == me) & (mine[vcs] >= cost[:, None]))
        b = vvalid & (s.beacons != me) & cell_ok.all(-1) & ok_b
        return jnp.concatenate([t.ravel(), d.ravel(), sh.ravel(), mo.ravel(), b, (~s.done)[None]])

    # ------------------------------------------------------------------ moves
    def step(self, s, a, check=True):
        """The state after action a. With check, an illegal action ends the turn instead; search
        and the bots only pick legal actions, so they skip the check.

        Every move kind's effect is computed directly as a few scatters weighted by whether this
        action is of that kind, rather than through lax.switch, which under vmap evaluates every
        branch in full (including the board view and the whole transfer table)."""
        L, cfg, N = self.L, self.cfg, self.L.n
        if check:
            a = jnp.where(self.legal(s)[a], a, L.END)
        kind = L.kind(a)
        is_t, is_d, is_s, is_m, is_b, is_e = [kind == j for j in range(6)]
        me, opp = s.player, 1 - s.player
        clip = lambda x, hi: jnp.clip(x, 0, hi - 1)
        i = clip(a - L.T0, 18 * N)
        ct, et, bt = i // 18, (i // 3) % 6, i % 3
        i = clip(a - L.D0, 2 * N)
        cd, bd = i // 2, i % 2
        i = clip(a - L.S0, 5 * N)
        cs, ks = i // 5, i % 5 + 1
        i = clip(a - L.M0, 6 * N)
        cm, node = i // 6, i % 6
        v = clip(a - L.B0, L.v)

        # transfer: only the one crossing that is used
        nb_raw = jnp.where(s.mask[jnp.maximum(self.nbr0[ct, et], 0)], self.nbr0[ct, et], -1)
        nb = jnp.maximum(nb_raw, 0)
        p, m = self.P[s.shells[ct], s.shells[nb], et], self.M[s.shells[ct], s.shells[nb], et]
        mine_t = s.stores[ct, me]
        movable = mine_t if cfg.multi_transfer else mine_t - s.arrived[ct]
        nmax = jnp.minimum(max_affordable(mine_t, p, m), movable)
        n_t = jnp.stack([jnp.int32(1), nmax // 2, nmax])[bt] * is_t
        m1 = jnp.maximum(m, 1)
        pay_t = jnp.where(p == 0, 0, (n_t + m1 - 1) // m1 * p) * is_t
        # destroy, shift, modify, beacon
        n_d = jnp.where(bd == 0, 1, jnp.minimum(s.stores[cd, me], s.stores[cd, opp])) * is_d
        dom_cs = jnp.where(s.stores[cs, 0] > s.stores[cs, 1], 0, jnp.where(s.stores[cs, 1] > s.stores[cs, 0], 1, -1))
        cost_s = jnp.where(dom_cs == me, 0, cfg.shift_cost) * is_s
        cost_m = cfg.modify_cost * is_m
        vcells = jnp.where((self.vcells0[v] >= 0) & s.mask[jnp.maximum(self.vcells0[v], 0)], self.vcells0[v], -1)
        cost_b = jnp.where(s.beacons[v] < 0, cfg.beacon_build_cost, cfg.beacon_capture_cost) * is_b

        st = s.stores.at[ct, me].add(-(n_t + pay_t)).at[nb, me].add(n_t)
        st = st.at[cd, me].add(-n_d).at[cd, opp].add(-n_d)
        st = st.at[cs, me].add(-cost_s).at[cm, me].add(-cost_m)
        st = st.at[jnp.maximum(vcells, 0), me].add(jnp.where(vcells >= 0, -cost_b, 0))
        shells = s.shells.at[cs].set(jnp.where(is_s, rot(s.shells[cs], ks), s.shells[cs]))
        shells = shells.at[cm].set(jnp.where(is_m, shells[cm] ^ (1 << node), shells[cm]))
        spent = (s.spent | (pay_t > 0) | (n_d > 0) | (cost_s > 0) | (cost_m > 0) | (cost_b > 0))
        t = jnp.where(is_t, 0, jnp.where(is_d, 1, jnp.where(is_s | is_m, 2, 3)))
        g = jnp.asarray(self.group_of, jnp.int32)[t]
        g0, g1 = s.groups[0], s.groups[1]
        groups = jnp.where(g0 < 0, jnp.stack([g, jnp.int32(-1)]),
                           jnp.where((g0 != g) & (g1 < 0), jnp.stack([g0, g]), s.groups))
        moved = s._replace(stores=st, shells=shells, beacons=s.beacons.at[v].set(jnp.where(is_b, me, s.beacons[v])),
                           arrived=s.arrived.at[nb].add(n_t), spent=spent, groups=groups.astype(jnp.int32),
                           moves=s.moves + 1)
        s2 = jax.tree.map(lambda x, y: jnp.where(is_e, x, y), self._end(s, a), moved)
        s2 = s2._replace(steps=s.steps + 1)
        trunc = (s2.steps >= self.max_steps) & ~s2.done
        s2 = s2._replace(done=s2.done | trunc, truncated=trunc)
        return jax.tree.map(lambda x, y: jnp.where(s.done, x, y), s, s2)

    def _end(self, s, a):
        cfg = self.cfg
        streak = jnp.where(s.spent, 0, s.streak + 1)
        turns = s.turns + 1
        n_active = s.active.sum()
        over = (streak >= n_active) | (turns >= cfg.max_turns_per_round)
        s = s._replace(streak=streak, turns=turns, n_turns=s.n_turns + 1,
                       capped_turns=s.capped_turns + (s.moves >= cfg.max_actions_per_turn),
                       capped_rounds=s.capped_rounds + (over & (streak < n_active)))
        opp = 1 - s.player
        s_next = s._replace(player=jnp.where(s.active[opp], opp, s.player))
        s = jax.tree.map(lambda h, t: jnp.where(over, h, t), self._harvest(s), s_next)
        return s._replace(groups=jnp.full(2, -1, jnp.int32), spent=jnp.bool_(False), moves=jnp.int32(0),
                          arrived=jnp.zeros_like(s.arrived))

    def _harvest(self, s):
        cfg = self.cfg
        tally = s.tally + jnp.stack([(s.beacons == 0).sum(), (s.beacons == 1).sum()]).astype(jnp.int32)
        nbr, _, _ = self.view(s.mask)
        occ = s.stores > 0
        adj = jnp.where((nbr >= 0)[:, :, None], occ[jnp.maximum(nbr, 0)], False).sum(1)
        income = (dominance(s.stores)[:, None] == jnp.arange(2)[None, :]) * (1 + adj)
        stores = s.stores + income.astype(jnp.int32)
        active = s.active & (stores > 0).any(0)
        rnd = s.round + 1
        done = (rnd >= cfg.rounds) | (active.sum() == 0) | (cfg.early_end & (active.sum() <= 1))
        return s._replace(stores=stores, tally=tally, active=active, round=rnd, done=done,
                          player=jnp.where(active[0], 0, 1).astype(jnp.int32), streak=jnp.int32(0), turns=jnp.int32(0))

    # ------------------------------------------------------------------ results and features
    def outcome(self, s):
        """+1 if player 0 wins, -1 if player 1 wins, 0 for a draw: tally, then dominant cells."""
        dom = dominance(s.stores)
        key0 = s.tally[0] * 1000 + (dom == 0).sum()
        key1 = s.tally[1] * 1000 + (dom == 1).sum()
        return jnp.sign(key0 - key1).astype(jnp.float32)

    def observe(self, s):
        """Features from the point of view of the player to move: per cell [N, F] and global [G]."""
        cfg = self.cfg
        me = s.player
        nbr, vc, vvalid = self.view(s.mask)
        mine, theirs = s.stores[:, me], s.stores[:, 1 - me]
        dom = dominance(s.stores)
        bits = (s.shells[:, None] >> jnp.arange(6)[None, :]) & 1
        nb = jnp.maximum(nbr, 0)
        h = jnp.where(nbr >= 0, self.H[s.shells[:, None], s.shells[nb], jnp.arange(6)[None, :]], 7)
        cv = self.corner_vertex
        own = s.beacons[cv]
        corners = jnp.stack([vvalid[cv], own == me, own == 1 - me], -1)
        f = lambda x: x.astype(jnp.float32)
        cells = jnp.concatenate([
            f(s.mask)[:, None], f(mine)[:, None] / 8, f(theirs)[:, None] / 8, f(mine > 0)[:, None], f(theirs > 0)[:, None],
            f(dom == me)[:, None], f(dom == 1 - me)[:, None], f(bits), f(s.arrived)[:, None] / 8,
            f(jax.nn.one_hot(h, 8)).reshape(self.L.n, 48), f(corners).reshape(self.L.n, 18)], -1)
        cells = cells * f(s.mask)[:, None]
        glob = jnp.concatenate([
            jax.nn.one_hot(s.round, cfg.rounds), f(jnp.stack([s.tally[me], s.tally[1 - me], s.tally[me] - s.tally[1 - me]])) / 10,
            jax.nn.one_hot(s.groups[0] + 1, 5), jax.nn.one_hot(s.groups[1] + 1, 5),
            f(s.moves)[None] / cfg.max_actions_per_turn, jax.nn.one_hot(jnp.minimum(s.streak, 2), 3), f(s.spent)[None],
            f(s.turns)[None] / cfg.max_turns_per_round, f(s.active[1 - me])[None], f(s.steps)[None] / self.max_steps])
        return cells, glob

    @property
    def n_cell_features(self):
        return 1 + 4 + 2 + 6 + 1 + 48 + 18

    @property
    def n_global_features(self):
        return self.cfg.rounds + 3 + 5 + 5 + 1 + 3 + 1 + 1 + 1 + 1
