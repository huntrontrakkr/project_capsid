"""Baseline players for the JAX engine: uniformly random, and greedy one-step lookahead.

The greedy player tries up to `candidates` of its legal moves, scores the position after each,
and plays the best one if it beats the current position; otherwise it ends its turn. The score,
for the player to move minus the same for the opponent:

    beacons owned x (rounds left + 1)    a beacon scores every remaining round
    ready corners x 0.75                 beacon positions where every cell around is ours
    dominant cells x 0.5
    next harvest's income x 0.25
    units x 0.1

It never plans past one move, so it pays whatever a crossing costs rather than rotating shells
first, which is the kind of simple strategy the design scorecard measures depth against.
"""
import jax
import jax.numpy as jnp

from .jax_engine import dominance

NEG = -1e9


def random_action(eng, s, key):
    return jax.random.categorical(key, jnp.where(eng.legal(s), 0.0, NEG))


def position_score(eng, s, p):
    """How good the position is for player p (minus the same for the opponent)."""
    nbr, vc, vvalid = eng.view(s.mask)
    dom = dominance(s.stores)
    occ = s.stores > 0
    adj = jnp.where((nbr >= 0)[:, :, None], occ[jnp.maximum(nbr, 0)], False).sum(1)      # [N, 2]
    income = jnp.where((dom[:, None] == jnp.arange(2)[None, :]), 1 + adj, 0).sum(0)      # [2]
    ready = jnp.stack([(vvalid & (s.beacons != q) & jnp.where(vc >= 0, dom[jnp.maximum(vc, 0)] == q, True).all(-1)).sum()
                       for q in (0, 1)])
    beacons = jnp.stack([(s.beacons == q).sum() for q in (0, 1)])
    dom_cells = jnp.stack([(dom == q).sum() for q in (0, 1)])
    units = s.stores.sum(0)
    left = jnp.maximum(eng.cfg.rounds - s.round, 0) + 1
    v = beacons * left + 0.75 * ready + 0.5 * dom_cells + 0.25 * income + 0.1 * units
    v = v.astype(jnp.float32)
    return v[p] - v[1 - p]


def greedy_action(eng, s, key, candidates=96):
    L = eng.L
    legal = eng.legal(s)
    pick = jnp.where(legal, jax.random.uniform(key, legal.shape), -1.0).at[L.END].set(-1.0)
    _, cand = jax.lax.top_k(pick, candidates)
    ok = legal[cand] & (cand != L.END)
    after = jax.vmap(lambda a: position_score(eng, eng.step(s, a), s.player))(cand)
    after = jnp.where(ok, after, NEG)
    best = jnp.argmax(after)
    better = after[best] > position_score(eng, s, s.player) + 1e-4
    return jnp.where(better, cand[best], L.END)
