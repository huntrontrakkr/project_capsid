"""AlphaZero-style self-play training with Gumbel search (mctx) on generated boards.

    uv run python -m capsid.train --run r3-baseline --radius 3 --hours 2

Every iteration plays `chunk` moves in each of `envs` parallel games, writes every position to
a replay buffer, fills in the result once each game ends, and takes `updates` gradient steps on
finished positions. Every `eval_every` iterations the agent plays a match against a uniformly
random player (both seats) and against its previous checkpoint. Everything is written to
out/runs/<run>/: config.json, metrics.jsonl, and params checkpoints.

Logged per iteration, for the design scorecard: decisions per turn and per game, turns per
round, how often the safety caps bind, the effective number of options at the root (prior and
search policy), legal moves per decision, which kinds of moves are played, and results by seat.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# CUDA graphs (XLA command buffers) segfault inside libcuda on this WSL2 driver (610.88) when a
# compiled loop is rerun with different weights; autotuning only slows compiles down.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=")

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import NamedTuple

import flax.serialization
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax

from .board import GenParams, generate
from .bots import greedy_action
from .jax_engine import KINDS, Engine, Layout, State
from .net import Net
from .rules import Config

NEG = -1e9


@dataclasses.dataclass
class TrainConfig:
    run: str = "run"
    radius: int = 4
    boards: int = 2000             # pool of generated boards; each game draws one
    envs: int = 256                # games played in parallel
    sims: int = 16                 # search simulations per decision
    considered: int = 16           # actions considered at the root (Gumbel)
    chunk: int = 32                # self-play moves per game per iteration
    buffer_per_env: int = 1024
    topk: int = 16                 # policy targets kept per position
    batch: int = 1024
    updates: int = 8               # gradient steps per iteration
    lr: float = 3e-4
    hidden: int = 128
    layers: int = 6
    max_steps: int = 1000          # decisions per game before it is cut off
    max_actions_per_turn: int = 8
    max_turns_per_round: int = 16
    start_units: int = 5           # per start cell
    start_cells: int = 3
    rounds: int = 12
    beacon_build_cost: int = 2
    early_end: int = 1             # 1: the game ends when one player is left with units
    eval_every: int = 25
    eval_games: int = 128
    eval_sims: int = 16
    iterations: int = 1_000_000
    hours: float = 1.0
    seed: int = 0
    resume: int = 0                # 1: continue from out/runs/<run>/latest.msgpack if it exists
    init_from: str = ""            # start from another run's weights: "<run>" (its latest) or "<run>/<checkpoint>"
    greedy_mix: float = 0.0        # share of self-play games with the greedy bot in one seat
    bf16: int = 0                  # 1: run the network in bfloat16 (parameters stay float32)


class Buffer(NamedTuple):
    states: State       # leaves [E, L, ...]
    pidx: jax.Array     # [E, L, K] policy target actions
    pw: jax.Array       # [E, L, K] policy target weights
    value: jax.Array    # [E, L] result for the player to move
    ready: jax.Array    # [E, L] result filled in
    ep: jax.Array       # [E, L] episode number of the position
    ptr: jax.Array      # [E]
    cur_ep: jax.Array   # [E]


STAT_KEYS = ("decisions", "root_ppl", "prior_ppl", "legal", "games", "game_steps", "game_turns", "capped_turns",
             "capped_rounds", "truncated", "p0_win", "draw", "p0_loss", "tally_sum", "rounds", "vg_games", "vg_win",
             "vg_draw") + tuple(f"n_{k}" for k in KINDS)


class Trainer:
    def __init__(self, tc: TrainConfig):
        self.tc = tc
        self.layout = L = Layout(tc.radius)
        self.cfg = Config(allow_pay=False, max_actions_per_turn=tc.max_actions_per_turn,
                          max_turns_per_round=tc.max_turns_per_round, start_units=tc.start_units, rounds=tc.rounds,
                          beacon_build_cost=tc.beacon_build_cost, early_end=bool(tc.early_end))
        self.eng = Engine(L, self.cfg, tc.max_steps)
        self.net = Net(tc.hidden, tc.layers, jnp.bfloat16 if tc.bf16 else jnp.float32)
        self.vcorner = jnp.asarray(L.vcorner)
        gp = GenParams(radius=tc.radius, start_cells=tc.start_cells)
        boards = [L.board_arrays(generate(s, gp)) for s in range(tc.boards)]
        self.masks = jnp.asarray(np.stack([b[0] for b in boards]))
        self.shells = jnp.asarray(np.stack([b[1] for b in boards]))
        self.starts = jnp.asarray(np.stack([b[2] for b in boards]))
        self.vstep = jax.vmap(self.eng.step)
        self.vstep_fast = jax.vmap(lambda s, a: self.eng.step(s, a, check=False))   # for actions known to be legal
        self.vlegal = jax.vmap(self.eng.legal)
        self.voutcome = jax.vmap(self.eng.outcome)
        self.opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(tc.lr, weight_decay=1e-4))
        self.selfplay = jax.jit(self._selfplay, donate_argnums=(1, 2))
        self.train_steps = jax.jit(self._train_steps)
        self.match = jax.jit(self._match, static_argnames=("vs_random",))

    # ------------------------------------------------------------------ helpers
    def reset(self, key):
        i = jax.random.randint(key, (), 0, self.masks.shape[0])
        return self.eng.init(self.masks[i], self.shells[i], self.starts[i])

    def evaluate(self, params, states):
        def one(s):
            cells, glob = self.eng.observe(s)
            nbr, vc, _ = self.eng.view(s.mask)
            return self.net.apply(params, cells, glob, nbr, vc, self.vcorner, s.mask)
        return jax.vmap(one)(states)

    def recurrent_fn(self, params, key, action, states):
        new = self.vstep_fast(states, action)
        logits, value = self.evaluate(params, new)
        logits = jnp.where(self.vlegal(new), logits, NEG)
        z0 = self.voutcome(new)
        ended = new.done & ~states.done
        reward = jnp.where(ended, jnp.where(states.player == 0, z0, -z0), 0.0)
        discount = jnp.where(new.done, 0.0, jnp.where(new.player == states.player, 1.0, -1.0))
        return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits,
                                      value=jnp.where(new.done, 0.0, value)), new

    def search(self, params, key, states, sims):
        legal = self.vlegal(states)
        logits, value = self.evaluate(params, states)
        logits = jnp.where(legal, logits, NEG)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states)
        out = mctx.gumbel_muzero_policy(params, key, root, self.recurrent_fn, num_simulations=sims,
                                        invalid_actions=~legal, max_num_considered_actions=self.tc.considered)
        return out, logits, legal

    def init_buffer(self, states):
        E, Lb, K = self.tc.envs, self.tc.buffer_per_env, self.tc.topk
        st = jax.tree.map(lambda x: jnp.zeros((E, Lb) + x.shape[1:], x.dtype), states)
        return Buffer(st, jnp.zeros((E, Lb, K), jnp.int32), jnp.zeros((E, Lb, K)), jnp.zeros((E, Lb)),
                      jnp.zeros((E, Lb), bool), jnp.full((E, Lb), -1, jnp.int32), jnp.zeros(E, jnp.int32),
                      jnp.zeros(E, jnp.int32))

    # ------------------------------------------------------------------ self-play
    def _selfplay(self, params, states, buf, key):
        tc, E = self.tc, self.tc.envs
        e = jnp.arange(E)

        n_greedy = int(round(tc.greedy_mix * E))
        greedy_seat = e % 2                            # in the first n_greedy games, greedy plays this seat

        def body(carry, key):
            states, buf, stats = carry
            k_search, k_reset, k_greedy = jax.random.split(key, 3)
            out, logits, legal = self.search(params, k_search, states, tc.sims)
            action = out.action
            by_greedy = jnp.zeros(E, bool)
            if n_greedy:
                g_states = jax.tree.map(lambda x: x[:n_greedy], states)
                ga = jax.vmap(lambda s, k: greedy_action(self.eng, s, k))(g_states, jax.random.split(k_greedy, n_greedy))
                by_greedy = (e < n_greedy) & (states.player == greedy_seat)
                action = jnp.where(by_greedy, jnp.zeros(E, jnp.int32).at[:n_greedy].set(ga), action)
            w, idx = jax.lax.top_k(out.action_weights, tc.topk)
            w = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-9)
            buf = buf._replace(
                states=jax.tree.map(lambda b, x: b.at[e, buf.ptr].set(x), buf.states, states),
                pidx=buf.pidx.at[e, buf.ptr].set(idx), pw=buf.pw.at[e, buf.ptr].set(w),
                ready=buf.ready.at[e, buf.ptr].set(False),
                ep=buf.ep.at[e, buf.ptr].set(jnp.where(by_greedy, -2, buf.cur_ep)),   # greedy's moves never train
                ptr=(buf.ptr + 1) % tc.buffer_per_env)
            new = self.vstep_fast(states, action)
            ended = new.done
            z0 = self.voutcome(new)
            sel = (buf.ep == buf.cur_ep[:, None]) & ended[:, None] & ~buf.ready
            sign = 1.0 - 2.0 * buf.states.player
            buf = buf._replace(value=jnp.where(sel, z0[:, None] * sign, buf.value), ready=buf.ready | sel,
                               cur_ep=buf.cur_ep + ended)

            def ppl(p):
                p = jnp.where(legal, p, 0.0)
                return jnp.exp(-(p * jnp.log(jnp.maximum(p, 1e-12))).sum(-1))
            kinds = self.layout.kind(action)
            fe = ended.astype(jnp.float32)
            upd = {"decisions": E, "root_ppl": ppl(out.action_weights).sum(), "prior_ppl": ppl(jax.nn.softmax(logits)).sum(),
                   "legal": legal.sum(), "games": fe.sum(), "game_steps": (fe * new.steps).sum(),
                   "game_turns": (fe * new.n_turns).sum(), "capped_turns": (fe * new.capped_turns).sum(),
                   "capped_rounds": (fe * new.capped_rounds).sum(), "truncated": (fe * new.truncated).sum(),
                   "p0_win": (fe * (z0 > 0)).sum(), "draw": (fe * (z0 == 0)).sum(), "p0_loss": (fe * (z0 < 0)).sum(),
                   "tally_sum": (fe * new.tally.sum(-1)).sum(), "rounds": (fe * new.round).sum()}
            vg = fe * (e < n_greedy)                                  # finished games against greedy
            z_learner = jnp.where(greedy_seat == 0, -z0, z0)
            upd.update(vg_games=vg.sum(), vg_win=(vg * (z_learner > 0)).sum(), vg_draw=(vg * (z_learner == 0)).sum())
            for j, k in enumerate(KINDS):
                upd[f"n_{k}"] = (kinds == j).sum()
            stats = {k: stats[k] + upd[k] for k in STAT_KEYS}
            fresh = jax.vmap(self.reset)(jax.random.split(k_reset, E))
            states = jax.tree.map(lambda f, n: jnp.where(ended.reshape((-1,) + (1,) * (n.ndim - 1)), f, n), fresh, new)
            return (states, buf, stats), None

        stats = {k: jnp.float32(0) for k in STAT_KEYS}
        (states, buf, stats), _ = jax.lax.scan(body, (states, buf, stats), jax.random.split(key, tc.chunk))
        return states, buf, stats

    # ------------------------------------------------------------------ learning
    def _train_steps(self, params, opt_state, buf, key):
        tc = self.tc
        E, Lb = tc.envs, tc.buffer_per_env
        p = buf.ready.ravel().astype(jnp.float32)
        p = p / jnp.maximum(p.sum(), 1.0)

        def loss_fn(params, states, pidx, pw, z):
            logits, value = self.evaluate(params, states)
            logits = jnp.where(self.vlegal(states), logits, NEG)
            logp = jax.nn.log_softmax(logits)
            pl = -(pw * jnp.take_along_axis(logp, pidx, 1)).sum(1).mean()
            vl = ((value - z) ** 2).mean()
            return pl + vl, (pl, vl)

        def one(carry, key):
            params, opt_state = carry
            idx = jax.random.choice(key, E * Lb, (tc.batch,), p=p)
            e, l = idx // Lb, idx % Lb
            states = jax.tree.map(lambda x: x[e, l], buf.states)
            (_, (pl, vl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, states, buf.pidx[e, l],
                                                                               buf.pw[e, l], buf.value[e, l])
            updates, opt_state = self.opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), (pl, vl)

        (params, opt_state), (pl, vl) = jax.lax.scan(one, (params, opt_state), jax.random.split(key, tc.updates))
        return params, opt_state, pl.mean(), vl.mean()

    # ------------------------------------------------------------------ evaluation
    def _match(self, params, opp_params, key, vs_random):
        """The agent against a random player or another set of params, half the games in each seat."""
        tc = self.tc
        n = tc.eval_games
        k0, key = jax.random.split(key)
        states = jax.vmap(self.reset)(jax.random.split(k0, n))
        seat = jnp.arange(n) % 2                       # the agent plays this seat

        def body(c):
            states, key, t = c
            key, k1, k2 = jax.random.split(key, 3)
            out, _, legal = self.search(params, k1, states, tc.eval_sims)
            if vs_random:
                opp = jax.random.categorical(k2, jnp.where(legal, 0.0, NEG))
            else:
                opp = self.search(opp_params, k2, states, tc.eval_sims)[0].action
            a = jnp.where(states.player == seat, out.action, opp)
            return self.vstep(states, a), key, t + 1

        states, _, _ = jax.lax.while_loop(lambda c: ~c[0].done.all() & (c[2] <= tc.max_steps), body,
                                          (states, key, jnp.int32(0)))
        z = self.voutcome(states)
        za = jnp.where(seat == 0, z, -z)
        return {"win": (za > 0).mean(), "draw": (za == 0).mean(), "loss": (za < 0).mean(),
                "win_as_p0": ((za > 0) & (seat == 0)).sum() / (seat == 0).sum(),
                "win_as_p1": ((za > 0) & (seat == 1)).sum() / (seat == 1).sum(),
                "steps": states.steps.mean(), "truncated": states.truncated.mean()}


def summarize(stats, tc):
    s = {k: float(v) for k, v in stats.items()}
    d, g = max(s["decisions"], 1), max(s["games"], 1)
    out = {"root_ppl": s["root_ppl"] / d, "prior_ppl": s["prior_ppl"] / d, "legal": s["legal"] / d, "games": s["games"]}
    if s["games"]:
        out.update(game_steps=s["game_steps"] / g, game_turns=s["game_turns"] / g,
                   decisions_per_turn=s["game_steps"] / max(s["game_turns"], 1),
                   turns_per_round=s["game_turns"] / max(s["rounds"], 1),
                   capped_turn_share=s["capped_turns"] / max(s["game_turns"], 1),
                   capped_round_share=s["capped_rounds"] / max(s["rounds"], 1),
                   truncated=s["truncated"] / g, p0_win=s["p0_win"] / g, draw=s["draw"] / g, p0_loss=s["p0_loss"] / g,
                   tally=s["tally_sum"] / g)
    if s["vg_games"]:
        out["vs_greedy_selfplay"] = {"games": s["vg_games"], "win": s["vg_win"] / s["vg_games"],
                                     "draw": s["vg_draw"] / s["vg_games"]}
    out["move_mix"] = {k: s[f"n_{k}"] / d for k in KINDS}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(TrainConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    tc = TrainConfig(**vars(ap.parse_args()))
    out = Path(__file__).resolve().parents[1] / "out" / "runs" / tc.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(tc), indent=2))
    log = open(out / "metrics.jsonl", "a")

    tr = Trainer(tc)
    key = jax.random.PRNGKey(tc.seed)
    key, k1, k2 = jax.random.split(key, 3)
    states = jax.vmap(tr.reset)(jax.random.split(k1, tc.envs))
    s0 = jax.tree.map(lambda x: x[0], states)
    c, g = tr.eng.observe(s0)
    nbr, vc, _ = tr.eng.view(s0.mask)
    params = tr.net.init(k2, c, g, nbr, vc, tr.vcorner, s0.mask)
    opt_state = tr.opt.init(params)
    buf = tr.init_buffer(states)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"run {tc.run}: radius {tc.radius} ({tr.layout.n} slots, {tr.layout.n_actions} actions), "
          f"{n_params:,} parameters, {tc.envs} games x {tc.sims} simulations", flush=True)

    if tc.init_from:
        src, _, name = tc.init_from.partition("/")
        data = (out.parent / src / ((name or "latest") + ".msgpack")).read_bytes()
        params = (flax.serialization.msgpack_restore(data)["params"] if not name
                  else flax.serialization.from_bytes(params, data))
        params = jax.tree.map(jnp.asarray, params)
        opt_state = tr.opt.init(params)
        print(f"initialized from {tc.init_from}", flush=True)
    prev_params, start, elapsed0 = params, 1, 0.0
    latest = out / "latest.msgpack"
    if tc.resume and latest.exists():
        template = {"params": params, "opt_state": opt_state, "prev_params": params, "key": key,
                    "iter": np.int64(0), "elapsed": np.float64(0)}
        ck = flax.serialization.from_bytes(template, latest.read_bytes())
        params, opt_state, prev_params, key = ck["params"], ck["opt_state"], ck["prev_params"], ck["key"]
        start, elapsed0 = int(ck["iter"]) + 1, float(ck["elapsed"])
        print(f"resumed at iteration {start} after {elapsed0 / 3600:.2f} h (replay buffer starts empty)", flush=True)
    t0 = time.time() - elapsed0
    for it in range(start, tc.iterations + 1):
        key, k_sp, k_tr, k_ev = jax.random.split(key, 4)
        ts = time.time()
        states, buf, stats = tr.selfplay(params, states, buf, k_sp)
        ready = int(buf.ready.sum())
        pl = vl = float("nan")
        if ready >= tc.batch:
            params, opt_state, pl, vl = tr.train_steps(params, opt_state, buf, k_tr)
            pl, vl = float(pl), float(vl)
        rec = {"iter": it, "time": round(time.time() - t0, 1), "iter_s": round(time.time() - ts, 2),
               "decisions_per_s": round(tc.envs * tc.chunk / (time.time() - ts)), "ready": ready,
               "policy_loss": pl, "value_loss": vl, **summarize(stats, tc)}
        if it % tc.eval_every == 0:
            rec["vs_random"] = {k: float(v) for k, v in tr.match(params, params, k_ev, vs_random=True).items()}
            rec["vs_prev"] = {k: float(v) for k, v in tr.match(params, prev_params, k_ev, vs_random=False).items()}
            prev_params = params
            (out / f"params_{it:06d}.msgpack").write_bytes(flax.serialization.to_bytes(params))
            ck = {"params": params, "opt_state": opt_state, "prev_params": prev_params, "key": key,
                  "iter": np.int64(it), "elapsed": np.float64(time.time() - t0)}
            tmp = out / "latest.tmp"
            tmp.write_bytes(flax.serialization.to_bytes(ck))
            tmp.replace(latest)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if it == 1 or it % 5 == 0 or "vs_random" in rec:
            msg = (f"it {it} {rec['time']:.0f}s {rec['decisions_per_s']}/s loss {pl:.3f}/{vl:.3f} "
                   f"ppl {rec['root_ppl']:.1f}/{rec['prior_ppl']:.1f} legal {rec['legal']:.0f}")
            if "game_steps" in rec:
                msg += (f" | game {rec['game_steps']:.0f} dec, {rec['decisions_per_turn']:.1f}/turn, "
                        f"{rec['turns_per_round']:.1f} turns/round, p0 {rec['p0_win']:.2f}")
            if "vs_greedy_selfplay" in rec:
                msg += f" | vs greedy {rec['vs_greedy_selfplay']['win']:.2f} ({rec['vs_greedy_selfplay']['games']:.0f})"
            if "vs_random" in rec:
                msg += f" | vs random {rec['vs_random']['win']:.2f}, vs prev {rec['vs_prev']['win']:.2f}"
            print(msg, flush=True)
        if time.time() - t0 > tc.hours * 3600:
            break
    (out / "params_final.msgpack").write_bytes(flax.serialization.to_bytes(params))
    (out / "DONE").write_text(f"{it}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
