"""Play a trained checkpoint against baselines and report scorecard numbers.

    uv run python -m capsid.evaluate --run v2-tri12 --opponent greedy --games 128
    uv run python -m capsid.evaluate --run v2-tri12 --ckpt params_000100 --opponent params_000050

The opponent is `random`, `greedy`, or another checkpoint name from the same run. Half the games
put the checkpoint in each seat. Results are appended to out/runs/<run>/eval.jsonl.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=")

import argparse
import dataclasses
import json
import time
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp

from .bots import greedy_action, random_action
from .train import TrainConfig, Trainer

RUNS = Path(__file__).resolve().parents[1] / "out" / "runs"


def load_trainer(run):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    return Trainer(TrainConfig(**{k: v for k, v in cfg.items() if k in fields}))


def load_params(tr, run, name):
    path = RUNS / run / (name if name.endswith(".msgpack") else name + ".msgpack")
    s0 = jax.tree.map(lambda x: x[0], jax.vmap(tr.reset)(jax.random.split(jax.random.PRNGKey(0), 1)))
    c, g = tr.eng.observe(s0)
    nbr, vc, _ = tr.eng.view(s0.mask)
    template = tr.net.init(jax.random.PRNGKey(0), c, g, nbr, vc, tr.vcorner, s0.mask)
    data = path.read_bytes()
    if name.startswith("latest"):
        return flax.serialization.msgpack_restore(data)["params"]
    return flax.serialization.from_bytes(template, data)


def make_policy(tr, kind, params=None, sims=16):
    """A batched policy (states, key) -> actions."""
    if kind == "random":
        return lambda s, k: jax.vmap(lambda st, kk: random_action(tr.eng, st, kk))(s, jax.random.split(k, s.player.shape[0]))
    if kind == "greedy":
        return lambda s, k: jax.vmap(lambda st, kk: greedy_action(tr.eng, st, kk))(s, jax.random.split(k, s.player.shape[0]))
    return lambda s, k: tr.search(params, k, s, sims)[0].action


def match(tr, pol_a, pol_b, games, seed):
    """pol_a against pol_b; pol_a plays seat 0 in even games and seat 1 in odd ones."""
    key = jax.random.PRNGKey(seed)
    k0, key = jax.random.split(key)
    states = jax.vmap(tr.reset)(jax.random.split(k0, games))
    seat = jnp.arange(games) % 2

    def body(c):
        s, key, t = c
        key, ka, kb = jax.random.split(key, 3)
        a = jnp.where(s.player == seat, pol_a(s, ka), pol_b(s, kb))
        return tr.vstep(s, a), key, t + 1

    run = jax.jit(lambda s, k: jax.lax.while_loop(lambda c: ~c[0].done.all() & (c[2] <= tr.tc.max_steps), body, (s, k, 0)))
    s, _, _ = run(states, key)
    z = tr.voutcome(s)
    za = jnp.where(seat == 0, z, -z)
    ta = jnp.where(seat == 0, s.tally[:, 0], s.tally[:, 1])
    tb = jnp.where(seat == 0, s.tally[:, 1], s.tally[:, 0])
    f = lambda x: round(float(x), 3)
    return {"win": f((za > 0).mean()), "draw": f((za == 0).mean()), "loss": f((za < 0).mean()),
            "win_as_p0": f(((za > 0) & (seat == 0)).sum() / (seat == 0).sum()),
            "win_as_p1": f(((za > 0) & (seat == 1)).sum() / (seat == 1).sum()),
            "tally": f(ta.mean()), "opp_tally": f(tb.mean()), "beacons": f((s.beacons >= 0).sum(-1).mean()),
            "rounds": f(s.round.mean()), "decisions": f(s.steps.mean()), "turns": f(s.n_turns.mean()),
            "decisions_per_turn": f(s.steps.sum() / jnp.maximum(s.n_turns.sum(), 1)),
            "turns_per_round": f(s.n_turns.sum() / jnp.maximum(s.round.sum(), 1)),
            "capped_turn_share": f(s.capped_turns.sum() / jnp.maximum(s.n_turns.sum(), 1)),
            "truncated": f(s.truncated.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="latest", help="checkpoint name in the run folder (default: latest)")
    ap.add_argument("--opponent", default="greedy", help="random, greedy, or a checkpoint name")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    tr = load_trainer(args.run)
    params = load_params(tr, args.run, args.ckpt)
    pol_a = make_policy(tr, "net", params, args.sims)
    if args.opponent in ("random", "greedy"):
        pol_b = make_policy(tr, args.opponent)
    else:
        pol_b = make_policy(tr, "net", load_params(tr, args.run, args.opponent), args.sims)
    t = time.time()
    res = match(tr, pol_a, pol_b, args.games, args.seed)
    rec = {"ckpt": args.ckpt, "opponent": args.opponent, "games": args.games, "sims": args.sims,
           "seconds": round(time.time() - t, 1), **res}
    with open(RUNS / args.run / "eval.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
