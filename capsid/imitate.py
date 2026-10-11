"""Warm start: train the network to play like the greedy bot, then hand it to self-play.

    uv run python -m capsid.imitate --run bc-greedy --batches 12 --steps 20000
    uv run python -m capsid.train --run v4 --init-from bc-greedy ...

Self-play from random weights settled into a passive habit (stay on the start cells, end the
turn) that a 16-simulation search cannot see past, so it lost every game to greedy. Copying
greedy first gives the network a policy that expands and builds beacons and a value function
that has seen such games; self-play then improves on it. (AlphaGo started the same way, from
human games.)

Games are greedy against greedy. With probability `explore` a player makes a random legal move
instead, so the data also covers positions greedy would not reach; the training target is always
greedy's choice in that position. Value targets are the final results.
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
import numpy as np
import optax

from .bots import greedy_action, random_action
from .train import NEG, TrainConfig, Trainer

RUNS = Path(__file__).resolve().parents[1] / "out" / "runs"


def generate(tr, key, envs, steps, explore):
    """One batch of greedy-vs-greedy games: states, greedy's actions, results, and which rows count."""
    eng = tr.eng

    def body(s, key):
        kg, kr, ke = jax.random.split(key, 3)
        g = jax.vmap(lambda st, k: greedy_action(eng, st, k))(s, jax.random.split(kg, envs))
        r = jax.vmap(lambda st, k: random_action(eng, st, k))(s, jax.random.split(kr, envs))
        a = jnp.where(jax.random.uniform(ke, (envs,)) < explore, r, g)
        return tr.vstep_fast(s, a), (s, g, ~s.done)

    k0, key = jax.random.split(key)
    s0 = jax.vmap(tr.reset)(jax.random.split(k0, envs))
    final, (states, actions, live) = jax.lax.scan(body, s0, jax.random.split(key, steps))
    z0 = tr.voutcome(final)
    value = z0[None, :] * (1.0 - 2.0 * states.player)
    keep = live & final.done[None, :]                          # positions from games that finished
    return states, actions, value, keep, final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="bc-greedy")
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--max-game", type=int, default=700)
    ap.add_argument("--explore", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tc = TrainConfig(run=args.run, bf16=1, max_steps=args.max_game)
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(tc), indent=2))
    tr = Trainer(tc)
    key = jax.random.PRNGKey(args.seed)

    # ---- data
    gen = jax.jit(lambda k: generate(tr, k, args.envs, args.max_game, args.explore))
    parts, t0 = [], time.time()
    for b in range(args.batches):
        key, k = jax.random.split(key)
        states, actions, value, keep, final = gen(k)
        keep = np.asarray(keep)
        idx = np.nonzero(keep.ravel())[0]
        flat = lambda x: np.asarray(x).reshape((-1,) + x.shape[2:])[idx]
        parts.append((jax.tree.map(flat, states), flat(actions), flat(value)))
        print(f"batch {b + 1}/{args.batches}: {len(idx):,} positions from {int(np.asarray(final.done).sum())} finished games, "
              f"{float(final.steps.mean()):.0f} decisions per game, tally {np.asarray(final.tally).mean(0).round(1)}, "
              f"{time.time() - t0:.0f}s", flush=True)
    states = jax.tree.map(lambda *xs: np.concatenate(xs), *[p[0] for p in parts])
    actions = np.concatenate([p[1] for p in parts])
    value = np.concatenate([p[2] for p in parts])
    n = len(actions)
    print(f"{n:,} positions", flush=True)

    # ---- training
    s0 = jax.tree.map(lambda x: jnp.asarray(x[0]), states)
    c, g = tr.eng.observe(s0)
    nbr, vc, _ = tr.eng.view(s0.mask)
    key, k = jax.random.split(key)
    params = tr.net.init(k, c, g, nbr, vc, tr.vcorner, s0.mask)
    sched = optax.cosine_decay_schedule(args.lr, args.steps, 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    opt_state = opt.init(params)

    def loss_fn(params, st, a, z):
        logits, v = tr.evaluate(params, st)
        logits = jnp.where(tr.vlegal(st), logits, NEG)
        logp = jax.nn.log_softmax(logits)
        pl = -jnp.take_along_axis(logp, a[:, None], 1).mean()
        acc = (logits.argmax(-1) == a).mean()
        return pl + ((v - z) ** 2).mean(), (pl, ((v - z) ** 2).mean(), acc)

    @jax.jit
    def train_step(params, opt_state, st, a, z):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, st, a, z)
        upd, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, aux

    rng = np.random.default_rng(args.seed)
    held = rng.choice(n, min(8192, n // 10), replace=False)
    train_idx = np.setdiff1d(np.arange(n), held)
    eval_fn = jax.jit(lambda p, st, a, z: loss_fn(p, st, a, z)[1])
    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = rng.choice(train_idx, args.batch)
        params, opt_state, (pl, vl, acc) = train_step(params, opt_state, jax.tree.map(lambda x: jnp.asarray(x[i]), states),
                                                      jnp.asarray(actions[i]), jnp.asarray(value[i]))
        if step % 1000 == 0 or step == args.steps:
            j = held[:2048]
            hpl, hvl, hacc = eval_fn(params, jax.tree.map(lambda x: jnp.asarray(x[j]), states), jnp.asarray(actions[j]),
                                     jnp.asarray(value[j]))
            print(f"step {step}: train policy {float(pl):.3f} value {float(vl):.3f} agree {float(acc):.3f} | "
                  f"held-out policy {float(hpl):.3f} value {float(hvl):.3f} agree {float(hacc):.3f} | {time.time() - t0:.0f}s",
                  flush=True)
            (out / "latest.msgpack").write_bytes(flax.serialization.to_bytes({"params": params}))
    (out / "params_final.msgpack").write_bytes(flax.serialization.to_bytes(params))
    print("done", flush=True)


if __name__ == "__main__":
    main()
