#!/usr/bin/env bash
# Evaluate every Nth checkpoint of a run against greedy and random as it appears.
#   setsid nohup scripts/watch_eval.sh RUN [EVERY_ITERS] > out/runs/RUN.eval.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
run=$1; every=${2:-100}
dir=out/runs/$run
while true; do
  for f in "$dir"/params_*.msgpack; do
    [[ -e "$f" ]] || continue
    name=$(basename "$f" .msgpack); it=$((10#${name#params_}))
    (( it % every == 0 )) || continue
    grep -q "\"ckpt\": \"$name\"" "$dir/eval.jsonl" 2>/dev/null && continue
    uv run python -m capsid.evaluate --run "$run" --ckpt "$name" --opponent greedy --games 64 2>&1 | grep '^{'
    uv run python -m capsid.evaluate --run "$run" --ckpt "$name" --opponent random --games 64 2>&1 | grep '^{'
  done
  [[ -f "$dir/DONE" ]] && exit 0
  sleep 60
done
