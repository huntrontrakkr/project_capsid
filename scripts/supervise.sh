#!/usr/bin/env bash
# Run capsid.train and restart it from its last checkpoint if it crashes (the WSL CUDA driver
# occasionally segfaults). Stops once the run writes out/runs/<run>/DONE.
#
#   setsid nohup scripts/supervise.sh --run NAME [train options] > out/runs/NAME.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
run=""
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--run" ]]; then j=$((i + 1)); run="${!j}"; fi
done
[[ -n "$run" ]] || { echo "supervise.sh needs --run NAME"; exit 1; }
for attempt in $(seq 1 50); do
  echo "=== attempt $attempt $(date -Is)"
  uv run python -m capsid.train "$@" --resume 1
  echo "=== exit $? $(date -Is)"
  [[ -f "out/runs/$run/DONE" ]] && exit 0
  sleep 10
done
