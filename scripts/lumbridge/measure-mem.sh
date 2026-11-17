#!/usr/bin/env bash
# Runs an SFT config and reports how much unified memory it took from the box (DGX Spark).
# prime-rl's "Peak Mem" counts every process on unified memory, so we sample MemAvailable instead.
#   scripts/lumbridge/measure-mem.sh configs/lumbridge/bolt-e4b-smoke.toml [extra sft args]
set -u
cfg=$1; shift
avail() { awk '/MemAvailable/ {print int($2/1048576*10)/10}' /proc/meminfo; }
base=$(avail); low=$base
log=$(mktemp)
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
  nice -n 19 ~/.local/bin/uv run --no-sync sft @ "$cfg" --no-monitors.prime "$@" >"$log" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do a=$(avail); awk -v a=$a -v l=$low 'BEGIN{exit !(a<l)}' && low=$a; sleep 1; done
wait $pid; rc=$?
out=$(grep -o 'outputs/[^ ]*/logs/attempt_1/trainer.log' "$log" | head -1)
echo "config $cfg $* rc=$rc"
echo "available before ${base} GB, lowest ${low} GB, taken $(awk -v b=$base -v l=$low 'BEGIN{print b-l}') GB"
[ -n "$out" ] && grep -o 'Step [0-9] .*Grad. Norm [0-9.]*\|Throughput [0-9]* tokens/s' "$out" | paste - - | cut -c1-120
