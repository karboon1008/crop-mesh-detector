#!/bin/bash
#SBATCH --account=brics.b36bl
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# usage: sbatch --job-name=NAME --time=HH:MM:SS run_stream_job.sbatch <setting n2|n4|n6> <strategy> <seed>
set -uo pipefail
SETTING=$1; STRATEGY=$2; SEED=$3
cd ~/work/crop-mesh-detector-baselines2
source ~/work/crop-mesh-detector/.venv/bin/activate
OUT=runs/${SETTING}_${STRATEGY}_s${SEED}
mkdir -p "$OUT"
COMMON="--config configs/${SETTING}.yaml --strategy ${STRATEGY} --seed ${SEED} --output-dir ${OUT}"
echo "setting=${SETTING} strategy=${STRATEGY} seed=${SEED} base=$(cat COMMIT_BASE)"

python -m src.train $COMMON --reset > "$OUT/trigger_0.log" 2>&1 || { cat "$OUT/trigger_0.log"; echo "BATCH 0 FAILED"; exit 1; }
n=1
while true; do
  if python -m src.train $COMMON --next-batch > "$OUT/trigger_$n.log" 2>&1; then
    n=$((n + 1))
  else
    if grep -q "stream is exhausted" "$OUT/trigger_$n.log"; then
      echo "stream exhausted after $((n - 1)) --next-batch trigger(s)"
      break
    fi
    cat "$OUT/trigger_$n.log"; echo "TRIGGER $n FAILED (not exhaustion)"; exit 1
  fi
done

# keep reports/logs only: model checkpoints would overflow the shared filesystem across ~45 runs
find "$OUT" -name '*.pt' -delete
rm -f "$OUT"/continual/*/knowledge.db
echo "=== DONE ==="
