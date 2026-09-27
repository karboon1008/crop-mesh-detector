#!/bin/bash
# usage: submit_stage.sh <setting> "<seeds>" [strategies]
set -euo pipefail
cd ~/work/crop-mesh-detector-baselines2
SETTING=$1; SEEDS=$2; STRATS=${3:-"continual fedavg fedavg_gated dpsgd dpsgd_gated"}
case "$SETTING" in n6) LIMIT=08:00:00;; *) LIMIT=05:00:00;; esac
for seed in $SEEDS; do
  for s in $STRATS; do
    sbatch --parsable --job-name="bl-${SETTING}-${s}-s${seed}" --time="$LIMIT" run_stream_job.sbatch "$SETTING" "$s" "$seed"
  done
done
