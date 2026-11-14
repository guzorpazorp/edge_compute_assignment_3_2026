#!/usr/bin/env bash
# Reproducible end-to-end pipeline: train -> PTQ -> QAT -> pruning -> benchmark -> ONNX.
#
#   bash run_all.sh            # full run (GPU recommended, ~45-60 min on a Colab T4)
#   SMOKE=1 bash run_all.sh    # tiny settings to check that everything runs (~5 min)
#   SKIP_TRAIN=1 bash run_all.sh   # reuse existing checkpoints, only benchmark + export
#
# Console output of each stage is also saved to results/logs/<stage>.txt.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs checkpoints

THREADS=${THREADS:-2}
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "### SMOKE TEST MODE (tiny settings - numbers are NOT meaningful) ###"
  BASE="--epochs 1 --subset 2048"
  QAT="--epochs 1 --subset 2048 --freeze-bn-epoch 1 --freeze-observer-epoch 1"
  UNS="--targets 0.5,0.8 --ft-epochs 1 --final-ft-epochs 0 --subset 2048"
  STR="--ft-epochs 1 --subset 2048"
  PTQ="--calib-samples 256"
  BENCH="--latency-runs 20"
else
  BASE="--epochs ${EPOCHS:-30}"
  QAT="--epochs ${QAT_EPOCHS:-5}"
  UNS="--targets 0.3,0.5,0.7,0.8 --ft-epochs 2 --final-ft-epochs 2"
  STR="--ratio ${STRUCT_RATIO:-0.5} --ft-epochs ${STRUCT_EPOCHS:-10}"
  PTQ="--calib-samples 2048"
  BENCH="--latency-runs 200"
fi

run() {  # run <stage-name> <command...>
  local name=$1; shift
  echo -e "\n================ $name ================"
  echo "\$ $*"
  "$@" 2>&1 | tee "results/logs/$name.txt"
}

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  run 01_train_baseline     python train_baseline.py $BASE
  run 02_ptq                python ptq.py $PTQ
  run 03_qat                python qat.py $QAT
  run 04_prune_unstructured python prune_unstructured.py $UNS
  run 05_prune_structured   python prune_structured.py $STR
fi
run 06_benchmark            python benchmark.py --threads "$THREADS" $BENCH
run 07_export_onnx          python export_onnx.py
run 08_onnx_benchmark       python onnx_infer.py --benchmark --threads "$THREADS"
run 09_inference_demo       python infer.py --model all --samples 10
run 10_plots                python plot_results.py

echo -e "\nDone. Checkpoints: checkpoints/   ONNX: onnx_models/   CSVs + plots: results/"
