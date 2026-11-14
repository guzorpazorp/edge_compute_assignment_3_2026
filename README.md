# Assignment 3: Model Optimization for Edge Inference (Quantization + Pruning)

ResNet-18 on CIFAR-10, compressed four ways and benchmarked on CPU:

| # | Model | Script | Checkpoint |
|---|---|---|---|
| 1 | FP32 baseline | `train_baseline.py` | `checkpoints/fp32_resnet18.pth` |
| 2 | PTQ (static, INT8) | `ptq.py` | `checkpoints/ptq_int8.pt` |
| 3 | QAT (INT8) | `qat.py` | `checkpoints/qat_int8.pt` |
| 4 | Unstructured pruning | `prune_unstructured.py` | `checkpoints/unstructured_pruned.pth` |
| 5 | Structured channel pruning | `prune_structured.py` | `checkpoints/structured_pruned.pth` |

Inference: `infer.py` (PyTorch, all 5 models) and `onnx_infer.py` (ONNX Runtime, CPU).
Benchmarking: `benchmark.py` → `results/*.csv`. ONNX export: `export_onnx.py` → `onnx_models/`.

## Repository layout
```
src/models.py        quantization-ready CIFAR ResNet-18 (QuantStub, FloatFunctional add,
                     fusable Conv-BN-ReLU, configurable inner widths for channel pruning)
src/data.py          CIFAR-10 loading, normalisation, GPU-side augmentation, calibration subset
src/utils.py         train/eval loops, sparsity/params/MACs, latency, CSV logging
src/checkpoints.py   save/load for all five checkpoint formats
train_baseline.py    Q1.1 / Q2.1  FP32 training
ptq.py               Q1.2         fusion -> backend qconfig -> calibration -> INT8 convert -> export
qat.py               Q1.3         FP32 init -> fake-quant -> QAT fine-tune -> INT8 convert -> export
prune_unstructured.py Q2.2/Q2.4   iterative global magnitude pruning + fine-tune + prune.remove
prune_structured.py  Q2.3/Q2.4    L1 channel importance -> rebuild smaller network -> fine-tune
infer.py             Q1.4 / Q2.5  inference with any model (images, test samples, test set)
benchmark.py         Q1.5 / Q2.6  automated benchmark of all models -> CSV
export_onnx.py       Del. 8       ONNX export (+ INT8 ONNX) with validation
onnx_infer.py        Del. 8       ONNX Runtime CPU inference + benchmark
plot_results.py                   report figures
run_all.sh                        whole pipeline, one command
colab_run.ipynb                   Colab runner (GPU training)
```

## Installation
Python 3.10–3.12 is recommended.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                       # with CUDA (Colab / NVIDIA GPU)
# CPU-only machine:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
The pinned PyTorch 2.6 supports the eager-mode `torch.ao.quantization` API used here.
CIFAR-10 (~170 MB) is downloaded automatically into `data/` on first use.

## Step-by-step execution
Full pipeline (GPU recommended: about 45–60 min on a Colab T4, more than 15 h on a laptop CPU):
```bash
bash run_all.sh
```
Quick check that every stage runs (about 5 min, results not meaningful):
```bash
SMOKE=1 bash run_all.sh
```
Or run each stage yourself:
```bash
python train_baseline.py --epochs 30                 # 1. FP32 baseline
python ptq.py --calib-samples 2048                   # 2. PTQ  -> INT8
python qat.py --epochs 5                             # 3. QAT  -> INT8
python prune_unstructured.py --targets 0.3,0.5,0.7,0.8 --ft-epochs 2   # 4.
python prune_structured.py --ratio 0.5 --ft-epochs 10                  # 5.
python benchmark.py --threads 2                      # 6. CSV benchmark of all 5 models
python export_onnx.py                                # 7. ONNX export
python onnx_infer.py --benchmark                     # 8. ONNX Runtime CPU benchmark
python infer.py --model all --samples 10             # 9. inference demo
python infer.py --model qat --image my_cat.jpg       #    classify your own image
python plot_results.py                               # 10. figures
```
**Colab:** open `colab_run.ipynb`, set *Runtime → T4 GPU*, and run the cells top to bottom.
To re-benchmark on another CPU (e.g. a laptop as the "edge device"), copy `checkpoints/` and run
`SKIP_TRAIN=1 bash run_all.sh`.

## Outputs
| File | Content |
|---|---|
| `results/benchmark_summary.csv` | per model: top-1/top-5 accuracy, size (raw/gzip), params, sparsity, MACs, latency (bs=1 median/p95), throughput, speedups vs FP32 |
| `results/q1_quantization_results.csv` | FP32 vs PTQ vs QAT subset |
| `results/q2_pruning_results.csv` | FP32 vs unstructured vs structured subset |
| `results/predictions.csv` | per test image: label, prediction + confidence + correct flag for every model |
| `results/onnx_benchmark.csv` | ONNX Runtime accuracy / latency / throughput |
| `results/logs/*.csv, *.txt` | training curves and console output of each stage |
| `results/plots/*.png` | report figures |
| `onnx_models/*.onnx`, `export_manifest.csv` | ONNX models and how each was exported |

## Notes on design choices
- **One baseline for both questions.** Q2 requires ResNet-18, so one FP32 ResNet-18 is the
  starting point for every compressed model, which makes the comparisons fair.
- **INT8 runs on CPU only.** PyTorch's eager-mode INT8 kernels (`x86`/`fbgemm`, `qnnpack`) are
  CPU kernels. QAT *fine-tuning* runs on the GPU with fake-quant, and the INT8 `convert` runs on CPU.
- **Unstructured pruning does not speed up dense kernels.** The zeros are still stored and
  multiplied. The gain is in compressibility (`gzip_size_mb`) and in sparse-capable runtimes.
- **Structured pruning removes only the inner channels of each BasicBlock.** These are conv1
  outputs and conv2 inputs, and nothing else depends on them. The residual-connected channels stay intact, so every
  tensor dimension stays valid. The network is rebuilt with smaller tensors, which gives real
  reductions in params, MACs and latency.
- **No test-set leakage.** PTQ calibration and ONNX Runtime quantization use a training subset.
  The final-epoch checkpoint is saved, not the best test epoch. QAT evaluation disables the observers.
