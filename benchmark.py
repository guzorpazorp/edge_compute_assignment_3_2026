"""
Automated CPU benchmark of all five models on the full CIFAR-10 test set.

For every model it measures:
  accuracy (top-1 / top-5), agreement with the FP32 predictions,
  checkpoint size (raw + gzip), parameters / non-zero parameters / sparsity, MACs,
  single-image latency (median + p95), and batched throughput.

Outputs:
  results/benchmark_summary.csv      one row per model (all five)
  results/q1_quantization_results.csv  FP32 vs PTQ-INT8 vs QAT-INT8
  results/q2_pruning_results.csv       FP32 vs unstructured vs structured
  results/predictions.csv            per-image predictions + confidence of every model

    python benchmark.py --threads 2
"""

import argparse
import os
import platform
import time

import pandas as pd
import torch

from src.checkpoints import INT8_MODELS, MODEL_FILES, ckpt_path, load_float_model, load_model
from src.data import get_test_loader
from src.models import CIFAR10_CLASSES
from src.utils import (count_macs, count_params, evaluate, file_size_mb, gzip_size_mb,
                       measure_latency, weight_sparsity)

QUESTION = {"fp32": "Q1+Q2", "ptq": "Q1", "qat": "Q1", "unstructured": "Q2", "structured": "Q2"}
DESCRIPTION = {
    "fp32": "Baseline ResNet-18 (FP32)",
    "ptq": "Post-training static quantization (INT8)",
    "qat": "Quantization-aware training (INT8)",
    "unstructured": "Global magnitude pruning (FP32, sparse)",
    "structured": "Channel-pruned + reconstructed (FP32)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FP32 / INT8 / pruned models on CPU")
    p.add_argument("--models", default=",".join(MODEL_FILES))
    p.add_argument("--threads", type=int, default=2, help="CPU threads (fixed for fairness)")
    p.add_argument("--batch-size", type=int, default=256, help="batch size for throughput")
    p.add_argument("--latency-runs", type=int, default=200)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


@torch.no_grad()
def benchmark_model(name, model, loader, args, ref_stats):
    path = ckpt_path(name, args.ckpt_dir)

    # accuracy + throughput over the whole test set
    t0 = time.perf_counter()
    res = evaluate(model, loader)
    total_s = time.perf_counter() - t0
    n = len(res["labels"])

    # single-image latency (batch size 1), the typical edge-inference case
    x1 = next(iter(loader))[0][:1]
    lat_med, lat_p95 = measure_latency(model, x1, warmup=20, runs=args.latency_runs)

    # structural statistics (INT8 TorchScript models share the FP32 architecture)
    if name in INT8_MODELS:
        stats = dict(ref_stats)
    else:
        stats = {"params": count_params(model),
                 "nonzero_params": count_params(model, nonzero_only=True),
                 "weight_sparsity_pct": weight_sparsity(model),
                 "macs_m": count_macs(model) / 1e6}

    row = {
        "model": name, "question": QUESTION[name], "description": DESCRIPTION[name],
        "precision": "INT8" if name in INT8_MODELS else "FP32",
        "top1_acc": res["acc"], "top5_acc": res["top5"],
        "size_mb": file_size_mb(path), "gzip_size_mb": gzip_size_mb(path),
        **stats,
        "latency_bs1_median_ms": lat_med, "latency_bs1_p95_ms": lat_p95,
        "throughput_img_s": n / total_s, "testset_time_s": total_s,
    }
    return row, res


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    names = [n for n in args.models.split(",") if os.path.exists(ckpt_path(n, args.ckpt_dir))]
    missing = set(args.models.split(",")) - set(names)
    if missing:
        print(f"Skipping (no checkpoint): {', '.join(sorted(missing))}")
    print(f"CPU: {platform.processor() or platform.machine()} | threads: {args.threads} | "
          f"torch {torch.__version__} | quant engine {torch.backends.quantized.engine}")

    loader = get_test_loader(args.data_dir, args.batch_size)

    # FP32 architecture statistics, reused for the INT8 models
    ref = load_float_model(ckpt_path("fp32", args.ckpt_dir))
    ref_stats = {"params": count_params(ref), "nonzero_params": count_params(ref, True),
                 "weight_sparsity_pct": weight_sparsity(ref), "macs_m": count_macs(ref) / 1e6}

    rows, preds = [], {}
    for name in names:
        print(f"\n>> {name}: {DESCRIPTION[name]}")
        model = load_model(name, args.ckpt_dir)
        row, res = benchmark_model(name, model, loader, args, ref_stats)
        rows.append(row)
        preds[name] = res
        print("   " + " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in row.items() if k not in ("description", "question")))

    df = pd.DataFrame(rows)
    # comparisons against the FP32 baseline
    if "fp32" in preds:
        base = df.set_index("model").loc["fp32"]
        fp32_pred = preds["fp32"]["preds"]
        df["acc_delta_vs_fp32"] = df["top1_acc"] - base["top1_acc"]
        df["agreement_with_fp32_pct"] = [
            100.0 * (preds[m]["preds"] == fp32_pred).float().mean().item() for m in df["model"]]
        df["size_reduction_x"] = base["size_mb"] / df["size_mb"]
        df["gzip_size_reduction_x"] = base["gzip_size_mb"] / df["gzip_size_mb"]
        df["latency_speedup_x"] = base["latency_bs1_median_ms"] / df["latency_bs1_median_ms"]
        df["throughput_speedup_x"] = df["throughput_img_s"] / base["throughput_img_s"]
    df["threads"] = args.threads
    df = df.round(4)

    df.to_csv(os.path.join(args.out_dir, "benchmark_summary.csv"), index=False)
    df[df["model"].isin(["fp32", "ptq", "qat"])].to_csv(
        os.path.join(args.out_dir, "q1_quantization_results.csv"), index=False)
    df[df["model"].isin(["fp32", "unstructured", "structured"])].to_csv(
        os.path.join(args.out_dir, "q2_pruning_results.csv"), index=False)

    # per-image predictions of every model
    labels = next(iter(preds.values()))["labels"]
    pred_df = pd.DataFrame({"image_idx": range(len(labels)), "true_label": labels.numpy(),
                            "true_class": [CIFAR10_CLASSES[i] for i in labels.tolist()]})
    for name, r in preds.items():
        pred_df[f"{name}_pred"] = r["preds"].numpy()
        pred_df[f"{name}_class"] = [CIFAR10_CLASSES[i] for i in r["preds"].tolist()]
        pred_df[f"{name}_conf"] = r["confs"].numpy().round(4)
        pred_df[f"{name}_correct"] = (r["preds"] == labels).numpy().astype(int)
    pred_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    cols = ["model", "precision", "top1_acc", "size_mb", "weight_sparsity_pct", "macs_m",
            "latency_bs1_median_ms", "throughput_img_s"]
    cols += [c for c in ("acc_delta_vs_fp32", "size_reduction_x", "latency_speedup_x") if c in df]
    print("\n" + "=" * 30 + " BENCHMARK SUMMARY " + "=" * 30)
    print(df[cols].to_string(index=False))
    print(f"\nCSV files written to {args.out_dir}/")


if __name__ == "__main__":
    main()
