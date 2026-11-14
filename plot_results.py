"""
Figures for the technical report, built from the CSVs in results/.

    python plot_results.py   ->  results/plots/*.png
"""

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "results"
OUT = os.path.join(RES, "plots")
COLORS = {"fp32": "#5b6770", "ptq": "#2f7fc1", "qat": "#1a4f8b",
          "unstructured": "#d98a29", "structured": "#a3521b"}


def bar(ax, df, col, title, fmt="{:.2f}"):
    bars = ax.bar(df["model"], df[col], color=[COLORS.get(m, "gray") for m in df["model"]])
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, df[col]):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = os.path.join(RES, "benchmark_summary.csv")
    if os.path.exists(summary):
        df = pd.read_csv(summary)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        bar(axes[0], df, "top1_acc", "Top-1 accuracy (%)")
        axes[0].set_ylim(df["top1_acc"].min() - 3, 100)
        bar(axes[1], df, "size_mb", "Checkpoint size (MB)")
        bar(axes[2], df, "gzip_size_mb", "Gzipped size (MB)")
        bar(axes[3], df, "latency_bs1_median_ms", "CPU latency, batch 1 (ms)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "benchmark_overview.png"), dpi=150)
        print(f"wrote {OUT}/benchmark_overview.png")

    logs = {
        "baseline_training.csv": ("epoch", ["train_acc", "test_acc"], "Baseline training"),
        "qat_training.csv": ("epoch", ["train_acc", "fakequant_test_acc"], "QAT fine-tuning"),
        "structured_pruning.csv": ("epoch", ["train_acc", "test_acc"], "Structured pruning fine-tune"),
    }
    for file, (x, ys, title) in logs.items():
        path = os.path.join(RES, "logs", file)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6, 4))
        for y in ys:
            ax.plot(d[x], d[y], marker="o", label=y)
        ax.set(title=title, xlabel=x, ylabel="accuracy (%)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, file.replace(".csv", ".png")), dpi=150)
        print(f"wrote {OUT}/{file.replace('.csv', '.png')}")

    path = os.path.join(RES, "logs", "unstructured_pruning.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d["actual_sparsity"], d["acc_after_prune"], marker="x", label="right after pruning")
        ax.plot(d["actual_sparsity"], d["acc_after_finetune"], marker="o", label="after fine-tuning")
        ax.set(title="Iterative unstructured pruning", xlabel="weight sparsity (%)",
               ylabel="test accuracy (%)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "unstructured_pruning.png"), dpi=150)
        print(f"wrote {OUT}/unstructured_pruning.png")


if __name__ == "__main__":
    main()
