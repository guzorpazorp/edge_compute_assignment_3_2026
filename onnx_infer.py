"""
ONNX Runtime CPU inference + benchmark for the exported models.
This file needs only onnxruntime and numpy at inference time; PyTorch is not used.

    python onnx_infer.py --benchmark              # all .onnx models on the test set -> CSV
    python onnx_infer.py --model onnx_models/qat_int8.onnx --image cat.png
"""

import argparse
import glob
import os
import pickle
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(u8_nhwc):
    """uint8 NHWC images -> normalized float32 NCHW."""
    x = u8_nhwc.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return ((x - MEAN) / STD).astype(np.float32)


def load_cifar_test(data_dir):
    """Read the CIFAR-10 test batch directly (downloaded earlier by torchvision)."""
    path = os.path.join(data_dir, "cifar-10-batches-py", "test_batch")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return preprocess(x), np.array(d[b"labels"])


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def benchmark(path, x, y, threads, batch_size, runs):
    sess = make_session(path, threads)
    name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    preds = np.concatenate([sess.run(None, {name: x[i:i + batch_size]})[0].argmax(1)
                            for i in range(0, len(x), batch_size)])
    total = time.perf_counter() - t0
    one = x[:1]
    for _ in range(20):
        sess.run(None, {name: one})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: one})
        times.append((time.perf_counter() - t) * 1000)
    return {"model": os.path.splitext(os.path.basename(path))[0],
            "top1_acc": round(100.0 * (preds == y).mean(), 3),
            "size_mb": round(os.path.getsize(path) / 1e6, 3),
            "latency_bs1_median_ms": round(float(np.median(times)), 4),
            "latency_bs1_p95_ms": round(float(np.percentile(times, 95)), 4),
            "throughput_img_s": round(len(x) / total, 2),
            "threads": threads}, preds


def main():
    p = argparse.ArgumentParser(description="ONNX Runtime CPU inference")
    p.add_argument("--model", help="a single .onnx file (default: all in --onnx-dir)")
    p.add_argument("--onnx-dir", default="onnx_models")
    p.add_argument("--image", nargs="*")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latency-runs", type=int, default=200)
    p.add_argument("--out", default="results/onnx_benchmark.csv")
    args = p.parse_args()

    paths = [args.model] if args.model else sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))
    if not paths:
        raise SystemExit("No .onnx files found - run export_onnx.py first.")

    for img_path in args.image or []:
        img = np.asarray(Image.open(img_path).convert("RGB").resize((32, 32), Image.BILINEAR))
        x = preprocess(img[None])
        print(f"\n{img_path}")
        for path in paths:
            sess = make_session(path, args.threads)
            prob = softmax(sess.run(None, {sess.get_inputs()[0].name: x})[0])[0]
            top = prob.argsort()[::-1][:3]
            print(f"  {os.path.basename(path):28s} " +
                  "  ".join(f"{CLASSES[i]}:{prob[i]:.2f}" for i in top))

    if args.benchmark or not args.image:
        x, y = load_cifar_test(args.data_dir)
        rows = []
        for path in paths:
            row, _ = benchmark(path, x, y, args.threads, args.batch_size, args.latency_runs)
            rows.append(row)
            print(row)
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print("\n" + df.to_string(index=False) + f"\n\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
