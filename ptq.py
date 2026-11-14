"""
Q1.2 - Post-Training Static Quantization (PTQ): FP32 -> INT8.

Steps:
  1. Load the trained FP32 model (eval mode, CPU).
  2. Fuse Conv+BN+ReLU / Conv+BN groups (BN is folded into the conv weights).
  3. Configure the quantization backend ('x86' / 'fbgemm' for Intel/AMD CPUs,
     'qnnpack' for ARM) and attach a static qconfig:
       - weights:     per-channel symmetric INT8 (min/max observer)
       - activations: per-tensor affine UINT8 (histogram observer)
  4. prepare(): insert observers that record activation ranges.
  5. Calibrate: run a subset of *training* images through the model so the
     observers learn scale/zero-point for every activation tensor.
  6. convert(): swap float modules for INT8 kernels using the calibrated ranges.
  7. Evaluate on the CIFAR-10 test set and save as TorchScript for deployment.

    python ptq.py --calib-samples 2048
"""

import argparse
import copy
import time

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from src.checkpoints import ckpt_path, load_float_model, save_int8_torchscript
from src.data import get_calibration_loader, get_test_loader
from src.utils import evaluate, file_size_mb, select_quant_backend, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Post-training static quantization (INT8)")
    p.add_argument("--fp32", default=ckpt_path("fp32"))
    p.add_argument("--out", default=ckpt_path("ptq"))
    p.add_argument("--backend", default="x86", choices=["x86", "fbgemm", "qnnpack"])
    p.add_argument("--calib-samples", type=int, default=2048)
    p.add_argument("--calib-batch-size", type=int, default=64)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def calibrate(model, loader):
    """Forward passes only: observers record min/max/histograms of activations."""
    model.eval()
    for i, (x, _) in enumerate(loader, 1):
        model(x)
        print(f"\r  calibration batch {i}/{len(loader)}", end="")
    print()


def main():
    args = parse_args()
    set_seed(args.seed)
    backend = select_quant_backend(args.backend)
    print(f"Quantization backend: {backend}")

    fp32 = load_float_model(args.fp32)              # CPU, eval mode
    test_loader = get_test_loader(args.data_dir, 256)
    fp32_acc = evaluate(fp32, test_loader)["acc"]
    print(f"FP32 test accuracy: {fp32_acc:.2f}%")

    # 1) fusion
    model = copy.deepcopy(fp32)
    model.fuse_model(qat=False)

    # 2) backend config + observers
    model.qconfig = get_default_qconfig(backend)
    print(f"qconfig: {model.qconfig}")
    prepare(model, inplace=True)

    # 3) calibration on a training subset
    calib_loader = get_calibration_loader(args.data_dir, args.calib_samples,
                                          args.calib_batch_size, seed=args.seed)
    t0 = time.time()
    calibrate(model, calib_loader)
    print(f"Calibrated on {args.calib_samples} training images in {time.time() - t0:.1f}s")

    # 4) convert to INT8
    convert(model, inplace=True)
    print(model)

    # 5) evaluate + export
    int8_acc = evaluate(model, test_loader)["acc"]
    save_int8_torchscript(model, args.out, backend=backend, method="ptq",
                          calib_samples=args.calib_samples, test_acc=int8_acc,
                          fp32_test_acc=fp32_acc)

    print("\n=========== PTQ summary ===========")
    print(f"FP32 accuracy : {fp32_acc:.2f}%   size {file_size_mb(args.fp32):.2f} MB")
    print(f"INT8 accuracy : {int8_acc:.2f}%   size {file_size_mb(args.out):.2f} MB")
    print(f"Accuracy drop : {fp32_acc - int8_acc:.2f} pp")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
