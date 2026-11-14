"""
Q1.3 - Quantization-Aware Training (QAT): FP32 -> fake-quant fine-tune -> INT8.

Steps:
  1. Initialise from the pretrained FP32 checkpoint.
  2. Fuse Conv+BN(+ReLU) in *train* mode (fuse_modules_qat). BN stays live
     inside ConvBn modules so its statistics can adapt during fine-tuning.
  3. Attach the QAT qconfig and call prepare_qat(). This inserts FakeQuantize
     modules on weights and activations. In the forward pass they round values
     to the INT8 grid; in the backward pass they let gradients through
     (straight-through estimator). The network therefore learns weights that
     are robust to quantization error.
  4. Fine-tune for a few epochs with a small LR (GPU is fine for this stage).
     Later epochs freeze BN statistics and then the observers (scale/zero-point),
     so the final epochs train against fixed quantization parameters.
  5. Move to CPU, convert() to a real INT8 model, evaluate, export TorchScript.

    python qat.py --epochs 5
"""

import argparse
import copy
import time

import torch
from torch.ao.nn.intrinsic import qat as nniqat
from torch.ao.quantization import (convert, disable_observer, enable_observer,
                                   get_default_qat_qconfig, prepare_qat)

from src.checkpoints import ckpt_path, load_float_model, save_int8_torchscript
from src.data import get_test_loader, get_train_loader
from src.utils import (CSVLogger, evaluate, file_size_mb, fmt_row, get_device,
                       select_quant_backend, set_seed, train_one_epoch)


def parse_args():
    p = argparse.ArgumentParser(description="Quantization-aware training (INT8)")
    p.add_argument("--fp32", default=ckpt_path("fp32"))
    p.add_argument("--out", default=ckpt_path("qat"))
    p.add_argument("--backend", default="x86", choices=["x86", "fbgemm", "qnnpack"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--freeze-bn-epoch", type=int, default=3,
                   help="freeze BN running stats after this many epochs")
    p.add_argument("--freeze-observer-epoch", type=int, default=4,
                   help="freeze scale/zero-point after this many epochs")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--log", default="results/logs/qat_training.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def evaluate_fakequant(model, loader, device, observers_frozen):
    """Evaluate without letting test images update the observers (no test-set leakage)."""
    model.apply(disable_observer)
    acc = evaluate(model, loader, device)["acc"]
    if not observers_frozen:
        model.apply(enable_observer)
    return acc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.cpu)
    backend = select_quant_backend(args.backend)
    print(f"Device: {device} | quantization backend: {backend}")

    train_loader = get_train_loader(args.data_dir, args.batch_size, device, args.subset)
    test_loader_dev = get_test_loader(args.data_dir, 512, device)
    test_loader_cpu = get_test_loader(args.data_dir, 256)

    # 1) start from the pretrained FP32 weights
    fp32 = load_float_model(args.fp32)
    fp32_acc = evaluate(fp32.to(device), test_loader_dev, device)["acc"]
    print(f"FP32 test accuracy: {fp32_acc:.2f}%")

    # 2) fuse in train mode, 3) insert fake-quant modules
    model = copy.deepcopy(fp32).cpu().train()
    model.fuse_model(qat=True)
    model.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(model, inplace=True)
    model.to(device)
    init_acc = evaluate_fakequant(model, test_loader_dev, device, observers_frozen=False)
    print(f"Fake-quant accuracy before QAT fine-tuning: {init_acc:.2f}%")

    # 4) QAT fine-tuning (no AMP: fake-quant should see full-precision values)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))
    logger = CSVLogger(args.log)
    logger.log({"epoch": 0, "train_loss": float("nan"), "train_acc": float("nan"),
                "fakequant_test_acc": init_acc, "bn_frozen": False, "observers_frozen": False,
                "epoch_time_s": 0.0})

    bn_frozen = obs_frozen = False
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        if epoch >= args.freeze_observer_epoch and not obs_frozen:
            model.apply(disable_observer)          # lock scale / zero-point
            obs_frozen = True
        if epoch >= args.freeze_bn_epoch and not bn_frozen:
            model.apply(nniqat.freeze_bn_stats)    # stop updating BN running stats
            bn_frozen = True
        test_acc = evaluate_fakequant(model, test_loader_dev, device, obs_frozen)
        row = {"epoch": epoch, "train_loss": loss, "train_acc": train_acc,
               "fakequant_test_acc": test_acc, "bn_frozen": bn_frozen,
               "observers_frozen": obs_frozen, "epoch_time_s": time.time() - t0}
        logger.log(row)
        print(fmt_row(row))

    # 5) convert the fine-tuned fake-quant model into a real INT8 model (CPU)
    model.cpu().eval()
    int8 = convert(model, inplace=False)
    print(int8)
    int8_acc = evaluate(int8, test_loader_cpu)["acc"]
    save_int8_torchscript(int8, args.out, backend=backend, method="qat", epochs=args.epochs,
                          test_acc=int8_acc, fp32_test_acc=fp32_acc)

    print("\n=========== QAT summary ===========")
    print(f"FP32 accuracy     : {fp32_acc:.2f}%   size {file_size_mb(args.fp32):.2f} MB")
    print(f"QAT INT8 accuracy : {int8_acc:.2f}%   size {file_size_mb(args.out):.2f} MB")
    print(f"Accuracy change   : {int8_acc - fp32_acc:+.2f} pp")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
