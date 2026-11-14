"""
Q1.1 / Q2.1 - Train the FP32 baseline ResNet-18 on CIFAR-10.

Recipe: SGD (Nesterov momentum 0.9, wd 5e-4), OneCycle LR schedule, random crop
+ flip augmentation, mixed precision on GPU. About 30 epochs reach ~94% test accuracy.

    python train_baseline.py                      # full run (use a GPU)
    python train_baseline.py --epochs 1 --subset 2000   # quick smoke test
"""

import argparse
import time

import torch

from src.checkpoints import ckpt_path, save_float_checkpoint
from src.data import get_test_loader, get_train_loader
from src.models import resnet18_cifar
from src.utils import CSVLogger, evaluate, fmt_row, get_device, set_seed, train_one_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Train FP32 ResNet-18 baseline on CIFAR-10")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate (OneCycle)")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--subset", type=int, default=None, help="use only N training images (debug)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=ckpt_path("fp32"))
    p.add_argument("--log", default="results/logs/baseline_training.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.cpu)
    amp = not args.no_amp
    print(f"Device: {device} | AMP: {amp and device.type == 'cuda'}")

    train_loader = get_train_loader(args.data_dir, args.batch_size, device, args.subset)
    test_loader = get_test_loader(args.data_dir, 512, device)

    model = resnet18_cifar().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    logger = CSVLogger(args.log)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler,
                                          amp, scaler, args.label_smoothing)
        test_acc = evaluate(model, test_loader, device, amp)["acc"]
        row = {"epoch": epoch, "train_loss": loss, "train_acc": train_acc, "test_acc": test_acc,
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": time.time() - t0}
        logger.log(row)
        print(fmt_row(row))

    # The final-epoch model is saved (not the best on the test set) so that the
    # reported test accuracy is not selected on the test data.
    final = evaluate(model, test_loader, device, amp)
    save_float_checkpoint(model, args.out, test_acc=final["acc"], epochs=args.epochs,
                          train_time_s=time.time() - start)
    print(f"\nFP32 baseline: test acc {final['acc']:.2f}% | top-5 {final['top5']:.2f}% "
          f"| total time {(time.time() - start) / 60:.1f} min | saved -> {args.out}")


if __name__ == "__main__":
    main()
