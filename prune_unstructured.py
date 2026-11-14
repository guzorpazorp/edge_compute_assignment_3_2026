"""
Q2.2 / Q2.4 - Iterative unstructured magnitude pruning + fine-tuning.

Steps:
  * Every Conv2d and Linear weight joins one global pool. Each iteration
    zeroes the smallest-|w| weights in that pool until the cumulative sparsity
    target is reached (e.g. 30% -> 50% -> 70% -> 80%). Global ranking lets
    layers with many redundant weights be pruned harder than sensitive ones.
  * torch.nn.utils.prune keeps the original weights in `weight_orig` and a
    binary `weight_mask`. The forward pass uses weight = weight_orig * mask, so
    pruned weights stay at zero and get no gradient during fine-tuning.
  * After each pruning step the network is fine-tuned to recover accuracy.
  * Finally prune.remove() makes the pruning permanent: the mask is multiplied
    into `weight` and the extra parameters / hooks are dropped. The result is
    an ordinary state_dict with exact zeros.

    python prune_unstructured.py --targets 0.3,0.5,0.7,0.8 --ft-epochs 2
"""

import argparse
import time

import torch
import torch.nn.utils.prune as prune

from src.checkpoints import ckpt_path, load_float_model, save_float_checkpoint
from src.data import get_test_loader, get_train_loader
from src.utils import (CSVLogger, count_params, evaluate, file_size_mb, fmt_row, get_device,
                       gzip_size_mb, prunable_layers, set_seed, train_one_epoch,
                       weight_sparsity)


def parse_args():
    p = argparse.ArgumentParser(description="Iterative global magnitude (unstructured) pruning")
    p.add_argument("--fp32", default=ckpt_path("fp32"))
    p.add_argument("--out", default=ckpt_path("unstructured"))
    p.add_argument("--targets", default="0.3,0.5,0.7,0.8",
                   help="cumulative sparsity after each iteration")
    p.add_argument("--ft-epochs", type=int, default=2, help="fine-tune epochs per iteration")
    p.add_argument("--final-ft-epochs", type=int, default=2,
                   help="extra fine-tune epochs after the last iteration")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--log", default="results/logs/unstructured_pruning.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def fine_tune(model, train_loader, device, epochs, lr, amp):
    if epochs <= 0:
        return
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for e in range(epochs):
        loss, acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, amp, scaler)
        print(f"    fine-tune epoch {e + 1}/{epochs}: loss {loss:.4f} train acc {acc:.2f}%")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.cpu)
    amp = not args.no_amp
    targets = [float(t) for t in args.targets.split(",")]
    assert all(0 < a < b < 1 for a, b in zip([1e-9] + targets, targets)), "targets must increase"

    train_loader = get_train_loader(args.data_dir, args.batch_size, device, args.subset)
    test_loader = get_test_loader(args.data_dir, 512, device)

    model = load_float_model(args.fp32, device)
    base_acc = evaluate(model, test_loader, device)["acc"]
    print(f"Baseline FP32: acc {base_acc:.2f}% | sparsity {weight_sparsity(model):.1f}%")

    params_to_prune = [(m, "weight") for m in prunable_layers(model)]
    total_weights = sum(m.weight.numel() for m, _ in params_to_prune)
    logger = CSVLogger(args.log)
    logger.log({"iteration": 0, "target_sparsity": 0.0, "actual_sparsity": 0.0,
                "acc_after_prune": base_acc, "acc_after_finetune": base_acc, "time_s": 0.0})

    for it, target in enumerate(targets, start=1):
        t0 = time.time()
        # Prune exactly enough *additional* weights to reach the cumulative target.
        # (An integer amount is applied to the still-unpruned weights.)
        already = round(total_weights * weight_sparsity(model) / 100)
        to_prune = round(total_weights * target) - already
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured,
                                  amount=to_prune)
        sparsity = weight_sparsity(model)
        acc_pruned = evaluate(model, test_loader, device)["acc"]
        print(f"\nIteration {it}: target {target:.0%} -> sparsity {sparsity:.2f}% | "
              f"acc right after pruning {acc_pruned:.2f}%")

        epochs = args.ft_epochs + (args.final_ft_epochs if it == len(targets) else 0)
        fine_tune(model, train_loader, device, epochs, args.lr, amp)
        acc_ft = evaluate(model, test_loader, device)["acc"]
        row = {"iteration": it, "target_sparsity": target * 100, "actual_sparsity": sparsity,
               "acc_after_prune": acc_pruned, "acc_after_finetune": acc_ft,
               "time_s": time.time() - t0}
        logger.log(row)
        print(fmt_row(row))

    # Make the pruning permanent: weight = weight_orig * weight_mask, drop the hooks.
    for module, name in params_to_prune:
        prune.remove(module, name)
    assert not any(hasattr(m, "weight_mask") for m, _ in params_to_prune)

    final = evaluate(model, test_loader, device)
    sparsity = weight_sparsity(model)
    save_float_checkpoint(model, args.out, test_acc=final["acc"], sparsity=sparsity,
                          targets=args.targets)

    print("\n======= Unstructured pruning summary =======")
    print(f"Accuracy  : {base_acc:.2f}% -> {final['acc']:.2f}%")
    print(f"Sparsity  : {sparsity:.2f}% of Conv/Linear weights are exactly zero")
    print(f"Non-zero params: {count_params(model, nonzero_only=True):,} / {count_params(model):,}")
    print(f"File size : {file_size_mb(args.fp32):.2f} MB -> {file_size_mb(args.out):.2f} MB "
          f"(dense storage); gzip {gzip_size_mb(args.fp32):.2f} -> {gzip_size_mb(args.out):.2f} MB")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
