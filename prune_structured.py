"""
Q2.3 / Q2.4 - Structured channel pruning with physical architecture reconstruction.

Where it is safe to remove channels in a ResNet:
  The output channels of conv2 in each BasicBlock are added to the shortcut,
  so they are tied to every other block in the same stage (and to the
  projection shortcut). Removing them would need identical masks across the
  whole residual chain. The *inner* channels (conv1 outputs = conv2 inputs) are
  private to their block, so removing them never breaks a tensor dimension.

Per BasicBlock:
  1. Score every conv1 output channel by importance:
       l1 : L1 norm of the filter weights   (default)
       bn : |gamma| of the following BatchNorm (a learned channel scale)
  2. Keep the top (1 - ratio) channels.
  3. Rebuild the block with smaller tensors:
       conv1.weight  [C_mid, C_in, 3, 3] -> [k, C_in, 3, 3]       (rows kept)
       bn1.*         [C_mid]             -> [k]
       conv2.weight  [C_out, C_mid, 3, 3] -> [C_out, k, 3, 3]     (columns kept)
  The new model is a real, smaller nn.Module (fewer params and MACs) with no
  masks and no zero-valued channels. It is then fine-tuned to recover accuracy.

    python prune_structured.py --ratio 0.5 --ft-epochs 10
"""

import argparse
import time

import torch

from src.checkpoints import ckpt_path, load_float_model, save_float_checkpoint
from src.data import get_test_loader, get_train_loader
from src.models import resnet18_cifar
from src.utils import (CSVLogger, count_macs, count_params, evaluate, file_size_mb, fmt_row,
                       get_device, set_seed, train_one_epoch)


def parse_args():
    p = argparse.ArgumentParser(description="Structured (channel) pruning of ResNet-18")
    p.add_argument("--fp32", default=ckpt_path("fp32"))
    p.add_argument("--out", default=ckpt_path("structured"))
    p.add_argument("--ratio", type=float, default=0.5,
                   help="fraction of inner channels removed in every BasicBlock")
    p.add_argument("--criterion", default="l1", choices=["l1", "bn"])
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--log", default="results/logs/structured_pruning.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def channel_importance(block, criterion):
    if criterion == "bn":
        return block.bn1.weight.detach().abs()
    return block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))


def select_channels(model, ratio, criterion):
    """Indices of the inner channels to keep, per block (sorted to preserve order)."""
    keep = {}
    for name, block in model.named_blocks():
        scores = channel_importance(block, criterion)
        n_keep = max(1, int(round(scores.numel() * (1.0 - ratio))))
        keep[name] = torch.topk(scores, n_keep).indices.sort().values
    return keep


@torch.no_grad()
def build_pruned_model(model, keep):
    """Create a narrower ResNet-18 and copy the surviving weights into it."""
    old_sd = model.state_dict()
    widths = [len(keep[name]) for name, _ in model.named_blocks()]
    pruned = resnet18_cifar(inner_widths=widths)
    new_sd = pruned.state_dict()

    for key in new_sd:
        tensor = old_sd[key]
        for name, idx in keep.items():
            idx = idx.to(tensor.device)
            if key == f"{name}.conv1.weight":
                tensor = tensor[idx]                                  # drop output filters
            elif key.startswith(f"{name}.bn1.") and tensor.dim() == 1:
                tensor = tensor[idx]                                  # matching BN entries
            elif key == f"{name}.conv2.weight":
                tensor = tensor[:, idx]                               # drop input channels
        assert tensor.shape == new_sd[key].shape, f"{key}: {tensor.shape} vs {new_sd[key].shape}"
        new_sd[key] = tensor.clone()

    pruned.load_state_dict(new_sd)
    return pruned


@torch.no_grad()
def self_test(model):
    """ratio=0 must reproduce the original network exactly (checks the copy logic)."""
    keep_all = select_channels(model, 0.0, "l1")
    clone = build_pruned_model(model, keep_all).eval()
    x = torch.randn(4, 3, 32, 32)
    diff = (clone(x) - model.cpu().eval()(x)).abs().max().item()
    assert diff < 1e-4, f"reconstruction mismatch {diff}"
    print(f"Self-test passed (ratio=0 reconstruction max |diff| = {diff:.2e})")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.cpu)
    amp = not args.no_amp

    train_loader = get_train_loader(args.data_dir, args.batch_size, device, args.subset)
    test_loader = get_test_loader(args.data_dir, 512, device)

    model = load_float_model(args.fp32)            # CPU for surgery
    self_test(model)
    base_params, base_macs = count_params(model), count_macs(model)
    base_acc = evaluate(model.to(device), test_loader, device)["acc"]

    # 1-3) score, select, rebuild
    keep = select_channels(model.cpu(), args.ratio, args.criterion)
    pruned = build_pruned_model(model, keep)
    print("\nInner widths per block:")
    for (name, block), w in zip(model.named_blocks(), pruned.inner_widths):
        print(f"  {name:9s} {block.conv1.out_channels:4d} -> {w:4d}")
    params, macs = count_params(pruned), count_macs(pruned)
    pruned.to(device)
    acc_pruned = evaluate(pruned, test_loader, device)["acc"]
    print(f"\nParams {base_params:,} -> {params:,} ({100 * params / base_params:.1f}%) | "
          f"MACs {base_macs / 1e6:.1f}M -> {macs / 1e6:.1f}M | "
          f"acc right after pruning {acc_pruned:.2f}%")

    # 4) fine-tune the reconstructed network
    logger = CSVLogger(args.log)
    logger.log({"epoch": 0, "train_loss": float("nan"), "train_acc": float("nan"),
                "test_acc": acc_pruned, "epoch_time_s": 0.0})
    optimizer = torch.optim.SGD(pruned.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, args.ft_epochs * len(train_loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for epoch in range(1, args.ft_epochs + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(pruned, train_loader, optimizer, device, scheduler,
                                          amp, scaler)
        test_acc = evaluate(pruned, test_loader, device)["acc"]
        row = {"epoch": epoch, "train_loss": loss, "train_acc": train_acc,
               "test_acc": test_acc, "epoch_time_s": time.time() - t0}
        logger.log(row)
        print(fmt_row(row))

    final_acc = evaluate(pruned, test_loader, device)["acc"]
    save_float_checkpoint(pruned, args.out, test_acc=final_acc, ratio=args.ratio,
                          criterion=args.criterion)

    print("\n======= Structured pruning summary =======")
    print(f"Accuracy : {base_acc:.2f}% -> {acc_pruned:.2f}% (pruned) -> {final_acc:.2f}% (fine-tuned)")
    print(f"Params   : {base_params:,} -> {params:,}")
    print(f"MACs     : {base_macs / 1e6:.1f}M -> {macs / 1e6:.1f}M")
    print(f"File size: {file_size_mb(args.fp32):.2f} MB -> {file_size_mb(args.out):.2f} MB")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
