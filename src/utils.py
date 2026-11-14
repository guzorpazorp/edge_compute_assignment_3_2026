"""Training / evaluation / measurement helpers shared by every script."""

import csv
import gzip
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Reproducibility & devices
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def select_quant_backend(preferred: str = "x86") -> str:
    """Pick an INT8 kernel backend this CPU / PyTorch build supports."""
    engines = torch.backends.quantized.supported_engines
    for cand in (preferred, "x86", "fbgemm", "qnnpack"):
        if cand in engines:
            torch.backends.quantized.engine = cand
            return cand
    raise RuntimeError(f"No quantized engine available (have {engines})")


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, device, scheduler=None, amp=False,
                    scaler=None, label_smoothing=0.0):
    """One pass over `loader`. `scheduler` (if given) is stepped every batch."""
    model.train()
    use_amp = amp and device.type == "cuda"
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device=torch.device("cpu"), amp=False):
    """Returns accuracy (%), top-5 accuracy (%), predictions, confidences, labels."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    preds, confs, labels, top5_hits = [], [], [], 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        probs = logits.float().softmax(1)
        conf, pred = probs.max(1)
        top5_hits += (probs.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
        preds.append(pred.cpu()); confs.append(conf.cpu()); labels.append(y.cpu())
    preds, confs, labels = torch.cat(preds), torch.cat(confs), torch.cat(labels)
    return {
        "acc": 100.0 * (preds == labels).float().mean().item(),
        "top5": 100.0 * top5_hits / len(labels),
        "preds": preds, "confs": confs, "labels": labels,
    }


# --------------------------------------------------------------------------- #
# Model statistics
# --------------------------------------------------------------------------- #
def prunable_layers(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def count_params(model, nonzero_only=False):
    if nonzero_only:
        return int(sum((p != 0).sum().item() for p in model.parameters()))
    return int(sum(p.numel() for p in model.parameters()))


def weight_sparsity(model):
    """Fraction of zero weights across all Conv/Linear weight tensors (%)."""
    zeros = total = 0
    for m in prunable_layers(model):
        w = m.weight.detach()
        zeros += (w == 0).sum().item()
        total += w.numel()
    return 100.0 * zeros / max(total, 1)


@torch.no_grad()
def count_macs(model, input_size=(1, 3, 32, 32)):
    """Multiply-accumulate count of Conv2d + Linear layers for one image."""
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs += out.numel() // out.shape[0] * k

    def fc_hook(m, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model(torch.zeros(input_size, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def gzip_size_mb(path):
    """Size after gzip: shows the storage gain of sparse (zero-heavy) weights."""
    with open(path, "rb") as f:
        return len(gzip.compress(f.read(), compresslevel=6)) / 1e6


@torch.no_grad()
def measure_latency(fn, example, warmup=20, runs=100):
    """Median / p95 latency in ms of fn(example), after warm-up iterations."""
    for _ in range(warmup):
        fn(example)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(example)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return float(np.median(times)), float(np.percentile(times, 95))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class CSVLogger:
    """Appends dict rows to a CSV file, writing the header on first use."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._header = None
        if os.path.exists(path):
            os.remove(path)

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if self._header is None:
                w.writeheader()
                self._header = list(row.keys())
            w.writerow(row)


def fmt_row(d: dict) -> str:
    return " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())
