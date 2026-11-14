"""
Saving and loading of the five model checkpoints.

  fp32          checkpoints/fp32_resnet18.pth        float state_dict + architecture
  ptq           checkpoints/ptq_int8.pt              TorchScript INT8 (self-contained)
  qat           checkpoints/qat_int8.pt              TorchScript INT8 (self-contained)
  unstructured  checkpoints/unstructured_pruned.pth  float state_dict (pruning made permanent)
  structured    checkpoints/structured_pruned.pth    float state_dict + reduced inner widths

Float checkpoints store the per-block inner widths. That lets the structured-pruned
(physically smaller) network be rebuilt with no extra information.
INT8 checkpoints are TorchScript archives, so they run without this source code.
"""

import json
import os

import torch

from .models import resnet18_cifar
from .utils import select_quant_backend

CKPT_DIR = "checkpoints"
MODEL_FILES = {
    "fp32": "fp32_resnet18.pth",
    "ptq": "ptq_int8.pt",
    "qat": "qat_int8.pt",
    "unstructured": "unstructured_pruned.pth",
    "structured": "structured_pruned.pth",
}
INT8_MODELS = {"ptq", "qat"}


def ckpt_path(name, ckpt_dir=CKPT_DIR):
    return os.path.join(ckpt_dir, MODEL_FILES[name])


def save_float_checkpoint(model, path, **meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    torch.save({"arch": "resnet18_cifar",
                "inner_widths": list(model.inner_widths),
                "state_dict": state,
                "meta": meta}, path)


def load_float_model(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet18_cifar(inner_widths=ckpt["inner_widths"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def save_int8_torchscript(qmodel, path, **meta):
    """Trace the converted INT8 model and save it as a standalone TorchScript file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    qmodel.eval()
    example = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        traced = torch.jit.trace(qmodel, example, check_trace=False)
    torch.jit.save(traced, path, _extra_files={"meta.json": json.dumps(meta)})
    return traced


def load_int8_model(path, backend="x86"):
    # Packed INT8 weights are unpacked for the *active* quantized engine at load
    # time, so the engine must match the one used at conversion time. Load once with
    # the default, then reload if the stored metadata names a different backend.
    active = select_quant_backend(backend)
    extra = {"meta.json": ""}
    model = torch.jit.load(path, map_location="cpu", _extra_files=extra)
    meta = json.loads(extra["meta.json"] or "{}")
    wanted = meta.get("backend", active)
    if wanted != active and wanted != backend and wanted in torch.backends.quantized.supported_engines:
        return load_int8_model(path, meta["backend"])
    return model.eval(), meta


def load_model(name, ckpt_dir=CKPT_DIR):
    """Load any of the five models for CPU inference."""
    path = ckpt_path(name, ckpt_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}: checkpoint not found at {path}")
    if name in INT8_MODELS:
        return load_int8_model(path)[0]
    return load_float_model(path)
