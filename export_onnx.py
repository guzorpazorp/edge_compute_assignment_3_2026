"""
Deliverable 8 - Export all five models to ONNX for framework-independent CPU deployment.

  FP32 / unstructured / structured : torch.onnx.export of the float model.
  PTQ-INT8 / QAT-INT8              : first try to export the PyTorch INT8 model
      directly (quantized ops -> ONNX QuantizeLinear/DequantizeLinear + QLinear*).
      If this PyTorch version cannot export it, or the exported graph's accuracy
      does not match, fall back to ONNX Runtime static quantization (QDQ format,
      per-channel INT8 weights) using the same calibration subset as ptq.py.
      The fallback quantizes the FP32 graph, so for QAT it does not keep the QAT
      weights; the manifest records which path was used.

Every exported file is checked with onnx.checker and ONNX Runtime.
Outputs go to onnx_models/*.onnx and onnx_models/export_manifest.csv.

    python export_onnx.py
"""

import argparse
import inspect
import os
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

from src.checkpoints import INT8_MODELS, MODEL_FILES, ckpt_path, load_model
from src.data import get_calibration_loader, get_test_loader

OPSET_FLOAT = 17
OPSET_INT8 = 13   # quantized-op symbolics in the TorchScript exporter target opset 13


def parse_args():
    p = argparse.ArgumentParser(description="Export models to ONNX")
    p.add_argument("--models", default=",".join(MODEL_FILES))
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--out-dir", default="onnx_models")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--calib-samples", type=int, default=2048)
    p.add_argument("--check-samples", type=int, default=2000,
                   help="test images used to validate each exported graph")
    p.add_argument("--int8-tolerance", type=float, default=1.5,
                   help="max accuracy gap (pp) between PyTorch INT8 and its ONNX export")
    return p.parse_args()


def torch_onnx_export(model, path, opset):
    """TorchScript-based exporter with a dynamic batch dimension."""
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False      # newer PyTorch defaults to the dynamo exporter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, torch.randn(1, 3, 32, 32), path, **kwargs)
    onnx.checker.check_model(onnx.load(path))


def ort_accuracy(path, x, y):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    preds = []
    for i in range(0, len(x), 250):
        preds.append(sess.run(None, {name: x[i:i + 250]})[0].argmax(1))
    return 100.0 * (np.concatenate(preds) == y).mean()


@torch.no_grad()
def torch_accuracy(model, x, y):
    return 100.0 * (model(torch.from_numpy(x)).argmax(1).numpy() == y).mean()


class CalibReader:
    """Feeds calibration batches to onnxruntime.quantization.quantize_static."""

    def __init__(self, loader, input_name):
        self.batches = iter([{input_name: xb.numpy()} for xb, _ in loader])

    def get_next(self):
        return next(self.batches, None)


def ort_static_quantize(fp32_onnx, out_path, calib_loader):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    src = fp32_onnx
    try:   # recommended pre-processing (shape inference + graph optimisation)
        from onnxruntime.quantization.shape_inference import quant_pre_process
        src = out_path.replace(".onnx", "_prep.onnx")
        quant_pre_process(fp32_onnx, src)
    except Exception as e:  # noqa: BLE001
        print(f"   (pre-processing skipped: {e})")
        src = fp32_onnx
    input_name = onnx.load(src).graph.input[0].name
    quantize_static(src, out_path, CalibReader(calib_loader, input_name),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    if src != fp32_onnx and os.path.exists(src):
        os.remove(src)
    onnx.checker.check_model(onnx.load(out_path))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    names = [n for n in args.models.split(",") if os.path.exists(ckpt_path(n, args.ckpt_dir))]

    xb, yb = next(iter(get_test_loader(args.data_dir, args.check_samples, subset=args.check_samples)))
    x, y = xb.numpy(), yb.numpy()
    calib_loader = get_calibration_loader(args.data_dir, args.calib_samples, 64)

    # The FP32 graph is needed first: it is the source for the INT8 fallback.
    fp32_onnx = os.path.join(args.out_dir, "fp32_resnet18.onnx")
    order = sorted(names, key=lambda n: n in INT8_MODELS)
    manifest = []
    for name in order:
        model = load_model(name, args.ckpt_dir)
        out = os.path.join(args.out_dir, f"{os.path.splitext(MODEL_FILES[name])[0]}.onnx")
        ref_acc = torch_accuracy(model, x, y)
        print(f"\n>> {name}: PyTorch acc on {len(y)} test images = {ref_acc:.2f}%")
        method = "torch.onnx.export"

        if name not in INT8_MODELS:
            torch_onnx_export(model, out, OPSET_FLOAT)
        else:
            ok = False
            try:
                torch_onnx_export(model, out, OPSET_INT8)
                gap = abs(ort_accuracy(out, x, y) - ref_acc)
                ok = gap <= args.int8_tolerance
                if not ok:
                    print(f"   direct INT8 export accuracy gap {gap:.2f}pp > tolerance")
            except Exception as e:  # noqa: BLE001
                print(f"   direct INT8 export not supported here: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:150]}")
            if not ok:
                assert os.path.exists(fp32_onnx), "export fp32 first (it is the fallback source)"
                print("   -> falling back to ONNX Runtime static quantization (QDQ)")
                ort_static_quantize(fp32_onnx, out, calib_loader)
                method = "onnxruntime.quantize_static(QDQ, from fp32)"

        onnx_acc = ort_accuracy(out, x, y)
        size = os.path.getsize(out) / 1e6
        print(f"   saved {out} ({size:.2f} MB) | ONNX Runtime acc {onnx_acc:.2f}% | {method}")
        manifest.append({"model": name, "onnx_file": out, "export_method": method,
                         "size_mb": round(size, 3), "torch_acc_subset": round(ref_acc, 2),
                         "onnx_acc_subset": round(onnx_acc, 2)})

    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "export_manifest.csv"), index=False)
    print(f"\nManifest -> {args.out_dir}/export_manifest.csv")


if __name__ == "__main__":
    main()
