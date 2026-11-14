"""
Inference pipeline for all five PyTorch models (CPU deployment).

    # classify your own image(s) with any model
    python infer.py --model qat --image cat.png dog.jpg

    # show predictions of every model for some random test images
    python infer.py --model all --samples 8

    # accuracy on the full CIFAR-10 test set
    python infer.py --model structured --testset
"""

import argparse
import time

import torch
from PIL import Image

from src.checkpoints import MODEL_FILES, load_model
from src.data import get_test_loader, normalize
from src.models import CIFAR10_CLASSES
from src.utils import evaluate


def load_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(32, 32, 3)
    return normalize(x.permute(2, 0, 1).unsqueeze(0).contiguous())


@torch.no_grad()
def predict(model, x, topk=3):
    probs = model(x).softmax(1)
    conf, idx = probs.topk(topk, dim=1)
    return [[(CIFAR10_CLASSES[i], c) for i, c in zip(ix.tolist(), cf.tolist())]
            for ix, cf in zip(idx, conf)]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="CIFAR-10 inference with FP32 / INT8 / pruned models")
    p.add_argument("--model", default="all", choices=list(MODEL_FILES) + ["all"])
    p.add_argument("--image", nargs="*", help="image file(s) to classify")
    p.add_argument("--samples", type=int, default=0, help="N random test images to show")
    p.add_argument("--testset", action="store_true", help="evaluate on the full test set")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if not (args.image or args.samples or args.testset):
        args.samples = 5

    names = list(MODEL_FILES) if args.model == "all" else [args.model]
    models = {n: load_model(n, args.ckpt_dir) for n in names}
    print(f"Loaded: {', '.join(names)}")

    for path in args.image or []:
        x = load_image(path)
        print(f"\n{path}")
        for n, m in models.items():
            top = predict(m, x)[0]
            print(f"  {n:13s} " + "  ".join(f"{c}:{p:.2f}" for c, p in top))

    if args.samples:
        loader = get_test_loader(args.data_dir, 10000)
        x, y = next(iter(loader))
        idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(args.seed))[:args.samples]
        header = f"{'#':>5} {'true':11s} " + " ".join(f"{n:13s}" for n in names)
        print("\n" + header + "\n" + "-" * len(header))
        preds = {n: m(x[idx]).argmax(1) for n, m in models.items()}
        for j, i in enumerate(idx.tolist()):
            cells = []
            for n in names:
                c = CIFAR10_CLASSES[preds[n][j]]
                cells.append(f"{c + (' ' if c == CIFAR10_CLASSES[y[i]] else '*'):13s}")
            print(f"{i:5d} {CIFAR10_CLASSES[y[i]]:11s} " + " ".join(cells))
        print("(* = wrong prediction)")

    if args.testset:
        loader = get_test_loader(args.data_dir, 256)
        for n, m in models.items():
            t0 = time.time()
            r = evaluate(m, loader)
            print(f"{n:13s} test acc {r['acc']:.2f}%  top-5 {r['top5']:.2f}%  "
                  f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
