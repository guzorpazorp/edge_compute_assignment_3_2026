"""
CIFAR-10 data pipeline.

The whole dataset (about 150 MB as uint8) is kept as a tensor on the target
device, and augmentation (random crop with 4px padding + horizontal flip) runs
on that device in batches. On Colab this matters: with only 2 vCPUs, a normal
PIL/DataLoader pipeline cannot keep a T4 busy.
"""

import torch
import torch.nn.functional as F
import torchvision

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _load_split(data_dir: str, train: bool):
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
    images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
    labels = torch.tensor(ds.targets, dtype=torch.long)
    return images, labels


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 or [0,1] float batch -> normalized float batch."""
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random 32x32 crop from a zero-padded image, plus random h-flip."""
    b, c, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad))
    dev = x.device
    off_y = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    off_x = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (off_y[:, None] + torch.arange(h, device=dev))[:, None, :, None]   # B,1,H,1
    cols = (off_x[:, None] + torch.arange(w, device=dev))[:, None, None, :]   # B,1,1,W
    bidx = torch.arange(b, device=dev)[:, None, None, None]
    cidx = torch.arange(c, device=dev)[None, :, None, None]
    out = padded[bidx, cidx, rows, cols]
    flip = torch.rand(b, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class TensorLoader:
    """A minimal DataLoader replacement that iterates over in-memory tensors."""

    def __init__(self, images, labels, batch_size, shuffle=False, augment=False,
                 device="cpu", drop_last=False):
        self.images = images.to(device)
        self.labels = labels.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.labels)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n, device=self.images.device) if self.shuffle else None
        for i in range(len(self)):
            sl = slice(i * self.batch_size, min((i + 1) * self.batch_size, n))
            idx = order[sl] if order is not None else sl
            x = self.images[idx].float().div_(255.0)
            if self.augment:
                x = random_crop_flip(x)
            yield normalize(x), self.labels[idx]


def get_train_loader(data_dir="data", batch_size=256, device="cpu", subset=None, augment=True):
    images, labels = _load_split(data_dir, train=True)
    if subset:
        images, labels = images[:subset], labels[:subset]
    return TensorLoader(images, labels, batch_size, shuffle=True, augment=augment,
                        device=device, drop_last=True)


def get_test_loader(data_dir="data", batch_size=512, device="cpu", subset=None):
    images, labels = _load_split(data_dir, train=False)
    if subset:
        images, labels = images[:subset], labels[:subset]
    return TensorLoader(images, labels, batch_size, device=device)


def get_calibration_loader(data_dir="data", num_samples=2048, batch_size=64, seed=0, device="cpu"):
    """A fixed random subset of the *training* set without augmentation.

    Used for PTQ observer calibration and ONNX Runtime static quantization.
    It never touches the test set, so the test accuracy stays an unbiased estimate.
    """
    images, labels = _load_split(data_dir, train=True)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(labels), generator=g)[:num_samples]
    return TensorLoader(images[idx], labels[idx], batch_size, device=device)
