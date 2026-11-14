"""
ResNet-18 adapted for CIFAR-10 (32x32 inputs) and made "quantization-ready".

Differences from torchvision's ImageNet ResNet-18:
  * 3x3 stride-1 stem convolution and no max-pool (32x32 images are tiny).
  * QuantStub / DeQuantStub at the input / output so eager-mode quantization
    knows where tensors enter and leave the INT8 domain.
  * The residual addition uses FloatFunctional, which becomes a quantized add
    after conversion (a plain "+" cannot be quantized in eager mode).
  * Every ReLU is its own module (never re-used) so it can be fused with the
    preceding Conv+BN.
  * The inner width of every BasicBlock (output channels of conv1) is
    configurable. Structured channel pruning uses this to rebuild a physically
    smaller network.
"""

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

# Output width of each stage and the stride of its first block.
STAGE_WIDTHS = [64, 128, 256, 512]
STAGE_STRIDES = [1, 2, 2, 2]
BLOCKS_PER_STAGE = 2

# Default inner widths (conv1 output channels) of the 8 BasicBlocks.
DEFAULT_INNER_WIDTHS = [w for w in STAGE_WIDTHS for _ in range(BLOCKS_PER_STAGE)]

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


class BasicBlock(nn.Module):
    """conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+ shortcut) -> ReLU."""

    def __init__(self, in_planes: int, mid_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # 1x1 projection when the spatial size or channel count changes.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.skip_add = FloatFunctional()   # quantizable residual add
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, self.shortcut(x))
        return self.relu2(out)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, inner_widths=None):
        super().__init__()
        inner_widths = list(inner_widths or DEFAULT_INNER_WIDTHS)
        assert len(inner_widths) == len(STAGE_WIDTHS) * BLOCKS_PER_STAGE
        self.inner_widths = inner_widths

        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        in_planes, block_idx = 64, 0
        for stage, (width, stride) in enumerate(zip(STAGE_WIDTHS, STAGE_STRIDES), start=1):
            blocks = []
            for b in range(BLOCKS_PER_STAGE):
                blocks.append(BasicBlock(in_planes, inner_widths[block_idx], width,
                                         stride if b == 0 else 1))
                in_planes = width
                block_idx += 1
            setattr(self, f"layer{stage}", nn.Sequential(*blocks))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.dequant(x)

    def named_blocks(self):
        """(name, BasicBlock) pairs in forward order, e.g. ('layer2.0', block)."""
        return [(n, m) for n, m in self.named_modules() if isinstance(m, BasicBlock)]

    def fuse_model(self, qat: bool = False):
        """Fuse Conv+BN(+ReLU) groups in place.

        PTQ fuses in eval mode (BN is folded into the conv weights).
        QAT fuses in train mode (BN stays live inside ConvBn modules so its
        statistics keep updating during fine-tuning).
        """
        fuse = fuse_modules_qat if qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for _, block in self.named_blocks():
            fuse(block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if len(block.shortcut) > 0:
                fuse(block.shortcut, [["0", "1"]], inplace=True)


def resnet18_cifar(num_classes: int = 10, inner_widths=None) -> ResNet18CIFAR:
    return ResNet18CIFAR(num_classes=num_classes, inner_widths=inner_widths)
