"""
Train a small ResNet on MNIST and/or CIFAR-10.

The PyTorch model exactly matches the emvp_resnet.py inference architecture:
  - Each ConvBN block applies ReLU internally (before the skip add).
  - Shortcut (when present) also applies ReLU.
  - Stem: Conv → BN → ReLU.
  - Three stages of n_blocks each (16/32/64 channels, stride-2 between stages).
  - GlobalAveragePool → FC.

After training, BN is folded into the preceding conv weights and the result
is saved as {dataset}_weights.npy for use with emvp_resnet.py.

Usage
-----
  python3 scripts/train.py                    # both datasets, n_blocks=2
  python3 scripts/train.py --dataset mnist    # MNIST only
  python3 scripts/train.py --dataset cifar10  # CIFAR-10 only
  python3 scripts/train.py --n-blocks 1       # smaller / faster model
"""

from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from emvp_resnet import batch_norm_fold

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# PyTorch model (must match emvp_resnet.py forward pass exactly)
# ---------------------------------------------------------------------------

class _ConvBN(nn.Module):
    """Conv2d + BN + ReLU — the building block of every layer."""
    def __init__(self, C_in, C_out, ksize=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, ksize, stride=stride, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class _ResBlock(nn.Module):
    """
    Residual block matching emvp_resnet.ResidualBlock:
      out      = ReLU(BN2(conv2(ReLU(BN1(conv1(x))))))
      residual = ReLU(BN_sc(conv_sc(x)))  if shape changes, else x
      output   = ReLU(out + residual)
    """
    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super().__init__()
        self.c1 = _ConvBN(C_in,  C_out, stride=stride)
        self.c2 = _ConvBN(C_out, C_out)
        self.sc = _ConvBN(C_in, C_out, ksize=1, stride=stride, pad=0) \
                  if (C_in != C_out or stride != 1) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out      = self.c2(self.c1(x))
        residual = self.sc(x) if self.sc is not None else x
        return F.relu(out + residual)


class SmallResNet(nn.Module):
    """16/32/64-channel ResNet matching emvp_resnet.ResNet."""
    def __init__(self, n_blocks: int = 2, num_classes: int = 10, C_in: int = 3):
        super().__init__()
        self.stem_conv = nn.Conv2d(C_in, 16, 3, stride=1, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(16)

        self.stage1 = nn.ModuleList([_ResBlock(16, 16) for _ in range(n_blocks)])
        self.stage2 = nn.ModuleList([
            _ResBlock(16 if i == 0 else 32, 32, stride=2 if i == 0 else 1)
            for i in range(n_blocks)
        ])
        self.stage3 = nn.ModuleList([
            _ResBlock(32 if i == 0 else 64, 64, stride=2 if i == 0 else 1)
            for i in range(n_blocks)
        ])
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.stage1: out = block(out)
        for block in self.stage2: out = block(out)
        for block in self.stage3: out = block(out)
        out = out.mean(dim=[2, 3])
        return self.fc(out)


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    W = conv.weight.detach().cpu().float().numpy()
    g = bn.weight.detach().cpu().float().numpy()
    b = bn.bias.detach().cpu().float().numpy()
    m = bn.running_mean.detach().cpu().float().numpy()
    v = bn.running_var.detach().cpu().float().numpy()
    return batch_norm_fold(W, g, b, m, v)


def extract_weights(model: SmallResNet) -> dict:
    w = {}
    w["conv1"] = _fold(model.stem_conv, model.stem_bn)
    for si, stage in enumerate([model.stage1, model.stage2, model.stage3], 1):
        for bi, block in enumerate(stage):
            p = f"s{si}b{bi}"
            w[f"{p}.c1"] = _fold(block.c1.conv, block.c1.bn)
            w[f"{p}.c2"] = _fold(block.c2.conv, block.c2.bn)
            if block.sc is not None:
                w[f"{p}.sc"] = _fold(block.sc.conv, block.sc.bn)
    W = model.fc.weight.detach().cpu().float().numpy().astype(np.float64)
    b = model.fc.bias.detach().cpu().float().numpy().astype(np.float64)
    w["fc"] = (W, b)
    return w


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def mnist_loaders(batch_size: int = 128):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(os.path.join(_ROOT, "data"), train=True,  download=True, transform=tf)
    test  = torchvision.datasets.MNIST(os.path.join(_ROOT, "data"), train=False, download=True, transform=tf)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=0),
        torch.utils.data.DataLoader(test,  batch_size=256,        shuffle=False, num_workers=0),
    )


def cifar10_loaders(batch_size: int = 128):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10(os.path.join(_ROOT, "data"), train=True,  download=True, transform=train_tf)
    test  = torchvision.datasets.CIFAR10(os.path.join(_ROOT, "data"), train=False, download=True, transform=test_tf)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=0),
        torch.utils.data.DataLoader(test,  batch_size=256,        shuffle=False, num_workers=0),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    dataset:     str,
    n_blocks:    int = 2,
    epochs:      int | None = None,
    batch_size:  int = 128,
    lr:          float = 0.1,
) -> dict:
    if dataset == "mnist":
        epochs    = epochs or 30
        C_in      = 1
        train_dl, test_dl = mnist_loaders(batch_size)
    else:
        epochs    = epochs or 100
        C_in      = 3
        train_dl, test_dl = cifar10_loaders(batch_size)

    model = SmallResNet(n_blocks=n_blocks, num_classes=10, C_in=C_in).to(DEVICE)
    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if dataset == "mnist":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 75])

    print(f"\n{'='*60}", flush=True)
    print(f"Training on {dataset.upper()}  |  device={DEVICE}  |  epochs={epochs}  |  n_blocks={n_blocks}", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        t0 = time.perf_counter()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct    += (model(x).argmax(1) == y).sum().item()
            n          += len(y)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_correct, val_n = 0, 0
            with torch.no_grad():
                for x, y in test_dl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    val_correct += (model(x).argmax(1) == y).sum().item()
                    val_n       += len(y)
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"loss={total_loss/n:.4f}  "
                f"train={correct/n*100:.1f}%  "
                f"val={val_correct/val_n*100:.1f}%  "
                f"({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    # Final test accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    final_acc = correct / total
    print(f"\n  Final test accuracy: {final_acc*100:.2f}%", flush=True)

    weights = extract_weights(model)
    weights_dir = os.path.join(_ROOT, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, f"{dataset}_weights.npy")
    np.save(path, weights)
    print(f"  Weights saved → {path}", flush=True)
    return weights


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["mnist", "cifar10", "both"], default="both")
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--epochs",   type=int, default=None,
                        help="Override epoch count (default: 30 for MNIST, 100 for CIFAR-10)")
    args = parser.parse_args()

    datasets = ["mnist", "cifar10"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        train(ds, n_blocks=args.n_blocks, epochs=args.epochs)
