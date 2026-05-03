"""
Benchmark: EMVP-Protected ResNet vs Plaintext on MNIST and CIFAR-10.

Requires trained weights produced by train.py:
  weights/mnist_weights.npy   (python3 scripts/train.py --dataset mnist)
  weights/cifar10_weights.npy (python3 scripts/train.py --dataset cifar10)

Metrics reported per dataset
-----------------------------
  Plaintext accuracy    (on n_plain images)
  EMVP accuracy         (on n_emvp images)
  Prediction agreement  (plaintext vs EMVP on the same images)
  Mean / max L2 logit error
  Per-image latency for each mode
  Slowdown factor (EMVP / plaintext)

Usage
-----
  python3 scripts/benchmark.py                  # both datasets, default counts
  python3 scripts/benchmark.py --dataset mnist  # MNIST only
  python3 scripts/benchmark.py --n-plain 500 --n-emvp 100
"""

from __future__ import annotations

import argparse, os, sys, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from emvp_resnet import ResNet, softmax


# ---------------------------------------------------------------------------
# Normalization constants (must match train.py)
# ---------------------------------------------------------------------------

_MNIST_MEAN  = np.array([0.1307],                    dtype=np.float32)
_MNIST_STD   = np.array([0.3081],                    dtype=np.float32)
_CIFAR_MEAN  = np.array([0.4914, 0.4822, 0.4465],    dtype=np.float32)
_CIFAR_STD   = np.array([0.2023, 0.1994, 0.2010],    dtype=np.float32)

def _norm_mnist(img: np.ndarray) -> np.ndarray:
    return (img - _MNIST_MEAN[:, None, None]) / _MNIST_STD[:, None, None]

def _norm_cifar(img: np.ndarray) -> np.ndarray:
    return (img - _CIFAR_MEAN[:, None, None]) / _CIFAR_STD[:, None, None]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_torchvision(dataset: str, n: int):
    import torchvision, torchvision.transforms as T
    cls  = torchvision.datasets.MNIST if dataset == "mnist" else torchvision.datasets.CIFAR10
    dset = cls(os.path.join(_ROOT, "data"), train=False, download=True, transform=T.ToTensor())
    imgs, labs = [], []
    for img, label in dset:
        imgs.append(img.numpy())
        labs.append(label)
        if len(imgs) >= n:
            break
    return np.array(imgs, dtype=np.float32), np.array(labs, dtype=np.int64)


def load_data(dataset: str, n: int):
    """Returns (images, labels): images in [0,1], shape (N,C,H,W)."""
    try:
        return _load_torchvision(dataset, n)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if dataset == "mnist":
            (_, _), (x, y) = tf.keras.datasets.mnist.load_data()
            x = x[:n, np.newaxis].astype(np.float32) / 255.0
        else:
            (_, _), (x, y) = tf.keras.datasets.cifar10.load_data()
            x = x[:n].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            y = y.flatten()
        return x, y[:n].astype(np.int64)
    except ImportError:
        pass

    print("ERROR: install torchvision or tensorflow to run this benchmark.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Per-dataset config
# ---------------------------------------------------------------------------

_CONFIG = {
    "mnist": dict(
        C_in=1, n_blocks=2, num_classes=10,
        normalize=_norm_mnist,
        weight_file=os.path.join(_ROOT, "weights", "mnist_weights.npy"),
    ),
    "cifar10": dict(
        C_in=3, n_blocks=2, num_classes=10,
        normalize=_norm_cifar,
        weight_file=os.path.join(_ROOT, "weights", "cifar10_weights.npy"),
    ),
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_dataset(
    dataset:  str,
    n_plain:  int,
    n_emvp:   int,
    k:        int = 16,
    s:        int = 4,
    seed:     int = 42,
) -> None:
    cfg = _CONFIG[dataset]
    label = dataset.upper()
    sep   = "=" * 65

    print(f"\n{sep}")
    print(f"{label}  —  EMVP-Protected ResNet vs Plaintext")
    print(sep)

    # Load weights
    wfile   = cfg["weight_file"]
    weights = None
    if os.path.exists(wfile):
        weights = np.load(wfile, allow_pickle=True).item()
        print(f"  Loaded weights from {wfile}")
    else:
        print(f"  WARNING: {wfile} not found — using random weights (accuracy ~10%)")
        print(f"           Run: python3 scripts/train.py --dataset {dataset}")

    # Build model
    model = ResNet(
        n_blocks=cfg["n_blocks"],
        num_classes=cfg["num_classes"],
        C_in=cfg["C_in"],
        k=k, s=s, seed=seed,
        weights=weights,
    )

    # Load data
    n_load = max(n_plain, n_emvp)
    raw_images, labels = load_data(dataset, n_load)
    norm = cfg["normalize"]
    images = np.stack([norm(raw_images[i]) for i in range(n_load)])
    print(f"  Images loaded: {n_load}  shape={images[0].shape}")
    print()

    # ── Plaintext ────────────────────────────────────────────────────────────
    print(f"  Plaintext forward pass ({n_plain} images) …", flush=True)
    plain_preds, plain_logits = [], []
    t0 = time.perf_counter()
    for i in range(n_plain):
        lg = model.plaintext_forward(images[i])
        plain_preds.append(int(np.argmax(lg)))
        plain_logits.append(lg)
    plain_time = time.perf_counter() - t0
    plain_acc  = np.mean(np.array(plain_preds) == labels[:n_plain])

    print(f"    accuracy   : {plain_acc*100:.1f}%  ({int(plain_acc*n_plain)}/{n_plain})")
    print(f"    total time : {plain_time:.3f} s")
    print(f"    per image  : {plain_time/n_plain*1000:.2f} ms")
    print()

    # ── EMVP ─────────────────────────────────────────────────────────────────
    print(f"  EMVP forward pass ({n_emvp} images) …", flush=True)
    emvp_preds, emvp_logits = [], []
    t0 = time.perf_counter()
    for i in range(n_emvp):
        lg = model.emvp_forward(images[i])
        emvp_preds.append(int(np.argmax(lg)))
        emvp_logits.append(lg)
        elapsed = time.perf_counter() - t0
        print(
            f"    [{i+1:2d}/{n_emvp}]  true={labels[i]}  pred={emvp_preds[-1]}  "
            f"({elapsed:.1f}s elapsed)",
            flush=True,
        )
    emvp_time = time.perf_counter() - t0
    emvp_acc  = np.mean(np.array(emvp_preds) == labels[:n_emvp])

    print(f"\n    accuracy   : {emvp_acc*100:.1f}%  ({int(emvp_acc*n_emvp)}/{n_emvp})")
    print(f"    total time : {emvp_time:.3f} s")
    print(f"    per image  : {emvp_time/n_emvp:.3f} s")
    print()

    # ── Comparison ───────────────────────────────────────────────────────────
    n_cmp = min(n_plain, n_emvp)
    agree = np.mean(np.array(plain_preds[:n_cmp]) == np.array(emvp_preds[:n_cmp]))
    l2_errs = [np.linalg.norm(plain_logits[i] - emvp_logits[i]) for i in range(n_cmp)]
    slowdown = (emvp_time / n_emvp) / (plain_time / n_plain)

    print("  " + "─" * 61)
    print("  Summary")
    print("  " + "─" * 61)
    print(f"    Plaintext accuracy          : {plain_acc*100:.1f}%")
    print(f"    EMVP accuracy               : {emvp_acc*100:.1f}%")
    print(f"    Prediction agreement        : {agree*100:.1f}%  ({int(agree*n_cmp)}/{n_cmp})")
    print()
    print(f"    Mean L2 logit error         : {np.mean(l2_errs):.6f}")
    print(f"    Max  L2 logit error         : {np.max(l2_errs):.6f}")
    print()
    print(f"    Plaintext speed             : {plain_time/n_plain*1000:.2f} ms/image")
    print(f"    EMVP speed                  : {emvp_time/n_emvp:.3f} s/image")
    print(f"    Slowdown (EMVP/plaintext)   : {slowdown:.0f}×")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMVP vs plaintext benchmark")
    parser.add_argument("--dataset",  choices=["mnist", "cifar10", "both"], default="both")
    parser.add_argument("--n-plain",  type=int, default=1000,
                        help="Images for plaintext pass (default: 1000)")
    parser.add_argument("--n-emvp",   type=int, default=1000,
                        help="Images for EMVP pass (default: 1000)")
    parser.add_argument("--k",        type=int, default=16,
                        help="EMVP key dimension k (default: 16)")
    parser.add_argument("--s",        type=int, default=4,
                        help="EMVP number of blocks s (default: 4)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    datasets = ["mnist", "cifar10"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, n_plain=args.n_plain, n_emvp=args.n_emvp,
                    k=args.k, s=args.s, seed=args.seed)
