"""
MNIST Benchmark: EMVP-Protected ResNet vs Plaintext ResNet
===========================================================
Measures accuracy and wall-clock speed for both inference modes on the
MNIST handwritten digit test set (28×28 greyscale, 10 classes).

Key metrics reported
--------------------
  - Per-image inference time (plaintext vs EMVP)
  - Speedup factor (EMVP / plaintext)
  - Top-1 accuracy on N test images for each mode
  - Prediction agreement between the two modes
  - Mean / max L2 error in logits (shows quantisation overhead)

Note: The model uses *random* weights (untrained), so absolute accuracy
will be near 10% for both modes. The meaningful comparisons are:
  1. Do plaintext and EMVP agree on predictions? (protocol correctness)
  2. How much slower is EMVP? (protocol cost)
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from emvp_resnet import ResNet, softmax


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist(n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (images, labels) as numpy arrays.
    images : (N, 1, 28, 28) float32 in [0, 1]
    labels : (N,) int

    Tries torchvision first, then tensorflow.keras, then sys.exit.
    """
    # --- torchvision --------------------------------------------------------
    try:
        import torchvision
        import torchvision.transforms as transforms

        dataset = torchvision.datasets.MNIST(
            root=os.path.join(_ROOT, "data"), train=False, download=True,
            transform=transforms.ToTensor(),
        )
        images, labels = [], []
        for img, label in dataset:
            images.append(img.numpy())   # (1, 28, 28) float32
            labels.append(label)
            if n is not None and len(images) >= n:
                break
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

    except ImportError:
        pass

    # --- tensorflow.keras ---------------------------------------------------
    try:
        import tensorflow as tf

        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test[:n].astype(np.float32) / 255.0          # (N, 28, 28)
        x_test = x_test[:, np.newaxis, :, :]                     # (N, 1, 28, 28)
        return x_test, y_test[:n].astype(np.int64)

    except ImportError:
        pass

    print("ERROR: neither torchvision nor tensorflow is installed.")
    print("Install one of them to run this benchmark.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    n_plain: int = 100,
    n_emvp:  int = 10,
    seed:    int = 42,
) -> None:
    print("=" * 65)
    print("MNIST Benchmark — EMVP-Protected ResNet vs Plaintext")
    print("=" * 65)
    print(f"  Plaintext images : {n_plain}")
    print(f"  EMVP images      : {n_emvp}")
    print(f"  Random seed      : {seed}")
    print()

    # Load data
    print("Loading MNIST test set …")
    n_load       = max(n_plain, n_emvp)
    images, labels = load_mnist(n=n_load)
    print(f"  Loaded {len(images)} images, shape {images[0].shape}")
    print()

    # Build model (C_in=1 for greyscale MNIST)
    print("Building ResNet (n_blocks=1, C_in=1) …")
    model = ResNet(n_blocks=1, num_classes=10, C_in=1, k=16, s=4, seed=seed)
    print("  Done.\n")

    # -----------------------------------------------------------------------
    # Plaintext pass
    # -----------------------------------------------------------------------
    print(f"Plaintext forward pass on {n_plain} images …", flush=True)
    plain_preds  = []
    plain_logits = []
    t0 = time.perf_counter()
    for i in range(n_plain):
        logits = model.plaintext_forward(images[i])
        plain_preds.append(int(np.argmax(logits)))
        plain_logits.append(logits)
    plain_elapsed = time.perf_counter() - t0

    plain_acc = np.mean(np.array(plain_preds) == labels[:n_plain])
    print(f"  Accuracy    : {plain_acc * 100:.1f}%  ({int(plain_acc * n_plain)}/{n_plain} correct)")
    print(f"  Total time  : {plain_elapsed:.3f} s")
    print(f"  Per image   : {plain_elapsed / n_plain * 1000:.2f} ms")
    print()

    # -----------------------------------------------------------------------
    # EMVP pass
    # -----------------------------------------------------------------------
    print(f"EMVP forward pass on {n_emvp} images …", flush=True)
    emvp_preds  = []
    emvp_logits = []
    t0 = time.perf_counter()
    for i in range(n_emvp):
        logits = model.emvp_forward(images[i])
        emvp_preds.append(int(np.argmax(logits)))
        emvp_logits.append(logits)
        elapsed_so_far = time.perf_counter() - t0
        print(
            f"  [{i+1:2d}/{n_emvp}]  true={labels[i]}  "
            f"pred={emvp_preds[-1]}  "
            f"({elapsed_so_far:.1f}s elapsed)",
            flush=True,
        )
    emvp_elapsed = time.perf_counter() - t0

    emvp_acc = np.mean(np.array(emvp_preds) == labels[:n_emvp])
    print()
    print(f"  Accuracy    : {emvp_acc * 100:.1f}%  ({int(emvp_acc * n_emvp)}/{n_emvp} correct)")
    print(f"  Total time  : {emvp_elapsed:.3f} s")
    print(f"  Per image   : {emvp_elapsed / n_emvp:.3f} s")
    print()

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    n_compare = min(n_plain, n_emvp)
    agree     = np.mean(
        np.array(plain_preds[:n_compare]) == np.array(emvp_preds[:n_compare])
    )
    l2_errors = [
        np.linalg.norm(plain_logits[i] - emvp_logits[i])
        for i in range(n_compare)
    ]
    speedup = (emvp_elapsed / n_emvp) / (plain_elapsed / n_plain)

    print("─" * 65)
    print("Summary")
    print("─" * 65)
    print(f"  Plaintext accuracy          : {plain_acc * 100:.1f}%")
    print(f"  EMVP accuracy               : {emvp_acc  * 100:.1f}%")
    print(f"  Prediction agreement        : {agree * 100:.1f}%  "
          f"({int(agree * n_compare)}/{n_compare} images)")
    print()
    print(f"  Mean L2 logit error         : {np.mean(l2_errors):.6f}")
    print(f"  Max  L2 logit error         : {np.max(l2_errors):.6f}")
    print()
    print(f"  Plaintext speed             : {plain_elapsed / n_plain * 1000:.2f} ms / image")
    print(f"  EMVP speed                  : {emvp_elapsed / n_emvp:.3f} s / image")
    print(f"  Slowdown (EMVP / plaintext) : {speedup:.0f}×")
    print()
    print("Note: random weights → ~10% accuracy for both modes is expected.")
    print("      High agreement + low L2 error validates the EMVP protocol.")
    print("=" * 65)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EMVP vs plaintext MNIST benchmark")
    parser.add_argument("--n-plain", type=int, default=100,
                        help="Number of images for plaintext pass (default: 100)")
    parser.add_argument("--n-emvp",  type=int, default=10,
                        help="Number of images for EMVP pass (default: 10)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for model init (default: 42)")
    args = parser.parse_args()

    run_benchmark(n_plain=args.n_plain, n_emvp=args.n_emvp, seed=args.seed)
