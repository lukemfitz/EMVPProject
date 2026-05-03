"""
Timing side-channel audit for EMVP-Protected ResNet.

Question being tested
---------------------
The README claims:
  "The overhead is fixed regardless of image content or predicted class,
   which is itself a useful security property — timing side-channels
   reveal no information about the query."

This script tests that empirically by measuring per-image inference time
across many different inputs and looking for systematic correlations
between input properties and timing.

Method
------
1. Take N test images covering all 10 classes plus crafted inputs:
     - all-zero image
     - all-ones image
     - random gaussian noise
     - very large magnitude (would produce many subnormals after norm)
     - sparse vs dense CIFAR-10 images
2. Time each one R times.  Use the *minimum* across repeats per input as
   the signal (suppresses scheduler/GC noise; keeps any data-dependent
   floor).
3. Compare:
     a) baseline jitter: time variance from repeating the SAME image
     b) cross-input variance: time variance across DIFFERENT images
   If (b) >> (a), the timing leaks information.
4. Test specific hypotheses:
     - Does timing correlate with predicted class?
     - Does timing correlate with input L2 norm?
     - Does timing correlate with sparsity (fraction of activations < eps)?
"""

from __future__ import annotations

import time
import numpy as np
import os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from emvp_resnet import ResNet


def load_cifar10(n: int):
    """Try real CIFAR-10; fall back to synthetic-but-realistic images."""
    try:
        import torchvision, torchvision.transforms as T
        dset = torchvision.datasets.CIFAR10(os.path.join(_ROOT, "data"), train=False, download=True,
                                             transform=T.ToTensor())
        imgs, labs = [], []
        for img, label in dset:
            imgs.append(img.numpy()); labs.append(label)
            if len(imgs) >= n: break
        return np.array(imgs, dtype=np.float32), np.array(labs, dtype=np.int64)
    except ImportError:
        # Fallback: realistic-distribution synthetic images. Each is a Beta
        # mixture in [0,1] (CIFAR-10 pixel range) with some spatial structure
        # so different inputs produce genuinely different activation patterns.
        rng = np.random.default_rng(0)
        imgs = np.empty((n, 3, 32, 32), dtype=np.float32)
        for i in range(n):
            base = rng.beta(2.0, 2.0, size=(3, 32, 32)).astype(np.float32)
            # Add coarse spatial features (low-freq sinusoid + per-channel offset)
            yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing='ij')
            phase = rng.uniform(0, 2*np.pi, size=3)
            for c in range(3):
                base[c] += 0.3 * np.sin((xx + yy + phase[c]) / 4.0).astype(np.float32)
            imgs[i] = np.clip(base, 0, 1)
        # Labels are arbitrary for timing purposes (we recompute them via the model)
        return imgs, np.zeros(n, dtype=np.int64)


_CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

def normalize(img):
    return (img - _CIFAR_MEAN[:, None, None]) / _CIFAR_STD[:, None, None]


def craft_inputs():
    """Adversarial inputs designed to expose data-dependent timing."""
    rng = np.random.default_rng(0)
    inputs = {}
    inputs["zeros"]      = np.zeros((3, 32, 32), dtype=np.float32)
    inputs["ones"]       = np.ones((3, 32, 32), dtype=np.float32)
    inputs["gaussian"]   = rng.standard_normal((3, 32, 32)).astype(np.float32)
    # Large-magnitude input: forces large activations through the network
    inputs["large_pos"]  = np.full((3, 32, 32), 10.0, dtype=np.float32)
    inputs["large_neg"]  = np.full((3, 32, 32), -10.0, dtype=np.float32)
    # Subnormals: tiny non-zero floats slow down FP units on most CPUs
    inputs["subnormal"]  = np.full((3, 32, 32), 1e-310, dtype=np.float32) \
                            .astype(np.float64).astype(np.float32)
    # Sparse: mostly zeros after normalization → many post-ReLU zeros
    inputs["sparse"]     = np.zeros((3, 32, 32), dtype=np.float32)
    inputs["sparse"][:, ::4, ::4] = 1.0
    return inputs


def time_inference(model, x, repeats: int = 5):
    """Run inference `repeats` times, return all times in milliseconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model.emvp_forward(x)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def summarise(name, times):
    print(f"  {name:>15}  min={times.min():7.2f}ms  "
          f"med={np.median(times):7.2f}ms  "
          f"max={times.max():7.2f}ms  "
          f"std={times.std():6.3f}ms  "
          f"range={times.max()-times.min():6.2f}ms")


def main():
    print("=" * 75)
    print("EMVP timing side-channel audit")
    print("=" * 75)

    weights = np.load(os.path.join(_ROOT, "weights", "cifar10_weights.npy"),
                      allow_pickle=True).item()
    model = ResNet(n_blocks=2, num_classes=10, C_in=3, k=16, s=4,
                   seed=42, weights=weights)

    # Warm-up: JIT, page-fault, allocator caches
    print("\nWarming up …", flush=True)
    warm = np.zeros((3, 32, 32), dtype=np.float32)
    for _ in range(3):
        model.emvp_forward(warm)

    # ── Test 1: baseline jitter (same input repeated many times) ──────────
    print("\n--- Test 1: baseline jitter on a single fixed input -----------")
    print("  (this is the noise floor — true 'data-dependent' signals must")
    print("   exceed this to be meaningful)")
    raw_imgs, labels = load_cifar10(20)
    fixed = normalize(raw_imgs[0])
    base_times = time_inference(model, fixed, repeats=20)
    summarise("fixed_input", base_times)
    print(f"  noise CV = {base_times.std()/base_times.mean()*100:.2f}%")

    # ── Test 2: predicted-class bucket timing ─────────────────────────────
    # Use the model itself to predict each input's class, then group inputs
    # by predicted class. If timing depends on predicted class, the buckets
    # will have systematically different timings.
    print("\n--- Test 2: timing grouped by predicted class -----------------")
    by_class = {}
    for i in range(len(raw_imgs)):
        x = normalize(raw_imgs[i])
        pred = int(np.argmax(model.emvp_forward(x)))
        by_class.setdefault(pred, []).append(x)
    cls_times = {}
    for c in sorted(by_class):
        # Time first example of each represented class
        t = time_inference(model, by_class[c][0], repeats=8)
        cls_times[c] = t
        summarise(f"pred_class_{c}", t)
    cls_mins = np.array([cls_times[c].min() for c in sorted(cls_times)])
    print(f"\n  Per-class min spread: {cls_mins.max()-cls_mins.min():.2f}ms")
    print(f"  Compared to baseline jitter range: "
          f"{base_times.max()-base_times.min():.2f}ms")

    # ── Test 3: adversarially crafted inputs ──────────────────────────────
    print("\n--- Test 3: adversarial inputs -------------------------------")
    print("  (zeros, ones, large magnitudes, subnormals, sparse — designed")
    print("   to provoke FP edge-case slowdowns or activation patterns)")
    crafted = craft_inputs()
    crafted_times = {}
    for name, x in crafted.items():
        x_norm = normalize(x)
        t = time_inference(model, x_norm, repeats=8)
        crafted_times[name] = t
        summarise(name, t)
    crafted_mins = np.array([t.min() for t in crafted_times.values()])
    print(f"\n  Crafted-input min spread: "
          f"{crafted_mins.max()-crafted_mins.min():.2f}ms")

    # ── Test 4: correlation between input properties and timing ───────────
    print("\n--- Test 4: correlation analysis (50 images) -------------------")
    raw_50, lab_50 = load_cifar10(50)
    norm_imgs = np.stack([normalize(raw_50[i]) for i in range(50)])
    # Time each (3 reps) and predict each
    rep_times = []
    preds = []
    for i in range(50):
        t = time_inference(model, norm_imgs[i], repeats=3)
        rep_times.append(t.min())   # min across reps to suppress jitter
        preds.append(int(np.argmax(model.emvp_forward(norm_imgs[i]))))
    rep_times = np.array(rep_times)
    preds = np.array(preds)

    l2_norms   = np.array([np.linalg.norm(norm_imgs[i]) for i in range(50)])
    abs_means  = np.array([np.abs(norm_imgs[i]).mean() for i in range(50)])
    near_zero  = np.array([(np.abs(norm_imgs[i]) < 0.01).mean() for i in range(50)])

    def corr(a, b, label):
        c = np.corrcoef(a, b)[0, 1]
        print(f"    corr({label:18s}, time) = {c:+.3f}")

    print(f"  Per-image min times: mean={rep_times.mean():.2f}ms  "
          f"std={rep_times.std():.3f}ms  "
          f"range={rep_times.max()-rep_times.min():.2f}ms")
    print()
    corr(l2_norms,  rep_times, "input_L2_norm")
    corr(abs_means, rep_times, "input_abs_mean")
    corr(near_zero, rep_times, "input_near_zero_frac")
    corr(preds.astype(float), rep_times, "predicted_class")

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("Verdict")
    print("=" * 75)
    noise = base_times.max() - base_times.min()
    crafted_spread = crafted_mins.max() - crafted_mins.min()
    cls_spread = cls_mins.max() - cls_mins.min()
    correlations = [
        abs(np.corrcoef(l2_norms, rep_times)[0,1]),
        abs(np.corrcoef(abs_means, rep_times)[0,1]),
        abs(np.corrcoef(near_zero, rep_times)[0,1]),
        abs(np.corrcoef(preds.astype(float), rep_times)[0,1]),
    ]
    max_corr = max(correlations)

    print(f"  Baseline jitter range (single input, 20 reps) : {noise:.2f}ms")
    print(f"  Cross-class min spread (10 classes)           : {cls_spread:.2f}ms")
    print(f"  Adversarial input min spread (7 inputs)       : {crafted_spread:.2f}ms")
    print(f"  Max |corr(input feature, timing)|             : {max_corr:.3f}")
    print()
    if crafted_spread < 2 * noise and max_corr < 0.3:
        print("  → No statistically meaningful timing channel detected.")
        print("    Cross-input timing variation is within baseline jitter, and")
        print("    no input feature correlates with timing.")
    else:
        print("  → A timing signal exceeding baseline jitter was detected.")
        print("    The README claim is partially false — at least some")
        print("    information about the input is leaked through timing.")


if __name__ == "__main__":
    main()
