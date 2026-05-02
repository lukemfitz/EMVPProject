"""
Benchmark: EMVP-Protected ResNet with vs without per-layer Freivalds checksum.

Four conditions per dataset
---------------------------
  1. Plaintext                       — baseline accuracy + speed
  2. EMVP                            — existing protocol, no verification
  3. EMVP + checksum (honest)        — verification overhead, accuracy preserved
  4. EMVP + checksum (cheating)      — server flips one F_p entry per layer;
                                       reports detection rate

The Freivalds check works in F_p so it is exact: in the honest condition no
inference should ever fail; in the cheating condition each layer's check
fails with prob 1 - 1/P, so the per-image detection rate is essentially 100%.

Usage
-----
  python3 benchmark_checksum.py                  # both datasets, default counts
  python3 benchmark_checksum.py --dataset mnist  # MNIST only
  python3 benchmark_checksum.py --n-emvp 50 --n-cheat 50
"""

from __future__ import annotations

import argparse, os, sys, time
import numpy as np

from emvp_resnet import ResNet, softmax


# ---------------------------------------------------------------------------
# Normalization constants (must match train.py)
# ---------------------------------------------------------------------------

_MNIST_MEAN  = np.array([0.1307],                    dtype=np.float32)
_MNIST_STD   = np.array([0.3081],                    dtype=np.float32)
_CIFAR_MEAN  = np.array([0.4914, 0.4822, 0.4465],    dtype=np.float32)
_CIFAR_STD   = np.array([0.2023, 0.1994, 0.2010],    dtype=np.float32)

def _norm_mnist(img): return (img - _MNIST_MEAN[:, None, None]) / _MNIST_STD[:, None, None]
def _norm_cifar(img): return (img - _CIFAR_MEAN[:, None, None]) / _CIFAR_STD[:, None, None]


# ---------------------------------------------------------------------------
# Data loading (mirrors benchmark.py)
# ---------------------------------------------------------------------------

def _load_torchvision(dataset, n):
    import torchvision, torchvision.transforms as T
    cls  = torchvision.datasets.MNIST if dataset == "mnist" else torchvision.datasets.CIFAR10
    dset = cls("./data", train=False, download=True, transform=T.ToTensor())
    imgs, labs = [], []
    for img, label in dset:
        imgs.append(img.numpy())
        labs.append(label)
        if len(imgs) >= n:
            break
    return np.array(imgs, dtype=np.float32), np.array(labs, dtype=np.int64)


def _load_parquet(dataset, n):
    """Fallback loader for the Hugging Face cifar10 / mnist parquets."""
    import io
    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError:
        return None
    path = f"data/cifar10_hf/test.parquet" if dataset == "cifar10" else f"data/mnist_hf/test.parquet"
    if not os.path.exists(path):
        return None
    table = pq.read_table(path)
    cols  = table.column_names
    img_col   = next(c for c in cols if "img" in c.lower() or "image" in c.lower())
    label_col = next(c for c in cols if "label" in c.lower())
    img_data  = table[img_col].to_pylist()[:n]
    labels    = np.array(table[label_col].to_pylist()[:n], dtype=np.int64)
    imgs = []
    for entry in img_data:
        # HF cifar10 stores {'bytes': <png-bytes>, 'path': None}
        raw = entry["bytes"] if isinstance(entry, dict) else entry
        im  = np.array(Image.open(io.BytesIO(raw))).astype(np.float32) / 255.0
        if im.ndim == 2:
            im = im[None, ...]               # (H,W) → (1,H,W)
        else:
            im = im.transpose(2, 0, 1)       # (H,W,C) → (C,H,W)
        imgs.append(im)
    return np.stack(imgs), labels


def load_data(dataset, n):
    parq = _load_parquet(dataset, n)
    if parq is not None:
        return parq
    try:
        return _load_torchvision(dataset, n)
    except (ImportError, RuntimeError, Exception) as e:
        print(f"  torchvision loader failed: {type(e).__name__}: {str(e)[:80]}")
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
    print("ERROR: install torchvision or tensorflow to run this benchmark, or "
          "place the HF test parquet at data/{cifar10,mnist}_hf/test.parquet.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Per-dataset config
# ---------------------------------------------------------------------------

_CONFIG = {
    "mnist": dict(
        C_in=1, n_blocks=2, num_classes=10,
        normalize=_norm_mnist, weight_file="mnist_weights.npy",
    ),
    "cifar10": dict(
        C_in=3, n_blocks=2, num_classes=10,
        normalize=_norm_cifar, weight_file="cifar10_weights.npy",
    ),
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_dataset(
    dataset:  str,
    n_plain:  int,
    n_emvp:   int,
    n_verify: int,
    n_cheat:  int,
    k:        int = 16,
    s:        int = 4,
    seed:     int = 42,
) -> dict:
    cfg   = _CONFIG[dataset]
    label = dataset.upper()
    sep   = "=" * 70
    print(f"\n{sep}\n{label}  —  Checksum benchmark\n{sep}")

    weights = None
    if os.path.exists(cfg["weight_file"]):
        weights = np.load(cfg["weight_file"], allow_pickle=True).item()
        print(f"  Loaded weights from {cfg['weight_file']}")
    else:
        print(f"  WARNING: {cfg['weight_file']} not found — using random weights")

    model = ResNet(
        n_blocks=cfg["n_blocks"], num_classes=cfg["num_classes"],
        C_in=cfg["C_in"], k=k, s=s, seed=seed, weights=weights,
    )

    n_load = max(n_plain, n_emvp, n_verify, n_cheat)
    raw, labels = load_data(dataset, n_load)
    images = np.stack([cfg["normalize"](raw[i]) for i in range(n_load)])
    print(f"  Images loaded: {n_load}  shape={images[0].shape}\n")

    # ── 1. Plaintext ─────────────────────────────────────────────────────────
    print(f"  [1/4] Plaintext  ({n_plain} images) …", flush=True)
    plain_preds = []
    t0 = time.perf_counter()
    for i in range(n_plain):
        plain_preds.append(int(np.argmax(model.plaintext_forward(images[i]))))
    t_plain  = time.perf_counter() - t0
    plain_preds = np.array(plain_preds)
    plain_acc   = float(np.mean(plain_preds == labels[:n_plain]))
    print(f"        accuracy   : {plain_acc*100:.1f}%  ({int(plain_acc*n_plain)}/{n_plain})")
    print(f"        per image  : {t_plain/n_plain*1000:.2f} ms")

    # ── 2. EMVP (no checksum) ────────────────────────────────────────────────
    print(f"\n  [2/4] EMVP, no checksum  ({n_emvp} images) …", flush=True)
    emvp_preds = []
    t0 = time.perf_counter()
    for i in range(n_emvp):
        emvp_preds.append(int(np.argmax(model.emvp_forward(images[i]))))
    t_emvp     = time.perf_counter() - t0
    emvp_preds = np.array(emvp_preds)
    emvp_acc   = float(np.mean(emvp_preds == labels[:n_emvp]))
    print(f"        accuracy   : {emvp_acc*100:.1f}%  ({int(emvp_acc*n_emvp)}/{n_emvp})")
    print(f"        per image  : {t_emvp/n_emvp*1000:.2f} ms")

    # ── 3. EMVP + checksum (honest server) ───────────────────────────────────
    print(f"\n  [3/4] EMVP + checksum, honest  ({n_verify} images) …", flush=True)
    verify_preds, verify_passed = [], []
    t0 = time.perf_counter()
    for i in range(n_verify):
        lg, ok = model.emvp_forward_with_verify(images[i], cheat=False)
        verify_preds.append(int(np.argmax(lg)))
        verify_passed.append(ok)
    t_verify     = time.perf_counter() - t0
    verify_preds = np.array(verify_preds)
    verify_acc   = float(np.mean(verify_preds == labels[:n_verify]))
    pass_rate    = float(np.mean(verify_passed))
    print(f"        accuracy        : {verify_acc*100:.1f}%  ({int(verify_acc*n_verify)}/{n_verify})")
    print(f"        verification ok : {int(pass_rate*n_verify)}/{n_verify}  ({pass_rate*100:.1f}%)")
    print(f"        per image       : {t_verify/n_verify*1000:.2f} ms")

    # ── 4. EMVP + checksum (cheating server) ─────────────────────────────────
    print(f"\n  [4/4] EMVP + checksum, cheating  ({n_cheat} images) …", flush=True)
    cheat_preds, cheat_passed = [], []
    t0 = time.perf_counter()
    for i in range(n_cheat):
        lg, ok = model.emvp_forward_with_verify(images[i], cheat=True)
        cheat_preds.append(int(np.argmax(lg)))
        cheat_passed.append(ok)
    t_cheat     = time.perf_counter() - t0
    cheat_preds = np.array(cheat_preds)
    detection   = float(np.mean(np.logical_not(cheat_passed)))   # detected = !passed
    cheat_acc   = float(np.mean(cheat_preds == labels[:n_cheat]))  # accuracy if blindly trusted
    print(f"        per image       : {t_cheat/n_cheat*1000:.2f} ms")
    print(f"        detection rate  : {int(detection*n_cheat)}/{n_cheat}  ({detection*100:.1f}%)")
    print(f"        accuracy if no verification : {cheat_acc*100:.1f}%  "
          f"(would be silently wrong)")

    # ── Summary ──────────────────────────────────────────────────────────────
    overhead_ms = t_verify/n_verify*1000 - t_emvp/n_emvp*1000
    overhead_pc = (t_verify/n_verify) / (t_emvp/n_emvp) - 1.0
    n_cmp       = min(n_emvp, n_verify)
    same_pred   = float(np.mean(emvp_preds[:n_cmp] == verify_preds[:n_cmp]))

    print()
    print("  " + "─" * 66)
    print("  SUMMARY")
    print("  " + "─" * 66)
    print(f"    {'Condition':38s} {'acc':>7s}  {'ms/img':>9s}")
    print(f"    {'plaintext':38s} {plain_acc*100:6.1f}%  {t_plain/n_plain*1000:8.2f}")
    print(f"    {'EMVP, no checksum':38s} {emvp_acc*100:6.1f}%  {t_emvp/n_emvp*1000:8.2f}")
    print(f"    {'EMVP + checksum, honest':38s} {verify_acc*100:6.1f}%  {t_verify/n_verify*1000:8.2f}")
    print(f"    {'EMVP + checksum, cheating':38s} {'(reject)':>7s}  {t_cheat/n_cheat*1000:8.2f}")
    print()
    print(f"    Checksum overhead vs. EMVP        : +{overhead_ms:.2f} ms/image  "
          f"(+{overhead_pc*100:.1f}%)")
    print(f"    EMVP vs EMVP+verify agreement     : {same_pred*100:.1f}%  "
          f"({int(same_pred*n_cmp)}/{n_cmp})")
    print(f"    Honest verification pass rate     : {pass_rate*100:.1f}%  "
          f"(expected 100%)")
    print(f"    Cheating detection rate           : {detection*100:.1f}%  "
          f"(expected ~100%)")
    print(sep)

    return dict(
        dataset=dataset,
        plain_acc=plain_acc,    plain_ms=t_plain/n_plain*1000,
        emvp_acc=emvp_acc,      emvp_ms=t_emvp/n_emvp*1000,
        verify_acc=verify_acc,  verify_ms=t_verify/n_verify*1000,
        cheat_ms=t_cheat/n_cheat*1000,
        overhead_ms=overhead_ms, overhead_pc=overhead_pc,
        pass_rate=pass_rate, detection=detection,
        n_plain=n_plain, n_emvp=n_emvp, n_verify=n_verify, n_cheat=n_cheat,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMVP checksum benchmark")
    parser.add_argument("--dataset",  choices=["mnist", "cifar10", "both"], default="both")
    parser.add_argument("--n-plain",  type=int, default=200)
    parser.add_argument("--n-emvp",   type=int, default=200)
    parser.add_argument("--n-verify", type=int, default=200)
    parser.add_argument("--n-cheat",  type=int, default=200)
    parser.add_argument("--k",        type=int, default=16)
    parser.add_argument("--s",        type=int, default=4)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    datasets = ["mnist", "cifar10"] if args.dataset == "both" else [args.dataset]
    results  = []
    for ds in datasets:
        results.append(run_dataset(
            ds,
            n_plain=args.n_plain, n_emvp=args.n_emvp,
            n_verify=args.n_verify, n_cheat=args.n_cheat,
            k=args.k, s=args.s, seed=args.seed,
        ))

    # Cross-dataset summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("OVERALL")
        print("=" * 70)
        print(f"  {'metric':32s}  {'MNIST':>14s}  {'CIFAR-10':>14s}")
        for key, label, fmt in [
            ("plain_ms",    "plaintext (ms/img)",       "{:.2f}"),
            ("emvp_ms",     "EMVP (ms/img)",            "{:.2f}"),
            ("verify_ms",   "EMVP + checksum (ms/img)", "{:.2f}"),
            ("overhead_pc", "checksum overhead",        "{:.1%}"),
            ("plain_acc",   "plaintext accuracy",       "{:.1%}"),
            ("emvp_acc",    "EMVP accuracy",            "{:.1%}"),
            ("verify_acc",  "EMVP+checksum accuracy",   "{:.1%}"),
            ("pass_rate",   "honest pass rate",         "{:.1%}"),
            ("detection",   "cheat detection rate",     "{:.1%}"),
        ]:
            vals = "  ".join(f"{fmt.format(r[key]):>14s}" for r in results)
            print(f"  {label:32s}  {vals}")
        print("=" * 70)
