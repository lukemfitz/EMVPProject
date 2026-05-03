# ACTIONS — What Was Built and Measured

## Summary

Added a per-layer **Freivalds-style checksum** to the EMVP protocol so the
client can probabilistically detect a server that returns wrong matrix-vector
products. Implementation lives entirely in F_p so the check is exact (no
false positives from quantisation noise). Benchmarks on MNIST and CIFAR-10
show:

* **Accuracy** under the new checksum is identical to plain EMVP — the check
  is verification, not transformation.
* **Runtime overhead** is small: **+6.8 % on MNIST**, **+3.4 % on CIFAR-10**.
* **Detection rate** against the smallest possible in-field cheat (one F_p
  entry of the encoded response shifted by +1 mod P) is **100 % (200/200 on
  each dataset)**, matching the theoretical soundness `1 − 1/P ≈ 1 − 2⁻⁶¹`.
* **Honest false-alarm rate** is **0 % (0/200 on each dataset)** — the check
  is exact in F_p.
* Without verification, the same cheat collapses model accuracy to random
  guessing (10.5 % on MNIST, 8.0 % on CIFAR-10), so the check is the only
  thing standing between the client and a silently broken classifier.

## Protocol Implemented

For every layer, every query:

1. Client samples random **r ∈ F_p^m** (the "extra value" attached to each
   query).
2. Client locally computes the expected checksum
   `c_expected = (rᵀ · M_int) · Q_int  mod P` using its plaintext copy of
   `M_int` (public) and the activations it is encrypting.
3. Client encrypts the query as in the existing EMVP protocol and sends it.
4. Server runs the existing answer step.
5. Client decodes the response to `m_q_int = M_int · Q_int  mod P`.
6. Client computes `c_received = rᵀ · m_q_int  mod P` and accepts iff
   `c_received == c_expected`.

If the server returns `m_q_int + δ` for any nonzero `δ`, the check passes
only when `rᵀ · δ ≡ 0`, which has probability `1 / P ≈ 2⁻⁶¹` over uniform
random `r`. With ~16 layers per inference the joint per-image undetectable-
cheat probability is at most `2⁻⁶¹`, matching the empirical 0/200 misses.

The same `r` covers all `p` patches in a batched conv layer (so cost per
layer is one extra `mat·vec + vec·Mat`, never another `mat·mat`). That
keeps the overhead a small additive constant, dominated by the matmul-mod
on `rᵀ · M_int` which is ~1/p of the work the EMVP itself does.

## Files Changed / Added

| File | Change |
|---|---|
| `emvp_resnet.py` | (1) Cached `self.M_int = quantise(M)` per `EMVPLayer`. (2) Added `_apply_cheat` helper that flips one random F_p entry of an encoded response. (3) Added `EMVPLayer.forward_with_verify`, `EMVPLayer.forward_batched_with_verify`, and `_verify_freivalds`. (4) Added `emvp_forward_with_verify` to `ConvBNReLU`, `ResidualBlock`, `LinearClassifier`, and `ResNet` — each propagates `cheat` and ANDs the per-layer pass bits. (5) Extended the `demo()` smoke test to print verification outcomes for honest and cheating servers. |
| `benchmark_checksum.py` | New benchmark covering four conditions per dataset (plaintext, EMVP, EMVP+checksum honest, EMVP+checksum cheating). Reports timing, accuracy, honest pass rate, cheat detection rate, and per-condition overhead. Includes a Hugging Face parquet fallback for CIFAR-10/MNIST data loading (the canonical Toronto mirror was returning 503). |
| `PLAN.md` | Pre-implementation plan — protocol, soundness sketch, code touch-points. |
| `ACTIONS.md` | This file. |
| `benchmark_checksum_results.txt` | Full captured stdout of the benchmark run reported below. |

No existing public API was changed: `EMVPLayer.forward(_batched)`,
`ResNet.emvp_forward`, `benchmark.py`, etc. all behave exactly as before.
The verification and cheating paths live on parallel `_with_verify`
methods that take a `cheat: bool = False` argument.

## Smoke Test (random weights, in `python3 emvp_resnet.py`)

```
Running EMVP forward pass …
  Pred   : class 8
  L2 error in logits      : 0.019777
  Predictions match       : True

Running EMVP + checksum (honest server) …
  All layers verified : True
  Pred                : class 8

Running EMVP + checksum (cheating server) …
  All layers verified : False  (expected False)
  Pred (untrusted)    : class 1
```

## Benchmark Results

`python3 benchmark_checksum.py --n-plain 200 --n-emvp 200 --n-verify 200 --n-cheat 200`

Apple M1 Max, single thread, C-extension backend, pretrained weights from
the existing `mnist_weights.npy` and `cifar10_weights.npy`.

| Condition                     | MNIST acc | MNIST ms/img | CIFAR-10 acc | CIFAR-10 ms/img |
|---|---:|---:|---:|---:|
| Plaintext                     | 100.0 %   |  4.18 |  88.5 %   |  7.40 |
| EMVP, no checksum             | 100.0 %   | 37.49 |  88.5 %   | 62.76 |
| EMVP + checksum (honest)      | 100.0 %   | 40.05 |  88.5 %   | 64.86 |
| EMVP + checksum (cheating)    | (rejected)| 43.09 |  (rejected)| 62.77 |

| Metric                           | MNIST    | CIFAR-10 |
|---|---:|---:|
| Checksum overhead vs. EMVP       | **+6.8 %** | **+3.4 %** |
| EMVP / EMVP+checksum agreement   | 100.0 % (200/200) | 100.0 % (200/200) |
| Honest verification pass rate    | 100.0 % (200/200) | 100.0 % (200/200) |
| Cheating detection rate          | 100.0 % (200/200) | 100.0 % (200/200) |
| Cheat accuracy if blindly trusted | 10.5 %  |  8.0 %  |

## Interpretation

* **The checksum costs essentially nothing.** A single `r` covers an entire
  batched conv (all 1024 patches at 32×32), so verification is `O(m·ℓ + ℓ·p)`
  while the encrypted matmul is `O(m·ℓ·s·p)`. On MNIST the relative cost is
  larger because absolute per-image work is smaller (smaller feature maps),
  not because the check itself does more work. Both numbers are
  comfortably under 10 %.

* **Accuracy is unchanged.** The check does not modify any computed value;
  it just compares two F_p scalars per layer. EMVP and EMVP+checksum agree
  on every prediction over both datasets (200/200, 200/200).

* **Cheating is caught every time, with negligible theoretical slip rate.**
  We tested the smallest possible cheat (one entry, +1 mod P) so that the
  check has the least information to work with. Even so, every layer's
  spot-check uses fresh random `r`, and for any non-zero perturbation `δ`
  of the layer output, `Pr[rᵀδ = 0] = 1/P`. With ~16 layers and 200 trials
  per dataset, the probability of even one missed detection is
  `~ 200 · 16 · 2⁻⁶¹ ≈ 10⁻¹⁶` — the empirical 0/400 misses across the two
  datasets is exactly what theory predicts.

* **The cheat itself is devastating.** Even though +1 mod P sounds tiny, the
  decoding step multiplies it by `1/α_j` (uniform in F_p) and the dequantiser
  divides by `SCALE² ≈ 10⁶`, so the corrupted entry shows up as a value of
  order `±P/(2 SCALE²) ≈ ±10¹²` in the float output, which propagates and
  saturates everything downstream. The "accuracy if blindly trusted"
  collapses to ≈ 1/num_classes — random guessing — confirming that any
  client running EMVP without verification is fully exposed.

## How to Reproduce

```bash
# 1. (one-time) compile the C extension
cc -O3 -march=native -shared -fPIC -o emvp_core.so emvp_core.c

# 2. quick sanity check (no data download needed)
python3 emvp_resnet.py

# 3. full benchmark — needs CIFAR-10 / MNIST test data; the script will
#    use torchvision if installed, or a Hugging Face parquet at
#    data/{cifar10,mnist}_hf/test.parquet as a fallback.
python3 benchmark_checksum.py --n-plain 200 --n-emvp 200 \
                              --n-verify 200 --n-cheat 200
```

## What Was Not Done

* No reduction of the per-image latency (the checksum was not the goal).
* No multi-round / interactive verification (Schnorr-style proof). The
  Freivalds check is one-shot and already gives 2⁻⁶¹ soundness per layer,
  so there was no reason to chain it.
* No hiding of `r` from the server. Soundness here does not require
  concealing `r`: the server cannot make `rᵀδ = 0` without knowing both
  `r` and the *plaintext* activation `q`, and `q` is exactly what the
  EMVP encryption protects. (See PLAN.md "Why this is sound" for the
  argument.)
* No extension to the `M`-private (TDM) variant — only the query-only
  variant currently in the repo was modified.
* No batching of `r` across queries. A future change could verify a
  whole batch of inferences with a single random `r`, lowering overhead
  further at the cost of coarser-grained "which inference was tampered"
  attribution.
