# EMVP-Protected ResNet

Privacy-preserving image classification using the **Encrypted Matrix-Vector
Product (EMVP)** protocol from:

> *Encrypted Matrix-Vector Products from Secret Dual Codes*  
> Benhamouda, Chen, Halevi, Ishai, Krawczyk, Mour, Rabin, Rosen  
> CCS 2025 / ePrint 2025/858

A client submits an image for classification. The server holds the model
weights and performs all computation, but **never sees the client's input
activations in the clear**. Only query privacy is implemented (weights are
public, activations are private).

---

## How It Works

### The Protocol

Every linear layer (convolution and FC) is replaced by an EMVP round-trip:

**Setup (offline, once per layer)**

The server encodes its weight matrix `M` using a secret dual code:

```
M_hat = M @ D    where D = [-C1^T | I_ℓ] ∈ F^{ℓ×n}
```

`C1` is the client's secret key. The key property is `C @ D^T = 0`
(dual orthogonality), which ensures the random blinding terms cancel on
decoding.

**Query (per inference)**

The client encrypts its activation vector `q`:

```
q_hat = α_j · (a^T C + [0 | q])_j    for each block j
```

where `a` is a fresh random blinding vector and `α_j` are secret scalar
masks (1D-SLSN). The server computes `M_hat @ q_hat` and returns the result.
The client decodes by multiplying by `1/α_j` and summing — the blinding
cancels by dual orthogonality, leaving `M @ q` exactly.

**Why 1D-SLSN scalar masking?** Without it, the server sees samples from a
linear secret-sharing code (LSN), which is vulnerable to algebraic attacks.
The scalar masks lift hardness to sub-exponential, defeating known attacks.

### Convolution as Matrix Multiply

Convolutions are reduced to matrix multiplies via **im2col**: each
receptive-field patch is flattened into a column, producing a patch matrix
`col` of shape `(C_in·kH·kW, n_patches)`. The conv weight `(C_out,
C_in·kH·kW)` then multiplies the entire patch matrix in one EMVP call,
rather than one call per patch.

### Batch Norm Folding

At inference time, each Conv+BN pair is a composition of two linear
transformations and is therefore a single linear map. The BN parameters
are absorbed into the conv weight matrix before encryption:

```
scale  = γ / √(var + ε)
W_fold = W · scale[:, None, None, None]
b_fold = β − mean · scale
```

The EMVP protocol sees `W_fold` as its matrix `M` — the protocol is
agnostic to whether `M` came from a single layer or a composition of layers.

### Fixed-Point Arithmetic

All EMVP arithmetic is over **F_p** where `P = 2^61 − 1` (61-bit Mersenne
prime). Floating-point activations are quantized before encryption and
dequantized after decoding:

```
quantise(x)   = round(x · SCALE) mod P        SCALE = 1024
dequantise(x) = to_signed(x) / SCALE²
```

Division by `SCALE²` (not `SCALE`) because the decoded value carries one
factor of `SCALE` from the weights and one from the activations.

---

## Architecture

CIFAR-10 style ResNet (also trained on MNIST with `C_in=1`):

```
Input (C_in, 32, 32)
  → Conv1 + BN (folded) + ReLU
  → Stage 1: n_blocks × ResidualBlock(16 → 16)
  → Stage 2: n_blocks × ResidualBlock(16 → 32, stride=2 on first block)
  → Stage 3: n_blocks × ResidualBlock(32 → 64, stride=2 on first block)
  → GlobalAveragePool
  → FC(64 → num_classes)
```

Each `ResidualBlock` is two ConvBNReLU layers with an optional 1×1 shortcut.
**Average pooling** (not max pooling) is used because max pooling is
nonlinear and cannot be evaluated on encrypted activations. **LayerNorm**
between blocks prevents quantization errors from accumulating across depth.

---

## Performance

Three successive optimizations bring per-image EMVP latency from ~3 seconds
down to ~90 ms (CPU):

| Stage | MNIST | CIFAR-10 |
|---|---|---|
| Python object arrays (baseline) | 3.02 s/img | 3.89 s/img |
| + C extension (`emvp_core.so`) | 0.51 s/img | 0.66 s/img |
| + Batched im2col | 0.061 s/img | 0.089 s/img |
| + CUDA GPU (`use_gpu=True`) | TBD | TBD |

**C extension** (`emvp_core.c`): implements `matmul_mod` using `unsigned
__int128` and the Mersenne identity `a·b ≡ hi + lo (mod 2^61−1)` to avoid
Python object-array arithmetic.

**Batched im2col**: instead of one EMVP round-trip per spatial patch, all
patches are stacked into a single matrix and processed in one batched
matrix-matrix product.

**CUDA GPU** (`use_gpu=True`): replaces the C extension matmul with a
`torch.einsum` on a CUDA device. Because CUDA supports float64, a 3-way
21-bit limb split is used to compute the exact modular matmul:

```
x = x_lo + x_mid·2^21 + x_hi·2^42

A @ B mod P  =  Σ_{i,j} (A_limb_i @ B_limb_j) · 2^(21(i+j))  mod P
```

Each limb product is ≤ 2^42 < 2^53 (float64 mantissa), so all 9
sub-products are exact. All 9 are computed in one `torch.einsum` call.

### Benchmark results (CPU, 500 images, pretrained weights)

| | MNIST | CIFAR-10 |
|---|---|---|
| Plaintext accuracy | 99.8% | 89.2% |
| EMVP accuracy | 99.8% | 89.0% |
| Prediction agreement | 100% | 99.8% |
| Mean L2 logit error | 0.020 | 0.115 |
| Plaintext speed | ~9 ms/img | ~12 ms/img |
| EMVP speed | 61 ms/img | 89 ms/img |
| Slowdown vs plaintext | 3× | 3× |

---

## Files

| File | Purpose |
|---|---|
| `emvp_resnet.py` | Core library: field arithmetic, EMVP protocol, ResNet model |
| `emvp_core.c` | C extension for fast F_p arithmetic (compile before use) |
| `train.py` | Train the ResNet on MNIST / CIFAR-10 and save BN-folded weights |
| `benchmark.py` | CLI benchmark: plaintext vs EMVP on both datasets |
| `colab_benchmark.ipynb` | Colab notebook: plaintext vs EMVP CPU vs EMVP GPU |
| `mnist_weights.npy` | Pretrained weights (99.68% MNIST test accuracy) |
| `cifar10_weights.npy` | Pretrained weights (90.12% CIFAR-10 test accuracy) |

---

## Quick Start

```bash
# 1. Compile the C extension (required for fast arithmetic)
cc -O3 -march=native -shared -fPIC -o emvp_core.so emvp_core.c

# 2. Run the benchmark (uses pretrained weights)
python3 benchmark.py

# 3. Retrain from scratch (optional)
python3 train.py --dataset mnist
python3 train.py --dataset cifar10
```

For GPU benchmarks, open `colab_benchmark.ipynb` in Google Colab with a
T4/A100 runtime. The notebook clones the repo, compiles the C extension,
and reports plaintext vs EMVP CPU vs EMVP GPU side by side.

---

## Dependencies

```
numpy >= 1.24
torch >= 2.0       # for GPU path and training only
torchvision        # for training and data loading only
```

The core EMVP protocol and ResNet inference (`emvp_resnet.py`) require only
NumPy. PyTorch is needed only for training (`train.py`) and the CUDA GPU
path (`use_gpu=True`).

---

## Security Notes

- **Threat model**: honest-but-curious server. The server follows the
  protocol faithfully but inspects all messages it receives.
- **What the server sees**: the encoded weight matrix `M_hat` and the
  encrypted query `q_hat`. Under the 1D-SLSN hardness assumption, `q_hat`
  reveals nothing about the client's activation `q`.
- **What the server does not see**: the secret key `(C1, alphas)`, the
  raw activation `q`, or any intermediate activation.
- **Weights are not hidden**: the right block of `M_hat` is literally `M`.
  This is the *query-only privacy* variant. Hiding the weights requires an
  additional Trapdoored Matrix (TDM) layer (see §4 of the paper).
- **Setup assumption**: the server receives `D` once to compute `M_hat`,
  then must delete it. If it retains `D`, it can invert the encryption.
