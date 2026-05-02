# EMVP-Protected ResNet — Project Context for Claude Code

This file gives you the full context you need to continue development.
Place it in the root of the VS Code project folder. Claude Code reads it automatically.

---

## What This Project Is

This project implements **secure neural network inference** using the **EMVP
(Encrypted Matrix-Vector Product)** protocol from:

> *Encrypted Matrix-Vector Products from Secret Dual Codes*
> Benhamouda, Chen, Halevi, Ishai, Krawczyk, Mour, Rabin, Rosen
> CCS 2025 / ePrint 2025/858

The goal is a client-server protocol for **ResNet image classification on
CIFAR-10** where the server holds the model weights and does all heavy
computation, but **never sees the client's input activations in the clear**.
Only query privacy is implemented (M is public, q is private).

---

## Layout

```
src/         emvp_resnet.py, emvp_core.{c,so}     # core library
scripts/     train.py, benchmark.py, benchmark_mnist.py,
             benchmark_checksum.py, plot_benchmark.py, timing_audit.py
weights/     mnist_weights.npy, cifar10_weights.npy
results/     benchmark_results.{md,png}, benchmark_checksum_results.txt
docs/        VERIFY_PLAN.md, VERIFY_ACTIONS.md, batch-GPU.md
data/        dataset cache (gitignored, downloaded on demand)
```

`scripts/` files all add `src/` to `sys.path` via a small shim, so they
import `emvp_resnet` directly. Run from the project root.

---

## The EMVP Protocol (Query-Only Privacy)

Every linear layer (conv and FC) is computed as follows instead of a plain
matrix multiply:

### Parameters
- Field: **F_p** where `P = 2^61 - 1` (61-bit Mersenne prime)
- Secret key: `sk = (C1 ∈ F^{k×ℓ}, alphas ∈ (F*)^s)`
- Derived codes:
  - Primal: `C = [I_k | C1]  ∈ F^{k×n}`,  `n = k + ℓ`
  - Dual:   `D = [-C1^T | I_ℓ]  ∈ F^{ℓ×n}`
  - Key property: `C @ D^T = 0` (dual orthogonality)

### Matrix Encoding (offline, server stores result)
```
M_hat = M @ D = [-M @ C1^T | M]  ∈ F^{m×n}
```
Dimension check: `M (m×ℓ) @ D (ℓ×n) = M_hat (m×n)` ✓
Note: the formula is `M @ D`, NOT `M @ D^T` — an earlier version had this wrong.

### Query Encryption (client, per inference)
```
1. Sample random a ← F^k
2. c = a^T C = (a | a^T C1)              # random primal codeword
3. q_tilde = c + [0_k | q]              # embed q, length n
4. split q_tilde into s blocks of length n/s
5. q_hat_j = alpha_j * q_tilde_j        # 1D-SLSN scalar masking
6. p' = (1/alpha_1, …, 1/alpha_s)       # decoding key, kept by client
```

### Server Answer
```
M' = [M_hat_j @ q_hat_j  for j in 1..s]  ∈ F^{m×s}
```

### Decoding (client)
```
Mq = M' @ p'
```
**Why this works:** The scalar masks cancel (`alpha_j * 1/alpha_j = 1`).
The primal codeword `c` vanishes by dual orthogonality (`M @ D @ c = 0`).
The remaining term gives `M @ D @ [0_k | q] = M @ q` exactly.

### Why 1D-SLSN Masking?
Without scalar masking the server sees basic LSN samples from the secret code
C, which can be broken in quasi-polynomial time by an algebraic attack
(Raz-style). Multiplying each block by a secret scalar lifts hardness to
sub-exponential, defeating known algebraic attacks.

---

## ResNet Architecture

CIFAR-10 style (32×32×3 input, 10 classes):

```
Input (3, 32, 32)
  → Conv1 + BN(folded) + ReLU + LayerNorm
  → Stage 1: n_blocks × ResidualBlock(16 → 16)
  → Stage 2: n_blocks × ResidualBlock(16 → 32, stride=2 on first block)
  → Stage 3: n_blocks × ResidualBlock(32 → 64, stride=2 on first block)
  → GlobalAveragePool  ← replaces max pooling (linear, client-side)
  → FC(64 → 10)
  → logits
```

Each `ResidualBlock` is:
```
x → ConvBNReLU → ConvBNReLU → (+x) → ReLU → LayerNorm
```
Optional 1×1 shortcut conv when C_in ≠ C_out or stride ≠ 1.

**Why average pooling?** Max pooling requires comparison (nonlinear),
which cannot be computed on encrypted values without client decryption.
Average pooling is a linear operation and folds into the preceding matmul.

**Why LayerNorm between blocks?** Without it, quantisation errors accumulate
exponentially across depth. LayerNorm (client-side, linear) keeps activations
bounded so errors stay O(1/SCALE) per layer.

**Why BatchNorm is folded into conv weights?** At inference time, BN is an
affine transformation that can be absorbed into the preceding conv weight and
bias, eliminating it as a separate operation:
```python
scale  = gamma / sqrt(var + eps)
W_fold = W * scale[:, None, None, None]
b_fold = beta - mean * scale
```

---

## Convolution as Matrix Multiply (im2col)

Convolution is reduced to EMVP via **im2col**: each receptive-field patch of
the input is flattened into a column vector, producing a matrix of shape
`(C_in * kH * kW, out_H * out_W)`. The conv weight `(C_out, C_in, kH, kW)`
is reshaped to `(C_out, C_in*kH*kW)`. Then:
```
output = weight_matrix @ patch_columns + bias
```
This means **one EMVP round-trip per spatial patch** (e.g., 1024 round-trips
for a 32×32 feature map). This is the dominant runtime cost.

---

## Fixed-Point Quantisation

All EMVP arithmetic is over **F_p** (integers), but model weights and
activations are floats. The conversion:

```python
SCALE = 1024  # 2^10

# Float → field element
quantise(x) = round(x * SCALE) mod P

# Field element → float  (after EMVP decode)
dequantise(x) = to_signed(x) / SCALE^2
```

**Why divide by SCALE² (not SCALE)?** The EMVP decode gives
`W_quantised @ q_quantised` which carries `SCALE` from the weights and
`SCALE` from the activations, so the result is in units of `SCALE²`.

**Critical bug that was fixed:** The old `quantise()` did:
```python
# WRONG — float % P loses precision (float64 has 53 bits, P ≈ 2^61)
np.round(x * SCALE).astype(object) % P
```
The correct version converts to `int64` FIRST (safe: rounded values are
small, well within int64 range), then reduces mod P:
```python
# CORRECT
np.round(x * SCALE).astype(np.int64)   # exact integer
then: .astype(object) % P              # exact modular reduction
```

**Overflow prevention:** All multiplications of large F_p values use Python
`object` arrays (arbitrary-precision integers) before casting back to `int64`.
Never multiply two large F_p values directly as `int64` — they will silently
overflow since `P ≈ 2^61` and `int64` max is `2^63`.

---

## Pretrained Weights

### Loading from torchvision (in Colab or locally)
```python
import torchvision
resnet_pt = torchvision.models.resnet18(pretrained=True)
resnet_pt.eval()
weights = extract_resnet_weights(resnet_pt)
model   = ResNet(weights=weights, n_blocks=2, num_classes=1000, k=32, s=4)
```

### How `extract_resnet_weights` works
It calls `extract_conv_bn_weights(conv, bn)` for each conv+BN pair, which:
1. Reads `conv.weight`, `bn.weight` (γ), `bn.bias` (β), `bn.running_mean`, `bn.running_var`
2. Calls `batch_norm_fold()` to produce `(W_fold, b_fold)`
3. Returns a dict keyed by layer names matching those used inside `ResNet.__init__`

### Weight dict key naming convention
```
"conv1"     → stem conv
"s1b0.c1"  → stage 1, block 0, conv 1
"s1b0.c2"  → stage 1, block 0, conv 2
"s1b0.sc"  → stage 1, block 0, shortcut (if present)
"s2b0.c1"  → stage 2, block 0, conv 1
... etc.
"fc"        → (W, b) for the final linear classifier
```

### How weights flow into the model
`ResNet.__init__` receives `weights: dict`, looks up each key, and passes
`(W_fold, b_fold)` to the corresponding `ConvBNReLU` call. If a key is
missing, that layer falls back to random He init. This means you can
partially load weights (e.g., only some stages).

---

## Key Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| Query-only privacy (M public) | Simpler protocol; sufficient when server owns the model and client owns the input |
| Average pooling instead of max | Max pooling is nonlinear; cannot be computed on encrypted values |
| LayerNorm between blocks | Prevents quantisation error accumulation across depth |
| 1D-SLSN (scalar masking) | Minimal-cost noise defeating algebraic attacks on LSN |
| `P = 2^61 - 1` | Mersenne prime; efficient modular arithmetic; large enough for SCALE² products |
| `SCALE = 1024` | 10 fractional bits; balance between precision and field overflow risk |
| im2col + EMVP per patch | Reduces conv to matmul; one round-trip per spatial patch |
| BN folded into conv | Eliminates BN as a separate layer at inference time |

---

## Protocol Security Model

- **Threat model:** Honest-but-curious server. It follows the protocol
  faithfully but examines everything it receives trying to infer `q`.
- **What the server sees:** `M_hat` (encoded public matrix), and for each
  query `q_hat` (encrypted query). It learns nothing about `q` under the
  **1D-SLSN hardness assumption**.
- **What the server does NOT see:** `C1`, `alphas`, `q`, or any intermediate
  activation vector.
- **Setup assumption:** The server receives `D = [-C1^T | I_ℓ]` once during
  setup to compute `M_hat`, then deletes it. If it retains `D`, it can
  recover `q` from `q_hat` (since `D @ q_hat` reveals `q` after undoing
  scalar masks). This deletion requirement is inherent to the protocol.
- **M is NOT hidden:** The right block of `M_hat` is literally `M`. This is
  the query-only-privacy variant. To hide M, a Trapdoored Matrix (TDM) layer
  must be added (see the paper, Section 4).

---

## Loading CIFAR-10 Data

### In Colab (easiest)
```python
import tensorflow as tf, numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.transpose(0,3,1,2).astype(np.float32) / 255.0
y_train = y_train.flatten()
label_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
horse_img = x_train[y_train == 7][0]   # first horse: shape (3,32,32)
```

### Raw pickle (no framework)
```python
import pickle, numpy as np
with open('cifar-10-batches-py/test_batch','rb') as f:
    batch = pickle.load(f, encoding='bytes')
images = batch[b'data'].reshape(-1,3,32,32).astype(np.float32) / 255.0
labels = np.array(batch[b'labels'])
```

### Displaying an image in Colab / matplotlib
```python
import matplotlib.pyplot as plt
plt.imshow(horse_img.transpose(1,2,0))   # (3,32,32) → (32,32,3) for matplotlib
plt.axis('off'); plt.show()
```

---

## Running the Code

All commands run from the project root.

```bash
# 0. (one-time) compile the C extension
cc -O3 -march=native -shared -fPIC -o src/emvp_core.so src/emvp_core.c

# Smoke test (no PyTorch needed — uses random weights, prints
# plaintext / EMVP / EMVP+checksum honest / EMVP+checksum cheating outputs)
python3 src/emvp_resnet.py

# Plaintext vs EMVP, with pretrained weights from weights/*.npy
python3 scripts/benchmark.py

# EMVP with vs without the Freivalds checksum, including a cheating-server run
python3 scripts/benchmark_checksum.py
```

In Colab or another notebook (paths assume the project root is the cwd):
```python
import sys; sys.path.insert(0, "src")
from emvp_resnet import ResNet
import numpy as np

weights = np.load("weights/cifar10_weights.npy", allow_pickle=True).item()
model   = ResNet(n_blocks=2, num_classes=10, C_in=3, k=16, s=4, weights=weights)

logits          = model.emvp_forward(image)                       # plain EMVP
logits, passed  = model.emvp_forward_with_verify(image)           # + checksum
logits, passed  = model.emvp_forward_with_verify(image, cheat=True)  # cheat sim
```

---

## Known Limitations and Next Steps

- **Speed:** Each EMVP round-trip uses Python object arrays for exact
  arithmetic. On a 32×32 image, a single conv layer does 1024 round-trips.
  This is correct but slow (~3s per image for a 1-block ResNet). Speedup
  options: batch multiple patches per EMVP call, use C extensions, or use
  a GPU-friendly finite field library.

- **Accuracy:** With random weights the model predicts random classes.
  With pretrained weights (from torchvision) the predictions are meaningful,
  but CIFAR-10 images are 32×32 while ImageNet-pretrained models expect
  224×224. Consider using a CIFAR-10-specific checkpoint or upsampling inputs.

- **Scale sensitivity:** `SCALE = 1024` gives ~0.001 per-element error.
  Increasing SCALE improves precision but risks overflow if activations are
  large. The benchmark includes a scale sensitivity study.

- **Depth:** With `n_blocks=1` (3 stages × 1 block × 2 convs + stem + FC
  = 8 linear layers), LayerNorm keeps errors bounded. Deeper models
  (n_blocks=2 or 3) work but accumulate more error.

- **The `M @ D` formula:** Earlier notes (and some intermediate code
  versions) incorrectly wrote `M @ D^T`. The correct formula is `M @ D`
  because `D ∈ F^{ℓ×n}` and `M ∈ F^{m×ℓ}`, so `M @ D ∈ F^{m×n}`. The
  transpose `D^T ∈ F^{n×ℓ}` cannot right-multiply `M`.

---

## Dependencies

```
numpy       >= 1.24   # core arithmetic
torch       >= 2.0    # only needed for weight extraction (optional)
torchvision >= 0.15   # only needed for pretrained models (optional)
matplotlib            # only needed for image display
```

No other ML framework is required. The EMVP protocol and ResNet forward
passes are implemented entirely in NumPy.
