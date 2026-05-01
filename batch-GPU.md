# Batched GPU EMVP — Implementation Spec

This document describes the batched GPU extension to the EMVP protocol
implemented in `emvp_resnet.py`. Read `CLAUDE.md` first for full project
context. This file describes what to build and why.

---

## The Problem with the Current Implementation

In `emvp_resnet.py`, the `emvp_forward` method of `ConvBNReLU` processes
spatial patches one at a time:

```python
for j in range(n_patches):
    out_flat[:, j] = self.emvp.forward(col[:, j])
```

For a 32×32 feature map this is 1024 sequential EMVP calls. Each call does
`s` matrix-vector products. This means `s × 1024` separate small kernel
launches — far too small to saturate a GPU. The current implementation runs
in ~3 seconds per image on CPU; GPU won't help until the operations are batched.

---

## The Fix: Batched Matrix-Matrix Products

Instead of `p` independent matrix-vector products `M @ q_j`, compute the
full matrix-matrix product `M @ Q` where `Q = [q_1 | … | q_p]` in one shot.

The EMVP protocol extends naturally:

### Batched Query Encryption

Stack all `p` encrypted query vectors as columns:

```
Q_hat = [q_hat_1 | … | q_hat_p]  ∈ F^{n × p}
```

Each column `q_hat_j = (α_1·q_tilde_{j,1} | … | α_s·q_tilde_{j,s})` is
computed independently (embarrassingly parallel — vectorise across j).

The decoding keys also stack:
```
P_prime = [p'_1 | … | p'_p]  ∈ F^{s × p}
```
where `p'_j = (1/α_1, …, 1/α_s)` for query j.

### Batched Server Answer

Split `M_hat` into `s` column blocks of width `n/s` and `Q_hat` into `s`
row blocks of height `n/s`. The server computes:

```
M_prime = sum_{j=1}^{s}  M_hat_j @ Q_hat_j   ∈ F^{m × p}
```

where `M_hat_j ∈ F^{m × n/s}` and `Q_hat_j ∈ F^{n/s × p}`.

This is `s` matrix-matrix multiplications — the dominant cost — each of
shape `(m × n/s) · (n/s × p)`. On a GPU these run as a single batched
matmul (e.g. `torch.matmul` over a stacked batch dimension), fully
utilising GPU parallelism.

### Batched Decoding

The client recovers all `p` output columns simultaneously:

```
M1 @ M2 = M_prime * diag(p')    (column j scaled by 1/α_j)
```

In practice: `out[:, j] = M_prime[:, j] * p_prime[j]` for all j — a
single elementwise broadcast multiply, trivially vectorised.

### Correctness

Correctness follows column-by-column from the single-vector proof in `CLAUDE.md`.
Each column satisfies `M_prime[:, j] * p'_j = M @ q_j` by the same dual
orthogonality argument.

---

## What to Implement

### 1. New function: `emvp_encrypt_queries_batched`

Signature:
```python
def emvp_encrypt_queries_batched(
    Q_int: np.ndarray,        # (ell, p)  quantised query columns
    key: EMVPKey,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        Q_hat   : (n, p)  int64  encrypted query matrix
        P_prime : (s, p)  int64  decoding keys (one column per query)
    """
```

Implementation notes:
- Sample `a_matrix ← F^{k × p}` (one random coefficient vector per query)
- Compute `C_left = a_matrix  ∈ F^{k × p}`
- Compute `C_right = C1 @ a_matrix  ... wait, C1 is (k, ell), a is (k, p)`
  — actually `C_right = C1.T @ ... ` — be careful with shapes here.
  The codeword for query j is `c_j = (a_j | C1^T ... )`.
  See `emvp_encrypt_query` in `emvp_resnet.py` for the single-vector version
  and vectorise it across the p dimension.
- Embed each query: `Q_tilde[:, j] = c_j + [0_k | Q_int[:, j]]`
- Apply scalar masking block-by-block:
  `Q_hat[block_j] = α_j * Q_tilde[block_j]`  (broadcast α_j across p columns)
- Compute `P_prime[j, :] = 1/α_j` for all queries (same α_j for all queries
  in a batch — one set of scalars per layer, shared across the batch)
- All operations must use object-dtype arithmetic to avoid int64 overflow
  (see `CLAUDE.md` — this is a known footgun in this codebase)

### 2. New function: `emvp_server_answer_batched`

Signature:
```python
def emvp_server_answer_batched(
    M_hat: np.ndarray,    # (m, n)  int64  encoded weight matrix
    Q_hat: np.ndarray,    # (n, p)  int64  encrypted query matrix
    s: int
) -> np.ndarray:
    """
    Returns M_prime : (m, p)  int64
    """
```

Implementation notes:
- Split `M_hat` into `s` column blocks `M_hat_j` of width `n/s`
- Split `Q_hat` into `s` row blocks `Q_hat_j` of height `n/s`
- Accumulate: `M_prime += M_hat_j @ Q_hat_j` for each block j
- Use `matmul_mod` (already defined) — but note: for GPU, replace with
  `torch.matmul` over a GPU tensor (see GPU section below)
- Return `M_prime` in F_p (int64)

### 3. New function: `emvp_decode_batched`

Signature:
```python
def emvp_decode_batched(
    M_prime: np.ndarray,   # (m, p)  int64
    P_prime: np.ndarray,   # (s, p)  int64  — or (s,) if shared across batch
) -> np.ndarray:
    """
    Returns decoded : (m, p)  int64  (still in SCALE^2 units)
    """
```

If `α_j` is shared across all p queries in the batch (same key for all),
then `P_prime` is shape `(s,)` and decoding is:
```
decoded = sum_j  M_prime_block_j * (1/α_j)
```
which is a single scaled sum. If each query has its own α_j (more secure but
more complex), P_prime is `(s, p)` and you need a column-wise scale.

Start with the shared-key version (simpler) and note the limitation.

### 4. New method: `EMVPLayer.forward_batched`

Signature:
```python
def forward_batched(self, act_matrix: np.ndarray) -> np.ndarray:
    """
    act_matrix : (ell, p)  float64  — p activation vectors stacked as columns
    returns    : (m,   p)  float64  — p output vectors stacked as columns
    """
```

This replaces the inner loop in `ConvBNReLU.emvp_forward`.

### 5. Update `ConvBNReLU.emvp_forward`

Replace:
```python
for j in range(n_patches):
    out_flat[:, j] = self.emvp.forward(col[:, j])
```

With:
```python
out_flat = self.emvp.forward_batched(col)   # col is already (ell, n_patches)
```

---

## GPU Implementation (PyTorch backend)

Once the batched NumPy version works, add a GPU path.

### Key idea

Replace `matmul_mod` in the server answer with `torch.matmul` on GPU tensors.
The catch is that F_p arithmetic requires exact integer arithmetic, but
PyTorch's GPU matmul uses floating point. Two options:

**Option A — float64 on GPU (simpler, may lose precision)**
```python
import torch
M_hat_gpu = torch.tensor(M_hat, dtype=torch.float64).cuda()
Q_hat_gpu = torch.tensor(Q_hat, dtype=torch.float64).cuda()
M_prime   = (M_hat_gpu @ Q_hat_gpu) % P
```
Works if intermediate values don't exceed 2^53 (float64 mantissa). Safe
when `n/s` is small (few accumulations), but may lose precision for large
batches. Check carefully.

**Option B — int64 chunked matmul on GPU (exact)**
Split the matmul into chunks small enough that partial sums don't overflow
int64, accumulate on GPU, reduce mod P between chunks. More complex but
exact. Implement this if Option A shows precision errors.

### Suggested structure

Add a `use_gpu: bool` flag to `EMVPLayer.__init__`. When True:
- Move `M_hat` to GPU at init time
- Use torch matmul in `forward_batched`
- Keep encryption/decryption on CPU (they're small operations)

```python
class EMVPLayer:
    def __init__(self, M, name, k, s, rng, use_gpu=False):
        ...
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.M_hat_gpu = torch.tensor(
                self.M_hat, dtype=torch.float64).cuda()
```

---

## Testing

Add a test function `test_batched_correctness` that:
1. Builds a small EMVPLayer (e.g. m=16, ell=27, k=17, s=4)
2. Generates a random `Q ∈ F^{ell × p}` for p=32
3. Runs `forward` p times independently → `out_sequential` (m, p)
4. Runs `forward_batched` once → `out_batched` (m, p)
5. Asserts `np.allclose(out_sequential, out_batched, atol=1/SCALE)`

Also add a timing comparison function `benchmark_batched` that measures
wall-clock time for sequential vs batched for increasing values of p
(p = 1, 4, 16, 64, 256, 1024) and prints a table.

---

## Files to Modify

- `emvp_resnet.py` — add the four new functions/methods above
- `benchmark_emvp.py` — add `benchmark_batched` timing comparison

## Files NOT to Modify

- `CLAUDE.md` — project context document, do not edit

---

## Precision Reminder

The most common bug in this codebase is int64 overflow. Whenever two large
F_p values are multiplied, use object dtype:

```python
# WRONG — silently overflows
result = A.astype(np.int64) * B.astype(np.int64) % P

# CORRECT
result = (A.astype(object) * B.astype(object) % P).astype(np.int64)
```

This applies everywhere in the batched matmuls too. The existing `matmul_mod`
and `mul_mod` functions in `emvp_resnet.py` already do this correctly — use
them rather than writing raw numpy matmuls.
