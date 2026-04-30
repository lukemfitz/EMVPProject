"""
EMVP-Protected ResNet Inference
================================
Implements a small ResNet (ResNet-20 style) for image classification where
every linear layer (conv and FC) is evaluated using the EMVP query-only-
privacy protocol from:

  "Encrypted Matrix-Vector Products from Secret Dual Codes"
  Benhamouda, Chen, Halevi, Ishai, Krawczyk, Mour, Rabin, Rosen
  CCS 2025 / ePrint 2025/858

Protocol recap (query-only privacy)
-------------------------------------
  Key:        sk = (C1 ∈ F^{k×ℓ},  alphas ∈ (F*)^s)
  Encoding:   M_hat = M @ D,  D = [-C1^T | I_ℓ] ∈ F^{ℓ×n},  n = k+ℓ
  Encrypt q:  q_tilde = c + [0_k | q],  c = a^T C  (random primal codeword)
              q_hat   = (α_1 q_tilde_1 | … | α_s q_tilde_s)
  Answer:     M' = [M_hat_j @ q_hat_j]  ∈ F^{m×s}
  Decode:     Mq = M' @ p',  p' = (1/α_1, …, 1/α_s)

All arithmetic is over F_p (a 61-bit Mersenne prime). Floating-point weights
are quantised to fixed-point integers before the protocol runs, then rescaled
after decoding.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 0.  Field arithmetic
# ---------------------------------------------------------------------------

P: int = (1 << 61) - 1  # 61-bit Mersenne prime


def mod(x) -> np.ndarray:
    return (np.asarray(x).astype(object) % P).astype(np.int64)


def matmul_mod(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.mod(A.astype(object) @ B.astype(object), P).astype(np.int64)


def inv_mod(a: int) -> int:
    return pow(int(a), P - 2, P)


def inv_vec(v: np.ndarray) -> np.ndarray:
    return np.array([inv_mod(int(x)) for x in v], dtype=np.int64)


def to_signed(x: np.ndarray) -> np.ndarray:
    """Map F_P elements back to signed integers in (-P/2, P/2]."""
    x = x % P
    return np.where(x > P // 2, x.astype(object) - P, x.astype(object)).astype(np.int64)


# ---------------------------------------------------------------------------
# 1.  Fixed-point quantisation
# ---------------------------------------------------------------------------

SCALE: int = 1 << 10  # 2^10 = 1024 (10 fractional bits)


def quantise(x: np.ndarray) -> np.ndarray:
    """Float → int64, mapped into F_P."""
    q = np.round(x * SCALE).astype(np.int64)
    return (q.astype(object) % P).astype(np.int64)


def dequantise(x: np.ndarray) -> np.ndarray:
    """Signed int64 → float, removing both SCALE factors (weights × activations)."""
    return to_signed(x).astype(np.float64) / (SCALE * SCALE)


# ---------------------------------------------------------------------------
# 2.  EMVP core
# ---------------------------------------------------------------------------

@dataclass
class EMVPKey:
    C1:     np.ndarray  # (k, ell)
    alphas: np.ndarray  # (s,)
    k:      int
    s:      int
    n:      int         # k + ell
    ell:    int


@dataclass
class QueryMessage:
    q_hat: np.ndarray   # (n,)


@dataclass
class ResponseMessage:
    M_prime: np.ndarray  # (m, s)


@dataclass
class DecodingKey:
    p_prime: np.ndarray  # (s,) = [1/alpha_j]


def emvp_keygen(ell: int, k: int, s: int, rng: np.random.Generator) -> EMVPKey:
    n = k + ell
    assert n % s == 0, f"n={n} must be divisible by s={s}"
    C1     = rng.integers(0, P, size=(k, ell),  dtype=np.int64)
    alphas = rng.integers(1, P, size=(s,),       dtype=np.int64)
    return EMVPKey(C1=C1, alphas=alphas, k=k, s=s, n=n, ell=ell)


def emvp_encode_matrix(M_int: np.ndarray, key: EMVPKey) -> np.ndarray:
    """M_hat = M @ D  where D = [-C1^T | I_ell].  Returns (m, n)."""
    neg_C1T = mod(-key.C1.T)                    # (ell, k)
    I_ell   = np.eye(key.ell, dtype=np.int64)
    D       = np.hstack([neg_C1T, I_ell])       # (ell, n)
    return matmul_mod(M_int, D)                 # (m, n)


def emvp_encrypt_query(
    q_int: np.ndarray, key: EMVPKey, rng: np.random.Generator
) -> Tuple[QueryMessage, DecodingKey]:
    k, ell, n, s = key.k, key.ell, key.n, key.s
    block_len = n // s

    a       = rng.integers(0, P, size=(k,), dtype=np.int64)
    c_left  = mod(a)
    c_right = matmul_mod(a.reshape(1, -1), key.C1).flatten()
    c       = np.concatenate([c_left, c_right])

    q_tilde = mod(c + np.concatenate([np.zeros(k, dtype=np.int64), q_int]))

    blocks  = q_tilde.reshape(s, block_len)
    scaled  = (blocks.astype(object) * key.alphas.reshape(s, 1).astype(object) % P).astype(np.int64)
    q_hat   = scaled.flatten()

    return QueryMessage(q_hat=q_hat), DecodingKey(p_prime=inv_vec(key.alphas))


def emvp_server_answer(M_hat: np.ndarray, msg: QueryMessage, s: int) -> ResponseMessage:
    m, n      = M_hat.shape
    block_len = n // s
    M_prime   = np.zeros((m, s), dtype=np.int64)
    for j in range(s):
        c0, c1        = j * block_len, (j + 1) * block_len
        M_prime[:, j] = matmul_mod(M_hat[:, c0:c1], msg.q_hat[c0:c1].reshape(-1, 1)).flatten()
    return ResponseMessage(M_prime=M_prime)


def emvp_decode(resp: ResponseMessage, dk: DecodingKey) -> np.ndarray:
    return matmul_mod(resp.M_prime, dk.p_prime.reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# 3.  EMVP layer wrapper
# ---------------------------------------------------------------------------

class EMVPLayer:
    """
    Wraps a weight matrix so the server stores M_hat and answers encrypted
    queries; the client decodes.  Both parties are simulated in one process.
    """

    def __init__(
        self, M: np.ndarray, name: str,
        k: int = 16, s: int = 4,
        rng: Optional[np.random.Generator] = None,
    ):
        self.name         = name
        self.m, self.ell  = M.shape
        self.rng          = rng or np.random.default_rng(0)
        self._M_float     = M.copy()  # kept for plaintext reference

        # Ensure n = k+ell is divisible by s
        n = k + self.ell
        while n % s != 0:
            k += 1
            n  = k + self.ell

        self.key   = emvp_keygen(self.ell, k, s, self.rng)
        self.M_hat = emvp_encode_matrix(quantise(M), self.key)

    # -- client ---------------------------------------------------------------

    def client_encrypt(self, activation: np.ndarray) -> Tuple[QueryMessage, DecodingKey]:
        return emvp_encrypt_query(quantise(activation), self.key, self.rng)

    def client_decode(self, resp: ResponseMessage, dk: DecodingKey) -> np.ndarray:
        return dequantise(emvp_decode(resp, dk))

    # -- server ---------------------------------------------------------------

    def server_answer(self, msg: QueryMessage) -> ResponseMessage:
        return emvp_server_answer(self.M_hat, msg, self.key.s)

    # -- combined (both parties in one process) --------------------------------

    def forward(self, activation: np.ndarray) -> np.ndarray:
        msg, dk = self.client_encrypt(activation)
        resp    = self.server_answer(msg)
        return self.client_decode(resp, dk)

    def plaintext_forward(self, activation: np.ndarray) -> np.ndarray:
        return self._M_float @ activation


# ---------------------------------------------------------------------------
# 4.  im2col: convolution → matrix multiply
# ---------------------------------------------------------------------------

def im2col(x: np.ndarray, kH: int, kW: int,
           stride: int = 1, pad: int = 1) -> np.ndarray:
    """
    x : (C, H, W)  →  col : (kH*kW*C, out_H*out_W)
    Each column is one receptive-field patch, flattened.
    """
    C, H, W = x.shape
    x_pad   = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    out_H   = (H + 2 * pad - kH) // stride + 1
    out_W   = (W + 2 * pad - kW) // stride + 1

    col = np.zeros((kH * kW * C, out_H * out_W), dtype=x.dtype)
    idx = 0
    for i in range(out_H):
        for j in range(out_W):
            patch       = x_pad[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
            col[:, idx] = patch.flatten()
            idx        += 1
    return col


# ---------------------------------------------------------------------------
# 5.  ResNet building blocks
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def batch_norm_fold(
    W: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
    mean: np.ndarray, var: np.ndarray, eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    std    = np.sqrt(var + eps)
    scale  = gamma / std
    W_fold = W * scale[:, None, None, None]
    b_fold = beta - mean * scale
    return W_fold, b_fold


def extract_conv_bn_weights(conv, bn) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold BatchNorm into Conv2d weights from PyTorch layers.
    Returns (W_fold, b_fold) ready to pass as the `weights` arg to ConvBNReLU.
    """
    W     = conv.weight.detach().cpu().float().numpy()
    gamma = bn.weight.detach().cpu().float().numpy()
    beta  = bn.bias.detach().cpu().float().numpy()
    mean  = bn.running_mean.detach().cpu().float().numpy()
    var   = bn.running_var.detach().cpu().float().numpy()
    return batch_norm_fold(W, gamma, beta, mean, var)


class ConvBNReLU:
    """Conv2d → BatchNorm (folded) → ReLU, with the conv run via EMVP."""

    def __init__(
        self, C_in: int, C_out: int,
        kH: int = 3, kW: int = 3,
        stride: int = 1, pad: int = 1,
        name: str = "conv",
        k: int = 32, s: int = 4,
        rng: Optional[np.random.Generator] = None,
        weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.C_out  = C_out
        self.kH, self.kW = kH, kW
        self.stride = stride
        self.pad    = pad
        rng = rng or np.random.default_rng(0)

        if weights is not None:
            self.W_fold = np.asarray(weights[0], dtype=np.float32)
            self.b_fold = np.asarray(weights[1], dtype=np.float32)
        else:
            fan_in = C_in * kH * kW
            W      = rng.standard_normal((C_out, C_in, kH, kW)).astype(np.float32)
            W     *= np.sqrt(2.0 / fan_in)
            gamma  = np.ones(C_out,  dtype=np.float32)
            beta   = np.zeros(C_out, dtype=np.float32)
            mean   = rng.standard_normal(C_out).astype(np.float32) * 0.01
            var    = np.abs(rng.standard_normal(C_out).astype(np.float32)) + 0.9
            self.W_fold, self.b_fold = batch_norm_fold(W, gamma, beta, mean, var)

        W_mat      = self.W_fold.reshape(C_out, -1).astype(np.float64)
        self.emvp  = EMVPLayer(W_mat, name=name, k=k, s=s, rng=rng)

    def _out_shape(self, H: int, W: int) -> Tuple[int, int]:
        out_H = (H + 2 * self.pad - self.kH) // self.stride + 1
        out_W = (W + 2 * self.pad - self.kW) // self.stride + 1
        return out_H, out_W

    def plaintext_forward(self, x: np.ndarray) -> np.ndarray:
        C, H, W      = x.shape
        out_H, out_W = self._out_shape(H, W)
        col          = im2col(x, self.kH, self.kW, self.stride, self.pad)
        W_mat        = self.W_fold.reshape(self.C_out, -1)
        out          = (W_mat @ col + self.b_fold[:, None]).reshape(self.C_out, out_H, out_W)
        return relu(out.astype(np.float32))

    def emvp_forward(self, x: np.ndarray) -> np.ndarray:
        C, H, W      = x.shape
        out_H, out_W = self._out_shape(H, W)
        col          = im2col(x.astype(np.float64), self.kH, self.kW, self.stride, self.pad)
        n_patches    = col.shape[1]

        out_flat = np.zeros((self.C_out, n_patches), dtype=np.float64)
        for j in range(n_patches):
            out_flat[:, j] = self.emvp.forward(col[:, j])

        out = (out_flat + self.b_fold[:, None]).reshape(self.C_out, out_H, out_W)
        return relu(out.astype(np.float32))


class ResidualBlock:
    """Two ConvBNReLU layers with an optional 1×1 shortcut."""

    def __init__(
        self, C_in: int, C_out: int,
        stride: int = 1, name: str = "resblock",
        k: int = 32, s: int = 4,
        rng: Optional[np.random.Generator] = None,
        weights: Optional[dict] = None,
    ):
        w   = weights or {}
        rng = rng or np.random.default_rng(0)
        self.conv1 = ConvBNReLU(C_in,  C_out, stride=stride, name=f"{name}.conv1",
                                k=k, s=s, rng=rng, weights=w.get("conv1"))
        self.conv2 = ConvBNReLU(C_out, C_out, stride=1,      name=f"{name}.conv2",
                                k=k, s=s, rng=rng, weights=w.get("conv2"))

        self.shortcut: Optional[ConvBNReLU] = None
        if C_in != C_out or stride != 1:
            self.shortcut = ConvBNReLU(
                C_in, C_out, kH=1, kW=1, stride=stride, pad=0,
                name=f"{name}.shortcut", k=k, s=s, rng=rng, weights=w.get("shortcut"),
            )

    def plaintext_forward(self, x: np.ndarray) -> np.ndarray:
        residual = x
        out      = self.conv1.plaintext_forward(x)
        out      = self.conv2.plaintext_forward(out)
        if self.shortcut:
            residual = self.shortcut.plaintext_forward(x)
        return relu(out + residual)

    def emvp_forward(self, x: np.ndarray) -> np.ndarray:
        residual = x
        out      = self.conv1.emvp_forward(x)
        out      = self.conv2.emvp_forward(out)
        if self.shortcut:
            residual = self.shortcut.emvp_forward(x)
        return relu(out + residual)


class GlobalAveragePool:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=(1, 2))  # (C, H, W) → (C,)


class LinearClassifier:
    def __init__(
        self, C_in: int, num_classes: int,
        name: str = "classifier",
        k: int = 16, s: int = 4,
        rng: Optional[np.random.Generator] = None,
        weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        rng = rng or np.random.default_rng(0)
        if weights is not None:
            W      = np.asarray(weights[0], dtype=np.float64)
            self.b = np.asarray(weights[1], dtype=np.float64)
        else:
            W      = rng.standard_normal((num_classes, C_in)).astype(np.float64)
            W     *= np.sqrt(2.0 / C_in)
            self.b = np.zeros(num_classes, dtype=np.float64)
        self.W    = W
        self.emvp = EMVPLayer(W, name=name, k=k, s=s, rng=rng)

    def plaintext_forward(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b

    def emvp_forward(self, x: np.ndarray) -> np.ndarray:
        return self.emvp.forward(x.astype(np.float64)) + self.b


# ---------------------------------------------------------------------------
# 6.  ResNet
# ---------------------------------------------------------------------------

class ResNet:
    """
    Small ResNet for image classification (CIFAR-10 or MNIST).

    Architecture (ResNet-20 style):
      conv1  : C_in → 16, 3×3
      stage1 : n_blocks × ResBlock(16 → 16)
      stage2 : n_blocks × ResBlock(16 → 32, stride=2 on first)
      stage3 : n_blocks × ResBlock(32 → 64, stride=2 on first)
      GAP    : GlobalAveragePool
      fc     : 64 → num_classes
    """

    def __init__(
        self,
        n_blocks: int = 1,
        num_classes: int = 10,
        C_in: int = 3,
        k: int = 16,
        s: int = 4,
        seed: int = 42,
        weights: Optional[dict] = None,
    ):
        w   = weights or {}
        rng = np.random.default_rng(seed)
        kw  = dict(k=k, s=s, rng=rng)

        self.conv1  = ConvBNReLU(C_in, 16, name="conv1", weights=w.get("conv1"), **kw)

        self.stage1 = [
            ResidualBlock(16, 16, name=f"s1b{i}",
                          weights={"conv1": w.get(f"s1b{i}.c1"), "conv2": w.get(f"s1b{i}.c2"),
                                   "shortcut": w.get(f"s1b{i}.sc")}, **kw)
            for i in range(n_blocks)
        ]
        self.stage2 = [
            ResidualBlock(16 if i == 0 else 32, 32, stride=2 if i == 0 else 1, name=f"s2b{i}",
                          weights={"conv1": w.get(f"s2b{i}.c1"), "conv2": w.get(f"s2b{i}.c2"),
                                   "shortcut": w.get(f"s2b{i}.sc")}, **kw)
            for i in range(n_blocks)
        ]
        self.stage3 = [
            ResidualBlock(32 if i == 0 else 64, 64, stride=2 if i == 0 else 1, name=f"s3b{i}",
                          weights={"conv1": w.get(f"s3b{i}.c1"), "conv2": w.get(f"s3b{i}.c2"),
                                   "shortcut": w.get(f"s3b{i}.sc")}, **kw)
            for i in range(n_blocks)
        ]
        self.gap = GlobalAveragePool()
        self.fc  = LinearClassifier(64, num_classes, name="fc", weights=w.get("fc"), **kw)

    def _run(self, x: np.ndarray, mode: str) -> np.ndarray:
        fwd = (lambda layer, inp:
               layer.emvp_forward(inp) if mode == "emvp" else layer.plaintext_forward(inp))
        out = fwd(self.conv1, x)
        for block in self.stage1:
            out = fwd(block, out)
        for block in self.stage2:
            out = fwd(block, out)
        for block in self.stage3:
            out = fwd(block, out)
        out = self.gap.forward(out)
        out = fwd(self.fc, out)
        return out

    def plaintext_forward(self, x: np.ndarray) -> np.ndarray:
        return self._run(x, mode="plain")

    def emvp_forward(self, x: np.ndarray) -> np.ndarray:
        return self._run(x, mode="emvp")


# ---------------------------------------------------------------------------
# 7.  Utilities
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# 8.  Smoke-test demo
# ---------------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("EMVP-Protected ResNet — Smoke Test (random weights)")
    print("=" * 65)
    print(f"  Field prime  : 2^61 - 1")
    print(f"  Quant scale  : SCALE = 2^10 = {SCALE}")
    print()

    rng   = np.random.default_rng(7)
    model = ResNet(n_blocks=1, num_classes=10, C_in=3, k=16, s=4, seed=42)

    x = rng.standard_normal((3, 32, 32)).astype(np.float32)
    print(f"  Input shape  : {x.shape}  (CIFAR-10 style)")
    print()

    print("Running plaintext forward pass …", flush=True)
    logits_plain = model.plaintext_forward(x)
    probs_plain  = softmax(logits_plain)
    print(f"  Logits : {logits_plain.round(4)}")
    print(f"  Probs  : {probs_plain.round(4)}")
    print(f"  Pred   : class {np.argmax(probs_plain)}")
    print()

    print("Running EMVP forward pass …", flush=True)
    logits_emvp = model.emvp_forward(x)
    probs_emvp  = softmax(logits_emvp)
    print(f"  Logits : {logits_emvp.round(4)}")
    print(f"  Probs  : {probs_emvp.round(4)}")
    print(f"  Pred   : class {np.argmax(probs_emvp)}")
    print()

    l2_err = np.linalg.norm(logits_plain - logits_emvp)
    print(f"  L2 error in logits      : {l2_err:.6f}")
    print(f"  Max abs error in logits : {np.abs(logits_plain - logits_emvp).max():.6f}")
    print(f"  Predictions match       : {np.argmax(logits_plain) == np.argmax(logits_emvp)}")
    print("=" * 65)


if __name__ == "__main__":
    demo()
