# PLAN — Per-Layer Checksum Verification for EMVP

## Goal

Detect a **maliciously cheating server** that returns wrong matrix-vector
products. The current EMVP protocol assumes an honest-but-curious server: it
follows the protocol but tries to learn `q`. We extend it so the client can
also verify, with overwhelming probability, that the result it received is
actually `M @ q` and not some tampered value.

The user's framing: *"the client adds a little extra value to every query it
sends, which is a checksum that it then verifies when it receives a matrix
back and decrypts."*

## Protocol

Per layer, per query, the client performs a Freivalds-style spot check
embedded into the EMVP round-trip:

1. Client samples a uniformly random vector **r ∈ F_p^m** (the "checksum
   challenge"; one fresh value per query).
2. Client locally computes the expected checksum
   `c_expected = (r^T @ M_int) @ Q_int  mod P  ∈ F_p^p`
   using its plaintext copy of `M_int` (public) and `Q_int` (the client's
   own quantised activation, known in clear).
3. Client encrypts `q` to `q_hat` as in the existing protocol; sends
   `(q_hat, r)` to the server.
4. Server runs the unmodified EMVP answer step and returns `M'`.
5. Client decodes `M'` to `m_q_int = M_int @ Q_int  mod P`.
6. Client computes the received checksum `c_received = r^T @ m_q_int  mod P`.
7. Client accepts iff `c_received == c_expected`.

### Why this is sound

If the server is honest, `m_q_int = M_int @ Q_int` and the check is an
identity. If the server tampers with the response so that
`m_q_int_tampered = M_int @ Q_int + δ` with `δ ≠ 0`, then
`c_received - c_expected = r^T @ δ`. For any fixed nonzero `δ`, a uniformly
random `r ∈ F_p^m` is orthogonal to `δ` with probability `1 / P ≈ 2^-61`.
So a cheating server fools the check with negligible probability per layer.

### Why "extra value in the query"

`r` is conceptually transmitted along with `q_hat`. Functionally the
verification is fully client-side (the server never has to act on `r`),
but the framing matches the user's intent: each query carries an extra
nonce that pins down the verification.

### Why F_p, not floats

Working in F_p means the check is **exact** — no tolerance, no false
positives from quantisation rounding. We re-use the existing
`matmul_mod` C routine for both the local pre-computation and the
verification combine.

## Implementation Sketch

Touch points in `emvp_resnet.py`:

| Where | Change |
|---|---|
| `EMVPLayer.__init__` | Cache `self.M_int = quantise(M)` (small extra storage; avoids re-quantising per query) |
| `EMVPLayer` | Add `forward_with_verify(activation)` (single query) and `forward_batched_with_verify(act_matrix)` (batched) — return `(result, passed: bool)` |
| `EMVPLayer` | Add `_simulate_cheat(M_blocks)` hook controlled by a flag for benchmarking |
| `ConvBNReLU`, `LinearClassifier`, `ResidualBlock`, `ResNet` | Mirror methods: `emvp_forward_with_verify(x)` returning `(result, all_layers_passed: bool)` |

Cheat simulation: a class-level flag `EMVPLayer.cheat_mode` set to
`"none"` or `"flip_one"`. When enabled, the server adds 1 (mod P) to a
random entry of `M_blocks` before returning. This is the smallest possible
in-field cheat; if the check catches that, it catches anything bigger.

## Benchmark

New file `benchmark_checksum.py` runs four conditions on CIFAR-10 and
MNIST and reports timing + accuracy + detection rate:

| Condition | What it measures |
|---|---|
| **Plaintext** | Baseline for accuracy and speed |
| **EMVP (no checksum)** | Existing protocol baseline |
| **EMVP + checksum, honest server** | Overhead added by verification; should preserve accuracy |
| **EMVP + checksum, cheating server** | Detection rate (expected ≈ 100%) |

Per-image metrics: `plaintext_pred`, `emvp_pred`, `verify_passed`,
per-layer pass count, latency.

Aggregate metrics:
- Accuracy under each condition (cheating-server condition reports
  rejection rate, since accepted-result accuracy is undefined when
  results are tampered)
- Mean per-image latency for each condition; checksum-overhead %
- Detection rate = fraction of cheating-server inferences caught at any
  layer

Smaller image counts than `benchmark.py` (e.g., 100 images) so the run
finishes quickly while still being statistically meaningful — even 10
images give 100% detection rate at 2^-61 soundness per layer.

## Out of Scope

- Multi-round / interactive verification (Schnorr-style)
- Hiding `r` from the server (not needed for soundness in this design)
- Verification batching across queries with shared `r` (would need a
  per-batch nonce; possible follow-up)
- Changes to the `M`-privacy variant (TDM); this targets only the
  query-only-privacy variant currently implemented
