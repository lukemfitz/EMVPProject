# Threat Model — EMVP-Protected ResNet with Checksum Verification

This document specifies the security setting under which the protocol in
this repository is intended to operate. It is the reference document for
"what does the protocol promise, against whom, and at what cost."

The original EMVP paper (Benhamouda et al., CCS 2025 / ePrint 2025/858)
states an honest-but-curious threat model. This implementation extends
that to a malicious server by adding a per-layer Freivalds-style checksum
(see [VERIFY_PLAN.md](VERIFY_PLAN.md), [VERIFY_ACTIONS.md](VERIFY_ACTIONS.md),
and `_verify_freivalds` in [src/emvp_resnet.py](../src/emvp_resnet.py)).

---

## 1. Setting

A **client** holds a private input image `q` and wants the inference
result of a public ResNet on `q`. A **server** holds a copy of the
public ResNet weights and runs inference on the client's behalf. The
client communicates with the server over an authenticated, integrity-
protected network channel (out of scope below).

Each linear layer of the ResNet is computed via the EMVP protocol; the
client encrypts the activation, the server returns an encoded response,
and the client decodes. Per-layer output is then passed through ReLU /
LayerNorm / etc. on the client side and the next layer's encryption
begins. After the final layer the client recovers logits.

---

## 2. Security goals

### G1. Query privacy

For every inference, the server learns nothing about `q` beyond what is
implied by the public model `M` and the act of an inference having taken
place. Stated formally: the joint view of all messages the server
receives across one or more inferences is computationally indistinguishable
from a simulation that does not know `q`. Privacy is **computational**,
reducing to the 1D-SLSN hardness assumption stated in the EMVP paper.

### G2. Result integrity

For every inference the client accepts, the recovered value is M·q, with
overwhelming probability over the client's verification randomness. If
the server returns any tampered value, the client rejects with probability
≥ 1 − ε for some statistical bound ε. Integrity is **statistical**,
holding unconditionally (no cryptographic assumption needed) given that
the client samples its verification challenge `r` uniformly at random.

These are independent: the protocol may reject (failing G2) without the
server having learned anything about `q` (G1 still holds).

---

## 3. Adversary capabilities

The adversary controls the server. It may:

- **Read everything it receives.** This includes `M_hat` at setup,
  `q_hat` for every query, and any verification challenge `r` if the
  client transmits one (in the present implementation `r` is *not*
  transmitted; the client computes both sides of the check itself).
- **Compute arbitrarily** for any polynomial running time. This includes
  unbounded precomputation against `M_hat` between queries.
- **Deviate from the protocol arbitrarily** when computing its response.
  It may return a wrong `M'`, swap entries, return random noise, return
  a value designed to fool a particular verification challenge, refuse to
  answer, or condition its answer on `q_hat` in any way.
- **Mount adaptive attacks across many queries** under the same client
  key `(C1, α_1, …, α_s)`. The protocol assumes the key is reused across
  queries, since the per-query randomness comes from the codeword
  blinding `a` and the per-query Freivalds challenge `r`.
- **Coordinate with other parties** outside the client. The protocol
  is single-server and gives no guarantees against collusion among
  multiple parties (none are needed; there is only one server).

The adversary is *static*: it fixes its strategy at the start. A
*rushing* malicious server (one that adaptively decides what to send
after observing partial client behavior within a single query) is also
covered — within one query the client speaks first and last, so there is
no opportunity for the server to rush mid-query.

---

## 4. Adversary limitations

The protocol is sound only because of the following limitations on the
adversary. Each is a load-bearing assumption.

| Limitation | Why it holds | What breaks if violated |
|---|---|---|
| Cannot solve the 1D-SLSN problem in PPT | EMVP paper's hardness assumption | G1 (privacy) collapses |
| Cannot influence `q_hat` construction | Client-side, fresh randomness `a` per query | G1 weakened (server may steer client toward predictable codewords) |
| Cannot influence `M_hat` construction | Client computes `M_hat = M·D` from public `M` and secret `D = [−C1ᵀ \| I_ℓ]`; server only ever receives the result | G1 collapses (server could pick `M_hat` so that decoding leaks `q`) |
| Cannot predict the per-query Freivalds challenge `r` | `r` is sampled by the client from a sound RNG and never transmitted | G2 collapses (server could craft a `δ` orthogonal to a known `r`) |
| Cannot read the client's secret state | Standard isolation between client process and server | Both G1 and G2 collapse |

---

## 5. What the protocol defends against

| Attack class | Defense | Quantitative bound |
|---|---|---|
| Server attempts to recover `q` from one or many `q_hat` | EMVP encryption (1D-SLSN) | Computational: best known attack is sub-exponential; concretely > 2⁸⁰ work for the parameters used |
| Server returns a wrong `M·q` (any non-trivial tampering) | Per-layer Freivalds check in F_p | Statistical: ≤ 1/P ≈ 2⁻⁶¹ slip per layer |
| Server returns `M·q + δ` for δ designed to look "small" in floats | Same Freivalds check; soundness depends only on δ ≠ 0 mod P, not on δ's magnitude | Same: 1/P |
| Server returns `M·q + δ` for δ designed to fool a *specific* `r` | `r` is fresh per-layer, per-query, and unknown to the server | Same: 1/P |
| Cumulative tampering across the L≈16 layers of one inference | Per-layer checks ANDed together | Union bound: ≤ L/P ≈ 2⁻⁵⁷ for one full inference |
| Adaptive attack across N queries (same key) | Privacy holds under standard 1D-SLSN; integrity holds independently per query | Privacy unchanged; integrity bound is ≤ L·N/P |
| Server designs `δ` based on observed `q_hat` (not just `M`) | The server may freely depend on `q_hat`, but it does not learn `q` from `q_hat`, and `r` is independent of both | Same 1/P bound |

The integrity bound matters in absolute terms: at L = 16 layers and
P = 2⁶¹−1, one inference is rejected by an unmodified protocol with
probability less than 16 / (2⁶¹) ≈ 7 × 10⁻¹⁸. Empirically the benchmark
saw 0 missed detections out of 400 cheating-server inferences, and
0 false rejections out of 400 honest inferences.

---

## 6. What the protocol does NOT defend against

These are explicit non-goals. Knowing them is part of the threat model.

### 6.1. Denial of service

A server that returns garbage, refuses to answer, or stalls is detected
(or is trivially observable in the case of refusal), but the client
*does not get the answer*. The protocol delivers "either correct M·q or
known-bad," not "always correct M·q." If liveness is required, route
the inference to multiple servers and majority-vote or similar.

### 6.2. Selective-abort / rejection-channel leakage

A malicious server may choose whether and how to corrupt its response
based on the query it receives. If the *client's behavior on rejection*
is observable to the adversary, a single bit (accept vs. reject) per
inference may leak. Realistic concerns:

- Client retries the same `q` after rejection — server learns it has
  caused a retry, possibly correlates with `q_hat` features.
- Client retries with a freshly perturbed `q` — server learns from the
  difference.
- Client logs the failure to a third party the server can observe.

**Recommended client behavior on rejection:** abort silently, do not
retry the same query, do not fall back to plaintext, surface the
failure only to the application code (not the network).

We did not formally analyze rejection-channel leakage. It is a known
limitation of all "verify-and-abort" verifiable computation schemes.

### 6.3. Side channels

Timing, cache, branch-prediction, power, EM, microarchitectural side
channels are all out of scope. The README claims "the overhead is
fixed regardless of image content"; `scripts/timing_audit.py` exists to
test that empirically. The Freivalds checksum was implemented with
constant-shape operations and should not introduce data-dependent
timing variation, but this has not been independently audited.

### 6.4. Confidentiality of the model `M`

`M` is public in the query-only-privacy variant of EMVP. The right
block of `M_hat` is literally `M`. If you need to hide weights from
the server, you need the Trapdoored-Matrix construction from §4 of the
EMVP paper, which is not implemented here.

### 6.5. Network-level attacks

Replay, man-in-the-middle, traffic analysis, version downgrade — all
out of scope. The protocol assumes an authenticated, integrity-protected
channel beneath it (TLS or equivalent). Without that:

- A network attacker can substitute its own `M_hat` at setup → privacy
  collapses (the client would be encrypting under an adversarial code).
- A network attacker can substitute `q_hat` mid-query → just causes a
  rejection, but cumulative substitution attacks under a static key
  could leak structure. Not analyzed.

### 6.6. Compromise of client secret state

If `(C1, α_1, …, α_s)` leaks, both G1 and G2 are gone. The protocol
relies on the client storing its key in memory only, never transmitting
it, and clearing it before process exit if multiple clients share
hardware.

### 6.7. The setup phase

The current implementation generates `M_hat` on the client and "stores"
it on the server (in the same process). A real deployment uploads
`M_hat` once at setup. Nothing in this protocol authenticates `M_hat`
on the server side. If the server tampers with its stored `M_hat`
between setup and queries, every subsequent inference will be wrong —
and our checksum *will* detect it (since the check is against
plaintext `M`, not `M_hat`). But the failure is binary: rejection,
not recovery.

### 6.8. Compromise of the verification RNG

If the client's RNG is predictable, the server can choose `δ`
orthogonal to the predicted `r` and slip past the check. This is the
standard cryptographic-RNG assumption.

---

## 7. Trust assumptions, summarized

The client must trust that:

1. The 1D-SLSN problem is hard for the adversary's running time and
   parameter regime.
2. Its own random number generator is cryptographically sound (used for
   key generation, codeword randomness `a`, and verification challenge
   `r`).
3. Its own secret key `(C1, α_1, …, α_s)` is not exfiltrated.
4. The network channel between client and server provides authenticity
   and integrity (TLS or equivalent).
5. The client's local code is not compromised (it is, after all,
   computing the verification check itself).

The client need NOT trust that:

- The server runs the protocol correctly. (The checksum catches any
  deviation that affects the result.)
- The server stores `M_hat` faithfully across queries. (Same.)
- The server is unable to read `q_hat`. (It can; encryption is the
  point.)
- The server is unable to read `r`. (In our implementation `r` is
  never sent. Even if it were, the check would still be sound.)

---

## 8. Where this fits in the verifiable-computation landscape

In standard terminology this is a:

- **Single-server**, **designated-verifier**, **single-prover**
  protocol — there is no third-party verifier, no proof transcript that
  can be checked offline by anyone other than the client.
- **Public-key-free for verification** — both checks (privacy and
  integrity) are symmetric-key in the sense that they use only the
  client's local secret state.
- **Computational privacy** (1D-SLSN) plus **statistical integrity**
  (1/P per layer).
- **Active / malicious soundness** in the standard MPC sense — the
  server may deviate from the protocol arbitrarily and is still caught
  with overwhelming probability.

Closely related primitives in the literature:

- **Freivalds 1977** for the matrix-product check itself.
- **Algorithm-Based Fault Tolerance** (Huang & Abraham 1984) for the
  ABFT view of the same idea.
- **Verifiable Delegated Computation** (Gennaro-Gentry-Parno 2010,
  Pinocchio 2013, Groth16 2016) — much heavier machinery aimed at
  *general* computation; our setting is restricted to repeated linear
  layers, which is why a Freivalds check suffices.
- **Goldwasser-Kalai-Rothblum 2008** "interactive proofs for muggles"
  and **Thaler 2013** for the more general sumcheck-based extensions.
- **zkML** systems (zkCNN, ZEN, zkLLM) — verify ML inference under
  zero-knowledge; our protocol gives input privacy without a
  zero-knowledge proof, by leveraging the linear structure plus a
  symmetric-key verifier.

What this protocol does *not* aim to be:

- A general-purpose zero-knowledge proof system.
- A multi-party computation protocol (no multiple distrusting servers).
- A fully-homomorphic-encryption inference scheme (we do not encrypt the
  weights, and we do not support arbitrary circuits over ciphertext).

---

## 9. Open questions / known weaknesses

Areas where the threat model is weak or unverified, in roughly
decreasing order of importance:

1. **Rejection-channel leakage** (§6.2). No formal bound. The safe
   client behavior is documented; a paranoid deployment should run
   inferences in batches and treat any rejection as a session-level
   abort.
2. **Side-channel resistance** (§6.3). Empirically not measured for
   the verification path. The base EMVP path has a small audit script
   (`scripts/timing_audit.py`); the verification path does not yet.
3. **Concrete LSN parameters.** The security parameter `k` defaults to
   16 in the benchmarks; this is small relative to the paper's
   recommended sub-exponential regime. For real deployment, `k` should
   be sized against the best known LSN attack at the desired security
   level.
4. **No M-privacy.** As stated, weights are public. A deployment in
   which the model itself is sensitive (e.g., a proprietary classifier)
   needs the TDM extension.
5. **Single-server assumption.** All eggs in one basket for liveness.
6. **No formal proof** of the malicious-server reduction in this
   write-up. The argument is the standard Freivalds soundness +
   independence of `r` from `δ`, but no machine-checked or paper-style
   proof has been written.

---

## 10. One-paragraph summary for a non-cryptographer

You are sending an image to a server to be classified. The server sees
only an encrypted form of the image and can never recover the original
from what you send (this is the existing EMVP guarantee, based on a
standard hardness assumption). On top of that, every classification
result the server returns is checked against a small random
fingerprint that you compute locally; if the server returned anything
other than the true output of the model, your check fails with
probability greater than 1 − 2⁻⁵⁷ per inference and you reject the
result. You do not learn what the server tried to do, only that it
tried — so you should drop the connection and not retry. The
guarantees do not cover liveness (the server can refuse to answer), do
not cover side channels, do not hide the model itself, and depend on
your having a good random number generator and a secure network
channel.
