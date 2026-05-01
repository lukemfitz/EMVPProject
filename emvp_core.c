/*
 * emvp_core.c — Fast F_p arithmetic for the EMVP protocol.
 *
 * Prime: P = 2^61 - 1 (61-bit Mersenne prime).
 *
 * Key trick: mulmod uses unsigned __int128 to hold the full 122-bit product,
 * then reduces via the Mersenne identity:
 *   a*b = hi*2^61 + lo  ≡  hi + lo  (mod 2^61-1)
 * One conditional subtraction finalises the reduction.
 *
 * Build (macOS / Linux):
 *   cc -O3 -march=native -shared -fPIC -o emvp_core.so emvp_core.c
 */

#include <stdint.h>

#define P ((uint64_t)((1ULL << 61) - 1))

static inline uint64_t addmod(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    if (r >= P) r -= P;
    return r;
}

static inline uint64_t mulmod(uint64_t a, uint64_t b) {
    unsigned __int128 prod = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)(prod & P);
    uint64_t hi = (uint64_t)(prod >> 61);
    uint64_t r  = lo + hi;
    if (r >= P) r -= P;
    return r;
}

/*
 * matmul_mod — Matrix-matrix product mod P.
 *
 * A: (m, k) row-major int64, values in [0, P-1]
 * B: (k, n) row-major int64, values in [0, P-1]
 * C: (m, n) row-major int64 output
 */
void matmul_mod(const int64_t *A, const int64_t *B, int64_t *C,
                int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            uint64_t acc = 0;
            for (int l = 0; l < k; l++)
                acc = addmod(acc, mulmod((uint64_t)A[i*k + l],
                                         (uint64_t)B[l*n + j]));
            C[i*n + j] = (int64_t)acc;
        }
    }
}

/*
 * scale_blocks_mod — Multiply each row i of `blocks` by scalar `alphas[i]`, mod P.
 *
 * blocks: (s, block_len) row-major int64, values in [0, P-1]
 * alphas: (s,)           int64,           values in [0, P-1]
 * result: (s, block_len) row-major int64 output
 */
void scale_blocks_mod(const int64_t *blocks, const int64_t *alphas,
                      int64_t *result, int s, int block_len) {
    for (int i = 0; i < s; i++) {
        uint64_t alpha = (uint64_t)alphas[i];
        for (int j = 0; j < block_len; j++)
            result[i*block_len + j] =
                (int64_t)mulmod((uint64_t)blocks[i*block_len + j], alpha);
    }
}

/*
 * scalecol_mod — Multiply every element of a 2-D matrix by a single scalar, mod P.
 * Used for batched block masking: scale an entire (block_len × p) sub-matrix
 * by one alpha value at once.
 *
 * mat:    (rows, cols) row-major int64, values in [0, P-1]
 * scalar: int64,                        value  in [0, P-1]
 * result: (rows, cols) row-major int64 output
 */
void scalecol_mod(const int64_t *mat, int64_t scalar,
                  int64_t *result, int rows, int cols) {
    uint64_t alpha = (uint64_t)scalar;
    int n = rows * cols;
    for (int i = 0; i < n; i++)
        result[i] = (int64_t)mulmod((uint64_t)mat[i], alpha);
}
