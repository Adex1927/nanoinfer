#include "ops.h"
#include <cmath>

// ── RMSNorm ──
//
// This is the very first computation in each transformer layer.
// It normalizes the input vector so that its root-mean-square is ~1.0,
// then scales each element by a learned weight.
//
// Why normalize? Without it, activations can grow or shrink as they pass
// through layers, making training unstable. Normalization keeps values
// in a healthy range.
//
// Why RMSNorm instead of LayerNorm? RMSNorm skips the mean-subtraction
// step that LayerNorm does. It's simpler, faster, and works just as well
// for LLaMA-family models.

void rmsnorm(float *out, const float *x, const float *weight, int n, float eps) {

    // Step 1: compute sum of squares
    //   ss = x[0]² + x[1]² + ... + x[n-1]²
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }

    // Step 2: compute mean of squares
    //   ss = ss / n
    // This gives us the average squared magnitude of the vector.
    ss /= n;

    // Step 3: compute the normalization factor
    //   ss = 1.0 / sqrt(ss + eps)
    //
    // We add eps (1e-5) to prevent division by zero if the vector
    // happens to be all zeros. Then we take 1/sqrt so we can multiply
    // instead of divide in the next step (multiplication is faster).
    ss = 1.0f / sqrtf(ss + eps);

    // Step 4: normalize and scale
    //   out[i] = x[i] * ss * weight[i]
    //
    // - x[i] * ss normalizes the value (makes RMS ≈ 1.0)
    // - * weight[i] applies the learned per-element scale
    //   (the model learned which dimensions should be amplified or dampened)
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

// ── Matrix-Vector Multiply ──
//
// The simplest possible implementation: two nested loops.
//
// For each output row i:
//   out[i] = W[i][0]*x[0] + W[i][1]*x[1] + ... + W[i][n_in-1]*x[n_in-1]
//
// This is O(n_out × n_in). For attn_q in TinyLlama that's 2048 × 2048 = ~4M
// multiply-adds. Naive but correct — optimization comes later.

void matvec(float *out, const float *W, const float *x, int n_out, int n_in) {
    for (int i = 0; i < n_out; i++) {
        // start of row i in the weight matrix
        const float *row = W + i * n_in;

        // dot product of this row with input vector x
        float sum = 0.0f;
        for (int j = 0; j < n_in; j++) {
            sum += row[j] * x[j];
        }

        out[i] = sum;
    }
}
