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

// ── Softmax ──
//
// Used in attention to turn raw dot-product scores into weights that sum to 1.
// Higher scores get exponentially more weight.

void softmax(float *out, const float *x, int n) {

    // Step 1: find the maximum value.
    // We'll subtract this from every element before calling exp().
    // This prevents overflow: exp(800) = infinity, but exp(800 - 800) = exp(0) = 1.
    // Mathematically: exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x)) — the c cancels.
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Step 2: compute exp(x[i] - max) for each element and accumulate the sum.
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    // Step 3: normalize by dividing by the sum.
    // Now every out[i] is in (0, 1] and all out[i] sum to exactly 1.0.
    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

// ── SiLU ──
//
// Used in the FFN as part of SwiGLU: FFN(x) = silu(gate) * up, then down.
// The gate vector controls which information flows through.

void silu(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // silu(x)    = x * sigmoid(x)
        //
        // When x is large positive: sigmoid → 1, so silu(x) ≈ x (pass through)
        // When x is large negative: sigmoid → 0, so silu(x) ≈ 0 (gate closed)
        // The transition is smooth, unlike ReLU's hard zero.
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        out[i] = x[i] * sigmoid;
    }
}

// ── RoPE ──
//
// We apply this after the Q and K projections, before attention.
// Q and K are both modified in-place. V is left unchanged — positions
// are only needed when computing similarity scores (dot products of Q and K).

void rope(float *q, float *k, int pos, int n_heads, int n_kv_heads,
          int head_dim, int rope_dim_count, float freq_base) {

    // Process each head independently.
    // Q has n_heads heads, K has n_kv_heads heads (fewer, due to GQA).
    // Both use the same rotation angles — only the number of heads differs.
    int n_heads_max = n_heads > n_kv_heads ? n_heads : n_kv_heads;

    for (int h = 0; h < n_heads_max; h++) {
        // pointer to the start of this head's slice
        float *qh = q + h * head_dim;                   // always valid for h < n_heads
        float *kh = k + h * head_dim;                   // only valid for h < n_kv_heads

        // only rotate rope_dim_count dims — remaining dims stay unchanged (partial RoPE)
        for (int d = 0; d < rope_dim_count; d += 2) {
            int i = d / 2;  // pair index: 0, 1, 2, ... head_dim/2 - 1

            // θᵢ = pos / (freq_base ^ (2i / head_dim))
            //
            // freq_base is 10000 for LLaMA. The exponent maps pair index to a
            // frequency: pair 0 rotates fastest (large angle per step),
            // pair head_dim/2-1 rotates slowest (tiny angle per step).
            // This gives the model sensitivity across many positional scales.
            float theta = (float)pos / powf(freq_base, (float)(2 * i) / head_dim);

            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            // Apply 2D rotation to the pair (q[d], q[d+1]):
            //   [cos  -sin] [q[d]  ]
            //   [sin   cos] [q[d+1]]
            //
            // We save q[d] first because we overwrite it before reading it again.
            if (h < n_heads) {
                float q0 = qh[d];
                float q1 = qh[d + 1];
                qh[d]     = q0 * cos_theta - q1 * sin_theta;
                qh[d + 1] = q0 * sin_theta + q1 * cos_theta;
            }
            if (h < n_kv_heads) {
                float k0 = kh[d];
                float k1 = kh[d + 1];
                kh[d]     = k0 * cos_theta - k1 * sin_theta;
                kh[d + 1] = k0 * sin_theta + k1 * cos_theta;
            }
        }
    }
}
