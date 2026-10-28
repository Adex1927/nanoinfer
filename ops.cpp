#include "ops.h"
#include "dequant.h"
#include "thread_pool.h"
#include <cmath>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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

// ── Fused Quantized Matrix-Vector Multiply (Q8_0) ──
template <DiagMode Mode>
static void matvec_q8_0_impl(float *out, const void *W, const float *x, int n_out, int n_in) {
    const BlockQ8_0 *blocks = (const BlockQ8_0 *)W;
    int blocks_per_row = n_in / 32;

    auto worker = [&](int row_start, int row_end) {
        for (int i = row_start; i < row_end; i++) {
            const BlockQ8_0 *row = blocks + i * blocks_per_row;
            float row_sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const float *x_block = x + b * 32;
                float block_sum = 0.0f;

                if constexpr (Mode == DiagMode::MEM_ONLY) {
                    for (int j = 0; j < 32; j++) {
                        block_sum += (float)row[b].quants[j];
                    }
                    row_sum += block_sum;
                } else if constexpr (Mode == DiagMode::COMPUTE_2X) {
                    for (int j = 0; j < 32; j++) {
                        float q = (float)row[b].quants[j];
                        block_sum += q * x_block[j];
                        block_sum += (q + 1.0f) * x_block[j];
                    }
                    row_sum += f16_to_f32(row[b].scale) * block_sum;
                } else {
#if defined(__ARM_NEON)
                    float32x4_t acc0 = vdupq_n_f32(0.0f);
                    float32x4_t acc1 = vdupq_n_f32(0.0f);

                    for (int j = 0; j < 32; j += 8) {
                        // Load 8 weights and widen to floats
                        int8x8_t q8 = vld1_s8(&row[b].quants[j]);
                        int16x8_t q16 = vmovl_s8(q8);
                        float32x4_t qf_low  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
                        float32x4_t qf_high = vcvtq_f32_s32(vmovl_high_s16(q16));

                        // Load 8 activations
                        float32x4_t x_low  = vld1q_f32(x_block + j);
                        float32x4_t x_high = vld1q_f32(x_block + j + 4);

                        // Fused multiply-accumulate on independent accumulators
                        acc0 = vfmaq_f32(acc0, qf_low,  x_low);
                        acc1 = vfmaq_f32(acc1, qf_high, x_high);
                    }

                    block_sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
                    for (int j = 0; j < 32; j++) {
                        block_sum += (float)row[b].quants[j] * x_block[j];
                    }
#endif
                    row_sum += f16_to_f32(row[b].scale) * block_sum;
                }
            }

            out[i] = row_sum;
        }
    };

    parallel_rows(n_out, worker);
}

void matvec_q8_0_diag(float *out, const void *W, const float *x, int n_out, int n_in, DiagMode mode) {
    switch (mode) {
        case DiagMode::MEM_ONLY:   matvec_q8_0_impl<DiagMode::MEM_ONLY>(out, W, x, n_out, n_in); break;
        case DiagMode::COMPUTE_2X: matvec_q8_0_impl<DiagMode::COMPUTE_2X>(out, W, x, n_out, n_in); break;
        default:                   matvec_q8_0_impl<DiagMode::NORMAL>(out, W, x, n_out, n_in); break;
    }
}

void matvec_q8_0(float *out, const void *W, const float *x, int n_out, int n_in) {
    matvec_q8_0_impl<DiagMode::NORMAL>(out, W, x, n_out, n_in);
}

// ── Fused Quantized Matrix-Vector Multiply (Q4_K) ──
template <DiagMode Mode>
static void matvec_q4_k_impl(float *out, const void *W, const float *x, int n_out, int n_in) {
    const BlockQ4_K *blocks = (const BlockQ4_K *)W;
    int blocks_per_row = n_in / 256;

    // Buffer for quantized activations. Sized for up to 32768 floats (zero heap allocation).
    static thread_local BlockQ8_K q8_scratch[128];
    BlockQ8_K *q8_buf = q8_scratch;
    std::vector<BlockQ8_K> q8_fallback;
    if (blocks_per_row > 128) {
        q8_fallback.resize(blocks_per_row);
        q8_buf = q8_fallback.data();
    }

    if constexpr (Mode == DiagMode::NORMAL) {
        quantize_q8_k(x, q8_buf, n_in);
    }

    auto worker = [&](int row_start, int row_end) {
        for (int i = row_start; i < row_end; i++) {
            const BlockQ4_K *row = blocks + i * blocks_per_row;
            float row_sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const BlockQ4_K *block = &row[b];
                const uint8_t *q = block->quants;

                float d    = f16_to_f32(block->d);
                float dmin = f16_to_f32(block->dmin);

                int sub_block = 0;

                if constexpr (Mode == DiagMode::MEM_ONLY) {
                    uint32_t q_sum = 0;
                    for (int l = 0; l < 32; l++) {
                        q_sum += q[l];
                    }
                    row_sum += (float)(q_sum * 2);
                } else if constexpr (Mode == DiagMode::COMPUTE_2X) {
                    const float *x_super = x + b * 256;
                    for (int chunk = 0; chunk < 256; chunk += 64) {
                        uint8_t sc1, m1_raw, sc2, m2_raw;
                        get_scale_min_k4(sub_block, block->scales, &sc1, &m1_raw);
                        get_scale_min_k4(sub_block + 1, block->scales, &sc2, &m2_raw);
                        float d1 = d * sc1, m1 = dmin * m1_raw;
                        float d2 = d * sc2, m2 = dmin * m2_raw;
                        const float *x_sub1 = x_super + chunk;
                        const float *x_sub2 = x_super + chunk + 32;
                        float q_sum1 = 0.0f, x_sum1 = 0.0f;
                        float q_sum2 = 0.0f, x_sum2 = 0.0f;
                        for (int l = 0; l < 32; l++) {
                            float q_val1 = (float)(q[l] & 0xF);
                            float q_val2 = (float)(q[l] >> 4);
                            q_sum1 += q_val1 * x_sub1[l] + (q_val1 + 1.0f) * x_sub1[l];
                            x_sum1 += x_sub1[l];
                            q_sum2 += q_val2 * x_sub2[l] + (q_val2 + 1.0f) * x_sub2[l];
                            x_sum2 += x_sub2[l];
                        }
                        row_sum += (d1 * q_sum1 - m1 * x_sum1) + (d2 * q_sum2 - m2 * x_sum2);
                        q += 32;
                        sub_block += 2;
                    }
                } else {
                    // Normal mode: SDOT
                    const BlockQ8_K *q8 = &q8_buf[b];
                    float super_sum = 0.0f;

                    for (int chunk = 0; chunk < 256; chunk += 64) {
                        uint8_t sc1, m1_raw, sc2, m2_raw;
                        get_scale_min_k4(sub_block, block->scales, &sc1, &m1_raw);
                        get_scale_min_k4(sub_block + 1, block->scales, &sc2, &m2_raw);
                        float d1 = d * sc1;
                        float m1 = dmin * m1_raw;
                        float d2 = d * sc2;
                        float m2 = dmin * m2_raw;

                        const int8_t *qs_sub1 = q8->qs + chunk;
                        const int8_t *qs_sub2 = q8->qs + chunk + 32;

                        int sum_qs1 = (int)q8->bsums[2 * sub_block]     + (int)q8->bsums[2 * sub_block + 1];
                        int sum_qs2 = (int)q8->bsums[2 * sub_block + 2] + (int)q8->bsums[2 * sub_block + 3];

#if defined(__ARM_FEATURE_DOTPROD)
                        uint8x16_t bytes0 = vld1q_u8(q);
                        uint8x16_t bytes1 = vld1q_u8(q + 16);

                        int8x16_t q1_0 = vreinterpretq_s8_u8(vandq_u8(bytes0, vdupq_n_u8(0x0F)));
                        int8x16_t q1_1 = vreinterpretq_s8_u8(vandq_u8(bytes1, vdupq_n_u8(0x0F)));

                        int8x16_t q2_0 = vreinterpretq_s8_u8(vshrq_n_u8(bytes0, 4));
                        int8x16_t q2_1 = vreinterpretq_s8_u8(vshrq_n_u8(bytes1, 4));

                        int8x16_t ax1_0 = vld1q_s8(qs_sub1);
                        int8x16_t ax1_1 = vld1q_s8(qs_sub1 + 16);
                        int8x16_t ax2_0 = vld1q_s8(qs_sub2);
                        int8x16_t ax2_1 = vld1q_s8(qs_sub2 + 16);

                        int32x4_t acc1 = vdotq_s32(vdupq_n_s32(0), q1_0, ax1_0);
                        acc1           = vdotq_s32(acc1,            q1_1, ax1_1);

                        int32x4_t acc2 = vdotq_s32(vdupq_n_s32(0), q2_0, ax2_0);
                        acc2           = vdotq_s32(acc2,            q2_1, ax2_1);

                        int32_t dot1 = vaddvq_s32(acc1);
                        int32_t dot2 = vaddvq_s32(acc2);
#else
                        int32_t dot1 = 0;
                        int32_t dot2 = 0;
                        for (int l = 0; l < 32; l++) {
                            dot1 += (int32_t)(q[l] & 0x0F) * (int32_t)qs_sub1[l];
                            dot2 += (int32_t)(q[l] >> 4)   * (int32_t)qs_sub2[l];
                        }
#endif
                        super_sum += (d1 * (float)dot1 - m1 * (float)sum_qs1);
                        super_sum += (d2 * (float)dot2 - m2 * (float)sum_qs2);

                        q += 32;
                        sub_block += 2;
                    }

                    row_sum += super_sum * q8->d;
                }
            }

            out[i] = row_sum;
        }
    };

    parallel_rows(n_out, worker);
}

void matvec_q4_k_diag(float *out, const void *W, const float *x, int n_out, int n_in, DiagMode mode) {
    switch (mode) {
        case DiagMode::MEM_ONLY:   matvec_q4_k_impl<DiagMode::MEM_ONLY>(out, W, x, n_out, n_in); break;
        case DiagMode::COMPUTE_2X: matvec_q4_k_impl<DiagMode::COMPUTE_2X>(out, W, x, n_out, n_in); break;
        default:                   matvec_q4_k_impl<DiagMode::NORMAL>(out, W, x, n_out, n_in); break;
    }
}

void matvec_q4_k(float *out, const void *W, const float *x, int n_out, int n_in) {
    matvec_q4_k_impl<DiagMode::NORMAL>(out, W, x, n_out, n_in);
}

// ── Fused Quantized Matrix-Vector Multiply (Q6_K) ──
template <DiagMode Mode>
static void matvec_q6_k_impl(float *out, const void *W, const float *x, int n_out, int n_in) {
    const BlockQ6_K *blocks = (const BlockQ6_K *)W;
    int blocks_per_row = n_in / 256;

    // Buffer for quantized activations (zero heap allocation)
    static thread_local BlockQ8_K q8_scratch[128];
    BlockQ8_K *q8_buf = q8_scratch;
    std::vector<BlockQ8_K> q8_fallback;
    if (blocks_per_row > 128) {
        q8_fallback.resize(blocks_per_row);
        q8_buf = q8_fallback.data();
    }

    if constexpr (Mode == DiagMode::NORMAL) {
        quantize_q8_k(x, q8_buf, n_in);
    }

    auto worker = [&](int row_start, int row_end) {
        for (int i = row_start; i < row_end; i++) {
            const BlockQ6_K *row = blocks + i * blocks_per_row;
            float row_sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const BlockQ6_K *block = &row[b];
                float d = f16_to_f32(block->d);

                const uint8_t *ql = block->ql;
                const uint8_t *qh = block->qh;
                const int8_t  *sc = block->scales;

                if constexpr (Mode == DiagMode::MEM_ONLY) {
                    uint32_t q_sum = 0;
                    for (int l = 0; l < 32; l++) q_sum += ql[l] + qh[l];
                    row_sum += (float)q_sum;
                } else if constexpr (Mode == DiagMode::COMPUTE_2X) {
                    const float *x_block = x + b * 256;
                    float block_sum = 0.0f;
                    for (int half = 0; half < 2; half++) {
                        const float *x_half = x_block + half * 128;
                        for (int l = 0; l < 32; l++) {
                            int is = l / 16;
                            int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                            int8_t q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                            int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                            block_sum += (float)q1 * x_half[l +  0] * (float)sc[is + 0] * 2.0f;
                            block_sum += (float)q2 * x_half[l + 32] * (float)sc[is + 2] * 2.0f;
                            block_sum += (float)q3 * x_half[l + 64] * (float)sc[is + 4] * 2.0f;
                            block_sum += (float)q4 * x_half[l + 96] * (float)sc[is + 6] * 2.0f;
                        }
                        ql += 64; qh += 32; sc += 8;
                    }
                    row_sum += d * block_sum;
                } else {
                    // Normal mode: SDOT
                    const BlockQ8_K *q8 = &q8_buf[b];
                    int32_t block_int_sum = 0;

                    for (int half = 0; half < 2; half++) {
                        const int8_t *qs_half = q8->qs + half * 128;

#if defined(__ARM_FEATURE_DOTPROD)
                        uint8x16_t mask_0f = vdupq_n_u8(0x0F);
                        uint8x16_t mask_03 = vdupq_n_u8(0x03);
                        int8x16_t c32      = vdupq_n_s8(32);

                        for (int l = 0; l < 32; l += 16) {
                            int is = l / 16;

                            uint8x16_t ql_a = vld1q_u8(ql + l);
                            uint8x16_t ql_b = vld1q_u8(ql + l + 32);
                            uint8x16_t qh_v = vld1q_u8(qh + l);

                            int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(ql_a, mask_0f),
                                vshlq_n_u8(vandq_u8(qh_v, mask_03), 4))), c32);

                            int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(ql_b, mask_0f),
                                vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 2), mask_03), 4))), c32);

                            int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8(ql_a, 4),
                                vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_v, 4), mask_03), 4))), c32);

                            int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8(ql_b, 4),
                                vshlq_n_u8(vshrq_n_u8(qh_v, 6), 4))), c32);

                            int8x16_t ax1 = vld1q_s8(qs_half + l + 0);
                            int8x16_t ax2 = vld1q_s8(qs_half + l + 32);
                            int8x16_t ax3 = vld1q_s8(qs_half + l + 64);
                            int8x16_t ax4 = vld1q_s8(qs_half + l + 96);

                            int32_t dot1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q1, ax1));
                            int32_t dot2 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q2, ax2));
                            int32_t dot3 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q3, ax3));
                            int32_t dot4 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), q4, ax4));

                            block_int_sum += dot1 * (int32_t)sc[is + 0];
                            block_int_sum += dot2 * (int32_t)sc[is + 2];
                            block_int_sum += dot3 * (int32_t)sc[is + 4];
                            block_int_sum += dot4 * (int32_t)sc[is + 6];
                        }
#else
                        for (int l = 0; l < 32; l++) {
                            int is = l / 16;
                            int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                            int8_t q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                            int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                            block_int_sum += (int32_t)q1 * (int32_t)qs_half[l +  0] * (int32_t)sc[is + 0];
                            block_int_sum += (int32_t)q2 * (int32_t)qs_half[l + 32] * (int32_t)sc[is + 2];
                            block_int_sum += (int32_t)q3 * (int32_t)qs_half[l + 64] * (int32_t)sc[is + 4];
                            block_int_sum += (int32_t)q4 * (int32_t)qs_half[l + 96] * (int32_t)sc[is + 6];
                        }
#endif
                        ql += 64;
                        qh += 32;
                        sc += 8;
                    }

                    row_sum += (float)block_int_sum * (d * q8->d);
                }
            }

            out[i] = row_sum;
        }
    };

    parallel_rows(n_out, worker);
}

void matvec_q6_k_diag(float *out, const void *W, const float *x, int n_out, int n_in, DiagMode mode) {
    switch (mode) {
        case DiagMode::MEM_ONLY:   matvec_q6_k_impl<DiagMode::MEM_ONLY>(out, W, x, n_out, n_in); break;
        case DiagMode::COMPUTE_2X: matvec_q6_k_impl<DiagMode::COMPUTE_2X>(out, W, x, n_out, n_in); break;
        default:                   matvec_q6_k_impl<DiagMode::NORMAL>(out, W, x, n_out, n_in); break;
    }
}

void matvec_q6_k(float *out, const void *W, const float *x, int n_out, int n_in) {
    matvec_q6_k_impl<DiagMode::NORMAL>(out, W, x, n_out, n_in);
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

    // A helper lambda to rotate one head's slice in-place.
    // Iterates over pairs (d, d+1), computing and applying the rotation angle.
    auto rotate_head = [&](float *h_ptr) {
        for (int d = 0; d < rope_dim_count; d += 2) {
            int i = d / 2;

            // θᵢ = pos / (freq_base ^ (2i / head_dim))
            float theta = (float)pos / powf(freq_base, (float)(2 * i) / head_dim);
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            float v0 = h_ptr[d];
            float v1 = h_ptr[d + 1];
            h_ptr[d]     = v0 * cos_t - v1 * sin_t;
            h_ptr[d + 1] = v0 * sin_t + v1 * cos_t;
        }
    };

    // Rotate all Q heads (n_heads total)
    for (int h = 0; h < n_heads; h++) {
        rotate_head(q + h * head_dim);
    }

    // Rotate all K heads (n_kv_heads total — fewer due to GQA)
    for (int h = 0; h < n_kv_heads; h++) {
        rotate_head(k + h * head_dim);
    }
}
