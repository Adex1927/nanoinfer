#include "ops.h"
#include "dequant.h"
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

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

    const int n_threads = (n_out >= 64) ? 4 : 1;
    if (n_threads == 1) {
        worker(0, n_out);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        int chunk_size = n_out / n_threads;

        for (int t = 0; t < n_threads; t++) {
            int r_start = t * chunk_size;
            int r_end   = (t == n_threads - 1) ? n_out : r_start + chunk_size;
            workers.emplace_back(worker, r_start, r_end);
        }
        for (auto &w : workers) {
            w.join();
        }
    }
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

    auto worker = [&](int row_start, int row_end) {
        for (int i = row_start; i < row_end; i++) {
            const BlockQ4_K *row = blocks + i * blocks_per_row;
            float row_sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const BlockQ4_K *block = &row[b];
                const float *x_super = x + b * 256;
                const uint8_t *q = block->quants;

                float d    = f16_to_f32(block->d);
                float dmin = f16_to_f32(block->dmin);

                int sub_block = 0;
                for (int chunk = 0; chunk < 256; chunk += 64) {
                    // sub-block 1: low nibble
                    uint8_t sc1, m1_raw;
                    get_scale_min_k4(sub_block, block->scales, &sc1, &m1_raw);
                    float d1 = d * sc1;
                    float m1 = dmin * m1_raw;

                    // sub-block 2: high nibble
                    uint8_t sc2, m2_raw;
                    get_scale_min_k4(sub_block + 1, block->scales, &sc2, &m2_raw);
                    float d2 = d * sc2;
                    float m2 = dmin * m2_raw;

                    const float *x_sub1 = x_super + chunk;
                    const float *x_sub2 = x_super + chunk + 32;

                    if constexpr (Mode == DiagMode::MEM_ONLY) {
                        uint32_t q_sum = 0;
                        for (int l = 0; l < 32; l++) {
                            q_sum += q[l];
                        }
                        row_sum += (float)(q_sum * 2);
                    } else if constexpr (Mode == DiagMode::COMPUTE_2X) {
                        float q_sum1 = 0.0f;
                        float x_sum1 = 0.0f;
                        for (int l = 0; l < 32; l++) {
                            float q_val = (float)(q[l] & 0xF);
                            float x_val = x_sub1[l];
                            q_sum1 += q_val * x_val;
                            q_sum1 += (q_val + 1.0f) * x_val;
                            x_sum1 += x_val;
                        }
                        row_sum += d1 * q_sum1 - m1 * x_sum1;

                        float q_sum2 = 0.0f;
                        float x_sum2 = 0.0f;
                        for (int l = 0; l < 32; l++) {
                            float q_val = (float)(q[l] >> 4);
                            float x_val = x_sub2[l];
                            q_sum2 += q_val * x_val;
                            q_sum2 += (q_val + 1.0f) * x_val;
                            x_sum2 += x_val;
                        }
                        row_sum += d2 * q_sum2 - m2 * x_sum2;
                    } else {
#if defined(__ARM_NEON)
                        // Multi-accumulators: 2 independent accumulators per stream
                        float32x4_t acc_q1_0 = vdupq_n_f32(0.0f);
                        float32x4_t acc_q1_1 = vdupq_n_f32(0.0f);
                        float32x4_t acc_x1_0 = vdupq_n_f32(0.0f);
                        float32x4_t acc_x1_1 = vdupq_n_f32(0.0f);

                        float32x4_t acc_q2_0 = vdupq_n_f32(0.0f);
                        float32x4_t acc_q2_1 = vdupq_n_f32(0.0f);
                        float32x4_t acc_x2_0 = vdupq_n_f32(0.0f);
                        float32x4_t acc_x2_1 = vdupq_n_f32(0.0f);

                        uint8x8_t mask = vdup_n_u8(0x0F);

                        for (int l = 0; l < 32; l += 8) {
                            uint8x8_t bytes = vld1_u8(q + l);
                            uint8x8_t q1_u8 = vand_u8(bytes, mask);
                            uint8x8_t q2_u8 = vshr_n_u8(bytes, 4);

                            // ── Sub-block 1 ──
                            uint16x8_t q1_16  = vmovl_u8(q1_u8);
                            float32x4_t q1_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q1_16)));
                            float32x4_t q1_f1 = vcvtq_f32_u32(vmovl_high_u16(q1_16));

                            float32x4_t x1_0  = vld1q_f32(x_sub1 + l);
                            float32x4_t x1_1  = vld1q_f32(x_sub1 + l + 4);

                            acc_q1_0 = vfmaq_f32(acc_q1_0, q1_f0, x1_0);
                            acc_q1_1 = vfmaq_f32(acc_q1_1, q1_f1, x1_1);
                            acc_x1_0 = vaddq_f32(acc_x1_0, x1_0);
                            acc_x1_1 = vaddq_f32(acc_x1_1, x1_1);

                            // ── Sub-block 2 ──
                            uint16x8_t q2_16  = vmovl_u8(q2_u8);
                            float32x4_t q2_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q2_16)));
                            float32x4_t q2_f1 = vcvtq_f32_u32(vmovl_high_u16(q2_16));

                            float32x4_t x2_0  = vld1q_f32(x_sub2 + l);
                            float32x4_t x2_1  = vld1q_f32(x_sub2 + l + 4);

                            acc_q2_0 = vfmaq_f32(acc_q2_0, q2_f0, x2_0);
                            acc_q2_1 = vfmaq_f32(acc_q2_1, q2_f1, x2_1);
                            acc_x2_0 = vaddq_f32(acc_x2_0, x2_0);
                            acc_x2_1 = vaddq_f32(acc_x2_1, x2_1);
                        }

                        // Combine interleaved accumulators
                        float q_sum1 = vaddvq_f32(vaddq_f32(acc_q1_0, acc_q1_1));
                        float x_sum1 = vaddvq_f32(vaddq_f32(acc_x1_0, acc_x1_1));
                        float q_sum2 = vaddvq_f32(vaddq_f32(acc_q2_0, acc_q2_1));
                        float x_sum2 = vaddvq_f32(vaddq_f32(acc_x2_0, acc_x2_1));

                        row_sum += (d1 * q_sum1 - m1 * x_sum1) + (d2 * q_sum2 - m2 * x_sum2);
#else
                        float q_sum1 = 0.0f;
                        float x_sum1 = 0.0f;
                        for (int l = 0; l < 32; l++) {
                            q_sum1 += (float)(q[l] & 0xF) * x_sub1[l];
                            x_sum1 += x_sub1[l];
                        }
                        row_sum += d1 * q_sum1 - m1 * x_sum1;

                        float q_sum2 = 0.0f;
                        float x_sum2 = 0.0f;
                        for (int l = 0; l < 32; l++) {
                            q_sum2 += (float)(q[l] >> 4) * x_sub2[l];
                            x_sum2 += x_sub2[l];
                        }
                        row_sum += d2 * q_sum2 - m2 * x_sum2;
#endif
                    }

                    q += 32;
                    sub_block += 2;
                }
            }

            out[i] = row_sum;
        }
    };

    // Determine thread count (default 4 if rows >= 64, else single thread)
    const int n_threads = (n_out >= 64) ? 4 : 1;
    if (n_threads == 1) {
        worker(0, n_out);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        int chunk_size = n_out / n_threads;

        for (int t = 0; t < n_threads; t++) {
            int r_start = t * chunk_size;
            int r_end   = (t == n_threads - 1) ? n_out : r_start + chunk_size;
            workers.emplace_back(worker, r_start, r_end);
        }
        for (auto &w : workers) {
            w.join();
        }
    }
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
                const float   *x_block = x + b * 256;

                float block_sum = 0.0f;

                for (int half = 0; half < 2; half++) {
                    const float *x_half = x_block + half * 128;

#if defined(__ARM_NEON)
                    float32x4_t acc0 = vdupq_n_f32(0.0f);
                    float32x4_t acc1 = vdupq_n_f32(0.0f);

                    uint8x8_t mask_0f = vdup_n_u8(0x0F);
                    uint8x8_t mask_03 = vdup_n_u8(0x03);
                    uint8x8_t c32     = vdup_n_u8(32);

                    for (int l = 0; l < 32; l += 8) {
                        int is = l / 16;

                        // 1. Load 8 bytes each of ql and qh
                        uint8x8_t ql_a = vld1_u8(ql + l);
                        uint8x8_t ql_b = vld1_u8(ql + l + 32);
                        uint8x8_t qh_v = vld1_u8(qh + l);

                        // 2. Extract 4-bit low nibbles
                        uint8x8_t q1_low = vand_u8(ql_a, mask_0f);
                        uint8x8_t q2_low = vand_u8(ql_b, mask_0f);
                        uint8x8_t q3_low = vshr_n_u8(ql_a, 4);
                        uint8x8_t q4_low = vshr_n_u8(ql_b, 4);

                        // 3. Extract 2-bit high slices and shift to bits 4..5
                        uint8x8_t q1_high = vshl_n_u8(vand_u8(qh_v, mask_03), 4);
                        uint8x8_t q2_high = vshl_n_u8(vand_u8(vshr_n_u8(qh_v, 2), mask_03), 4);
                        uint8x8_t q3_high = vshl_n_u8(vand_u8(vshr_n_u8(qh_v, 4), mask_03), 4);
                        uint8x8_t q4_high = vshl_n_u8(vshr_n_u8(qh_v, 6), 4);

                        // 4. Combine into 6-bit values and subtract 32 -> signed int8 [-32..+31]
                        int8x8_t q1_s8 = vsub_s8(vreinterpret_s8_u8(vorr_u8(q1_low, q1_high)), vreinterpret_s8_u8(c32));
                        int8x8_t q2_s8 = vsub_s8(vreinterpret_s8_u8(vorr_u8(q2_low, q2_high)), vreinterpret_s8_u8(c32));
                        int8x8_t q3_s8 = vsub_s8(vreinterpret_s8_u8(vorr_u8(q3_low, q3_high)), vreinterpret_s8_u8(c32));
                        int8x8_t q4_s8 = vsub_s8(vreinterpret_s8_u8(vorr_u8(q4_low, q4_high)), vreinterpret_s8_u8(c32));

                        // Helper lambda using 2 independent accumulators
                        auto fma_scaled = [&](int8x8_t q_vec, float scale, const float *x_ptr) {
                            int16x8_t q16 = vmovl_s8(q_vec);
                            float32x4_t qf0 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16))), scale);
                            float32x4_t qf1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(q16)), scale);

                            acc0 = vfmaq_f32(acc0, qf0, vld1q_f32(x_ptr));
                            acc1 = vfmaq_f32(acc1, qf1, vld1q_f32(x_ptr + 4));
                        };

                        fma_scaled(q1_s8, (float)sc[is + 0], x_half + l +  0);
                        fma_scaled(q2_s8, (float)sc[is + 2], x_half + l + 32);
                        fma_scaled(q3_s8, (float)sc[is + 4], x_half + l + 64);
                        fma_scaled(q4_s8, (float)sc[is + 6], x_half + l + 96);
                    }

                    block_sum += vaddvq_f32(vaddq_f32(acc0, acc1));
#else
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;

                        int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int8_t q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                        block_sum += (float)q1 * x_half[l +  0] * (float)sc[is + 0];
                        block_sum += (float)q2 * x_half[l + 32] * (float)sc[is + 2];
                        block_sum += (float)q3 * x_half[l + 64] * (float)sc[is + 4];
                        block_sum += (float)q4 * x_half[l + 96] * (float)sc[is + 6];
                    }
#endif

                    ql += 64;
                    qh += 32;
                    sc += 8;
                }

                row_sum += d * block_sum;
            }

            out[i] = row_sum;
        }
    };

    const int n_threads = (n_out >= 64) ? 4 : 1;
    if (n_threads == 1) {
        worker(0, n_out);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        int chunk_size = n_out / n_threads;

        for (int t = 0; t < n_threads; t++) {
            int r_start = t * chunk_size;
            int r_end   = (t == n_threads - 1) ? n_out : r_start + chunk_size;
            workers.emplace_back(worker, r_start, r_end);
        }
        for (auto &w : workers) {
            w.join();
        }
    }
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
