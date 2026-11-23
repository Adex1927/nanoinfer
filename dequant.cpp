#include "dequant.h"
#include <cstdio>
#include <cmath>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// ── quantize_q8_0 ──
// Quantizes float32 values into BlockQ8_0 blocks (32 values per 34-byte block).
// Each block stores:
//   - scale: float16(max(|x[0..31]|) / 127.0f)
//   - quants[32]: round(x[j] * (127.0f / max_abs)) in [-127, 127]
void quantize_q8_0(const float *src, BlockQ8_0 *dst, size_t n_elements) {
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *x = src + b * 32;
        BlockQ8_0 *block = &dst[b];

#if defined(__ARM_NEON)
        float32x4_t x0 = vld1q_f32(x + 0);
        float32x4_t x1 = vld1q_f32(x + 4);
        float32x4_t x2 = vld1q_f32(x + 8);
        float32x4_t x3 = vld1q_f32(x + 12);
        float32x4_t x4 = vld1q_f32(x + 16);
        float32x4_t x5 = vld1q_f32(x + 20);
        float32x4_t x6 = vld1q_f32(x + 24);
        float32x4_t x7 = vld1q_f32(x + 28);

        float32x4_t m01 = vmaxq_f32(vabsq_f32(x0), vabsq_f32(x1));
        float32x4_t m23 = vmaxq_f32(vabsq_f32(x2), vabsq_f32(x3));
        float32x4_t m45 = vmaxq_f32(vabsq_f32(x4), vabsq_f32(x5));
        float32x4_t m67 = vmaxq_f32(vabsq_f32(x6), vabsq_f32(x7));
        float max_val = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m01, m23), vmaxq_f32(m45, m67)));

        if (max_val == 0.0f) {
            block->scale = 0;
            std::memset(block->quants, 0, sizeof(block->quants));
            continue;
        }

        float d  = max_val / 127.0f;
        float id = 127.0f / max_val;
        block->scale = f32_to_f16(d);

        float32x4_t vid = vdupq_n_f32(id);
        int16x8_t q01 = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x0, vid))),
                                     vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x1, vid))));
        int16x8_t q23 = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x2, vid))),
                                     vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x3, vid))));
        int16x8_t q45 = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x4, vid))),
                                     vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x5, vid))));
        int16x8_t q67 = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x6, vid))),
                                     vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(x7, vid))));

        vst1q_s8(block->quants + 0,  vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23)));
        vst1q_s8(block->quants + 16, vcombine_s8(vqmovn_s16(q45), vqmovn_s16(q67)));
#else
        float max_val = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(x[j]);
            if (v > max_val) max_val = v;
        }
        if (max_val == 0.0f) {
            block->scale = 0;
            std::memset(block->quants, 0, sizeof(block->quants));
            continue;
        }
        float d  = max_val / 127.0f;
        float id = 127.0f / max_val;
        block->scale = f32_to_f16(d);
        for (int j = 0; j < 32; j++) {
            int q = (int)roundf(x[j] * id);
            if (q > 127)  q = 127;
            if (q < -128) q = -128;
            block->quants[j] = (int8_t)q;
        }
#endif
    }
}

// ── quantize_f16 / dequantize_f16 ──
// Hardware-accelerated FP32 <-> FP16 batch conversion (2 bytes per element).
void quantize_f16(const float *src, uint16_t *dst, size_t n_elements) {
#if defined(__ARM_NEON)
    size_t i = 0;
    for (; i + 8 <= n_elements; i += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(src + i));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(src + i + 4));
        vst1q_u16(dst + i, vreinterpretq_u16_f16(vcombine_f16(lo, hi)));
    }
    for (; i < n_elements; i++) {
        dst[i] = f32_to_f16(src[i]);
    }
#else
    for (size_t i = 0; i < n_elements; i++) {
        dst[i] = f32_to_f16(src[i]);
    }
#endif
}

void dequantize_f16(const uint16_t *src, float *dst, size_t n_elements) {
#if defined(__ARM_NEON)
    size_t i = 0;
    for (; i + 8 <= n_elements; i += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
        vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n_elements; i++) {
        dst[i] = f16_to_f32(src[i]);
    }
#else
    for (size_t i = 0; i < n_elements; i++) {
        dst[i] = f16_to_f32(src[i]);
    }
#endif
}

// ── dequantize_q8_0 ──
// The simplest one. Each block = 34 bytes = [float16 scale][32 × int8]
// For each weight: float_value = scale * quant_value

void dequantize_q8_0(const void *src, float *dst, size_t n_elements) {
    const BlockQ8_0 *blocks = (const BlockQ8_0 *)src;
    size_t n_blocks = n_elements / 32;  // 32 weights per block

    for (size_t b = 0; b < n_blocks; b++) {
        float scale = f16_to_f32(blocks[b].scale);

        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = scale * blocks[b].quants[j];
        }
    }
}

// ── dequantize_q4_k ──
// Each super-block = 144 bytes = 256 weights.
// The 256 weights are split into 8 sub-blocks of 32.
// Each sub-block has its own 6-bit scale and 6-bit min,
// packed into the 12-byte scales[] array.
//
// The quants are 4-bit: each byte holds 2 weights
//   low nibble  = quants[i] & 0xF    (bits 0-3)
//   high nibble = quants[i] >> 4      (bits 4-7)
//
// Formula per weight:
//   float = super_d * sub_scale * quant_4bit - super_dmin * sub_min

// Helper: unpack the 6-bit sub-block scale and min from the scales[] array.
// The 12 bytes encode 8 scales + 8 mins in a specific packed format:
//   bytes 0-3:  low 4 bits of scales for sub-blocks 0-7 (packed 2 per byte? no, 1 per byte for first 4)
//   bytes 4-7:  low 4 bits of mins for sub-blocks 0-7
//   bytes 8-11: high 2 bits of scales AND mins for sub-blocks 0-7
//
// For sub-blocks 0-3 (j < 4):
//   scale = scales[j] & 0x3F          (6 bits from one byte)
//   min   = scales[j+4] & 0x3F
//
// For sub-blocks 4-7 (j >= 4):
//   scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//   min   = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)

static inline void get_scale_min(int j, const uint8_t *scales, uint8_t *sc, uint8_t *m) {
    get_scale_min_k4(j, scales, sc, m);
}

void dequantize_q4_k(const void *src, float *dst, size_t n_elements) {
    const BlockQ4_K *blocks = (const BlockQ4_K *)src;
    size_t n_blocks = n_elements / 256;  // 256 weights per super-block

    for (size_t b = 0; b < n_blocks; b++) {
        const BlockQ4_K *block = &blocks[b];
        const uint8_t *q = block->quants;

        float d    = f16_to_f32(block->d);
        float dmin = f16_to_f32(block->dmin);

        // process 256 weights in chunks of 64 (2 sub-blocks of 32 per chunk)
        int sub_block = 0;
        for (int chunk = 0; chunk < 256; chunk += 64) {
            // first 32 weights in this chunk: use low nibble
            uint8_t sc, m;
            get_scale_min(sub_block, block->scales, &sc, &m);
            float d1 = d * sc;
            float m1 = dmin * m;
            for (int l = 0; l < 32; l++) {
                dst[b * 256 + chunk + l] = d1 * (q[l] & 0xF) - m1;
            }

            // next 32 weights: use high nibble of the same bytes
            sub_block++;
            get_scale_min(sub_block, block->scales, &sc, &m);
            float d2 = d * sc;
            float m2 = dmin * m;
            for (int l = 0; l < 32; l++) {
                dst[b * 256 + chunk + 32 + l] = d2 * (q[l] >> 4) - m2;
            }

            q += 32;  // advance to next 32 bytes of quants
            sub_block++;
        }
    }
}

// ── dequantize_q6_k ──
// 6-bit quantization. Each super-block = 210 bytes = 256 weights.
// The 6-bit value for each weight is split across two arrays:
//   ql[128] — low 4 bits  (two nibbles per byte)
//   qh[64]  — high 2 bits (four 2-bit values per byte)
//
// The 256 weights are processed in two halves of 128.
// Within each half, we iterate l=0..31 and extract 4 values per iteration:
//   - at offsets l+0, l+32, l+64, l+96 within the half
//
// Each of the 16 sub-blocks of 16 weights gets its own int8 scale.
// Formula: float = d * scale * (q6 - 32)

void dequantize_q6_k(const void *src, float *dst, size_t n_elements) {
    const BlockQ6_K *blocks = (const BlockQ6_K *)src;
    size_t n_blocks = n_elements / 256;  // 256 weights per super-block

    for (size_t b = 0; b < n_blocks; b++) {
        const BlockQ6_K *block = &blocks[b];
        float d = f16_to_f32(block->d);

        const uint8_t *ql = block->ql;
        const uint8_t *qh = block->qh;
        const int8_t  *sc = block->scales;
        float *y = dst + b * 256;

        // process two halves of 128 weights each
        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                // which sub-block (of 16 weights) determines the scale index
                int is = l / 16;

                // reconstruct 4 six-bit values from ql and qh:
                //   q1 @ offset l+0:   low nibble of ql[l]    + bits 0-1 of qh[l]
                //   q2 @ offset l+32:  low nibble of ql[l+32] + bits 2-3 of qh[l]
                //   q3 @ offset l+64:  high nibble of ql[l]   + bits 4-5 of qh[l]
                //   q4 @ offset l+96:  high nibble of ql[l+32]+ bits 6-7 of qh[l]
                int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }

            // advance to next half
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ── quantize_q8_k ──
// Quantize float activations into BlockQ8_K super-blocks (256 values per block).
void quantize_q8_k(const float *src, BlockQ8_K *dst, size_t n_elements) {
    size_t n_blocks = n_elements / 256;
    for (size_t b = 0; b < n_blocks; b++) {
        const float *x = src + b * 256;
        BlockQ8_K *block = &dst[b];

        float max_val = 0.0f;
        for (int i = 0; i < 256; i++) {
            float v = fabsf(x[i]);
            if (v > max_val) max_val = v;
        }

        if (max_val == 0.0f) {
            block->d = 0.0f;
            std::memset(block->qs, 0, sizeof(block->qs));
            std::memset(block->bsums, 0, sizeof(block->bsums));
            continue;
        }

        float d = max_val / 127.0f;
        float id = 127.0f / max_val;
        block->d = d;

        for (int c = 0; c < 16; c++) {
            int16_t sum = 0;
            for (int l = 0; l < 16; l++) {
                int idx = c * 16 + l;
                int q = (int)roundf(x[idx] * id);
                if (q > 127)  q = 127;
                if (q < -128) q = -128;
                block->qs[idx] = (int8_t)q;
                sum += (int16_t)q;
            }
            block->bsums[c] = sum;
        }
    }
}

// ── unified dequantize dispatcher ──

bool dequantize(const void *src, float *dst, size_t n_elements, GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:
            // no dequantization needed — just copy
            __builtin_memcpy(dst, src, n_elements * sizeof(float));
            return true;

        case GGML_TYPE_Q8_0:
            dequantize_q8_0(src, dst, n_elements);
            return true;

        case GGML_TYPE_Q4_K:
            dequantize_q4_k(src, dst, n_elements);
            return true;

        case GGML_TYPE_Q6_K:
            dequantize_q6_k(src, dst, n_elements);
            return true;

        default:
            fprintf(stderr, "dequantize: unsupported type %u\n", (unsigned)type);
            return false;
    }
}

// ── row size helper ──

size_t row_bytes(GGMLType type, size_t n_cols) {
    switch (type) {
        case GGML_TYPE_F32:
            return n_cols * sizeof(float);

        case GGML_TYPE_Q8_0:   // 32 weights per block
            return (n_cols % 32 == 0) ? (n_cols / 32) * sizeof(BlockQ8_0) : 0;

        case GGML_TYPE_Q4_K:   // 256 weights per super-block
            return (n_cols % 256 == 0) ? (n_cols / 256) * sizeof(BlockQ4_K) : 0;

        case GGML_TYPE_Q6_K:   // 256 weights per super-block
            return (n_cols % 256 == 0) ? (n_cols / 256) * sizeof(BlockQ6_K) : 0;

        default:
            return 0;
    }
}
