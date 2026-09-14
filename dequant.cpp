#include "dequant.h"

// ── f16_to_f32 ──
// IEEE 754 float16: [1 sign][5 exponent][10 mantissa]
// IEEE 754 float32: [1 sign][8 exponent][23 mantissa]
// We need to re-pack the bits from the smaller format into the larger one.

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;          // top bit
    uint32_t exp  = (h >> 10) & 0x1F;         // next 5 bits
    uint32_t mant = h & 0x3FF;                // bottom 10 bits

    uint32_t result;

    if (exp == 0) {
        // zero or subnormal — just treat as 0 for our purposes
        result = sign << 31;
    } else if (exp == 0x1F) {
        // inf or NaN — extend to float32
        result = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        // normal number:
        // float16 exponent bias is 15, float32 bias is 127
        // so we adjust: new_exp = exp - 15 + 127 = exp + 112
        result = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }

    // type-pun the uint32 bits into a float
    float f;
    __builtin_memcpy(&f, &result, sizeof(f));
    return f;
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

static void get_scale_min(int j, const uint8_t *scales, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
    }
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
