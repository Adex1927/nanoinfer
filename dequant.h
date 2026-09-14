#ifndef DEQUANT_H
#define DEQUANT_H

#include <cstdint>
#include <cstddef>

// ── Q8_0 format ──
// The simplest quantized format. Each block encodes 32 weights:
//
//   [ float16 scale ][ 32 × int8 values ]
//   = 2 + 32 = 34 bytes per block
//
// To get the original float: value = scale * quant
//
// Compare to F32: 32 floats × 4 bytes = 128 bytes
// So Q8_0 is ~3.7x smaller (34 vs 128 bytes for 32 weights)

struct BlockQ8_0 {
    uint16_t scale;      // float16 (we'll convert to float32)
    int8_t   quants[32]; // 32 quantized values
};

// ── Q4_K format ──
// More complex. Uses 4-bit quantization with block scales and mins.
// Each "super-block" encodes 256 weights, split into 8 sub-blocks of 32.
//
//   [ float16 d ][ float16 dmin ][ 12 bytes: sub-block scales+mins ][ 128 bytes: 256 × 4-bit values ]
//   = 2 + 2 + 12 + 128 = 144 bytes per super-block of 256 weights
//
// Compare to F32: 256 × 4 = 1024 bytes → ~7x smaller

struct BlockQ4_K {
    uint16_t d;            // super-block scale (float16)
    uint16_t dmin;         // super-block min (float16)
    uint8_t  scales[12];   // sub-block scales and mins (packed)
    uint8_t  quants[128];  // 256 × 4-bit values (2 per byte)
};

// ── API ──

// Convert float16 (as stored in GGUF) to float32
float f16_to_f32(uint16_t h);

// Dequantize a Q8_0 tensor: reads n_elements from src, writes floats to dst.
// dst must be pre-allocated with n_elements floats.
void dequantize_q8_0(const void *src, float *dst, size_t n_elements);

// Dequantize a Q4_K tensor.
void dequantize_q4_k(const void *src, float *dst, size_t n_elements);

#endif // DEQUANT_H
