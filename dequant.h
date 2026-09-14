#ifndef DEQUANT_H
#define DEQUANT_H

#include "model_loader.h"   // for GGMLType

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

// ── Q6_K format ──
// Higher-precision quantization: 6 bits per weight.
// Each super-block encodes 256 weights:
//
//   [ 128 bytes: ql ][ 64 bytes: qh ][ 16 bytes: scales ][ float16 d ]
//   = 128 + 64 + 16 + 2 = 210 bytes per super-block of 256 weights
//
// Each weight's 6-bit value is split across ql (low 4 bits) and qh (high 2 bits).
// The 16 int8 scales cover 16 sub-blocks of 16 weights each.
//
// Formula: float = d * scale * (q6 - 32)
//
// Compare to F32: 256 × 4 = 1024 bytes → ~4.9x smaller

struct BlockQ6_K {
    uint8_t  ql[128];    // low 4 bits of 6-bit quants (2 nibbles per byte)
    uint8_t  qh[64];     // high 2 bits of 6-bit quants (4 values per byte)
    int8_t   scales[16]; // sub-block scales (one per 16 weights)
    uint16_t d;          // super-block scale (float16)
};

// ── API ──

// Convert float16 (as stored in GGUF) to float32
float f16_to_f32(uint16_t h);

// Dequantize a Q8_0 tensor: reads n_elements from src, writes floats to dst.
// dst must be pre-allocated with n_elements floats.
void dequantize_q8_0(const void *src, float *dst, size_t n_elements);

// Dequantize a Q4_K tensor.
void dequantize_q4_k(const void *src, float *dst, size_t n_elements);

// Dequantize a Q6_K tensor.
void dequantize_q6_k(const void *src, float *dst, size_t n_elements);

// ── Unified dispatcher ──
// Dequantize any supported tensor type into float32.
// Handles F32 (memcpy), Q8_0, Q4_K, Q6_K.
// Returns false if the type is unsupported.
bool dequantize(const void *src, float *dst, size_t n_elements, GGMLType type);

#endif // DEQUANT_H
