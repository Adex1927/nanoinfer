#ifndef OPS_H
#define OPS_H

#include <cstddef>

// ── RMSNorm ──
// Applies Root Mean Square Layer Normalization.
//
// Formula:  out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
//
// Parameters:
//   out    — output buffer  (n floats, can be same as x for in-place)
//   x      — input vector   (n floats)
//   weight — learned scale  (n floats, from the model's norm tensor)
//   n      — vector length  (n_embd, e.g. 2048)
//   eps    — small constant to prevent division by zero (typically 1e-5)
//
void rmsnorm(float *out, const float *x, const float *weight, int n, float eps = 1e-5f);

#endif // OPS_H
