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

// ── Matrix-Vector Multiply ──
// Computes: out = W * x
//
// This is the core operation of neural network inference.
// Every linear layer (Q, K, V projections, FFN layers) is a mat-vec multiply.
//
// Parameters:
//   out   — output vector  (n_out floats)
//   W     — weight matrix  (n_out × n_in floats, stored row-major)
//   x     — input vector   (n_in floats)
//   n_out — number of output elements (rows of W)
//   n_in  — number of input elements  (columns of W, length of x)
//
// Row-major layout: W[row][col] = W[row * n_in + col]
// Each output element is the dot product of one row of W with x.
//
void matvec(float *out, const float *W, const float *x, int n_out, int n_in);

#endif // OPS_H
