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

// ── Softmax ──
// Converts a vector of raw scores into a probability distribution.
// All output values are in (0, 1] and sum to 1.0.
//
// Formula:  out[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
//
// Why subtract max first? exp() overflows quickly for large inputs.
// Subtracting the max keeps all exponents <= 0 (values in (0,1]),
// which is numerically safe. The result is mathematically identical.
//
// In-place safe: out and x can be the same pointer.
//
void softmax(float *out, const float *x, int n);

// ── SiLU (Sigmoid Linear Unit) ──
// The activation function used in LLaMA's FFN (as part of SwiGLU).
//
// Formula:  out[i] = x[i] * sigmoid(x[i])
//                  = x[i] / (1 + exp(-x[i]))
//
// Intuition: like ReLU but smooth — negative values are dampened
// rather than hard-zeroed, which helps gradient flow during training.
//
// Applied element-wise. In-place safe.
//
void silu(float *out, const float *x, int n);

// ── RoPE (Rotary Positional Embedding) ──
// Encodes token position by rotating Q and K vectors in-place.
//
// Each head's dimension is split into pairs (q[0],q[1]), (q[2],q[3]), ...
// Pair i is rotated by angle: θᵢ = pos / (freq_base ^ (2i / head_dim))
//
// Parameters:
//   q              — query vector  (n_heads    * head_dim floats), modified in-place
//   k              — key vector    (n_kv_heads * head_dim floats), modified in-place
//   pos            — token position in the sequence (0-indexed)
//   n_heads        — number of Q heads
//   n_kv_heads     — number of K/V heads (may differ from n_heads for GQA)
//   head_dim       — floats per head (n_embd / n_heads)
//   rope_dim_count — number of dimensions to rotate per head
//   freq_base      — RoPE frequency base (from hparams.rope_freq_base, e.g. 10000.0)
//
void rope(float *q, float *k, int pos, int n_heads, int n_kv_heads,
          int head_dim, int rope_dim_count, float freq_base);

#endif // OPS_H
