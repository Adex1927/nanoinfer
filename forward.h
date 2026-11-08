#ifndef FORWARD_H
#define FORWARD_H

#include "infer_state.h"
#include "llama_model.h"
#include "probe.h"

// ── forward.h ──
// Stateful forward operations that read/write InferState.
//
// Unlike ops.h (pure math on float*), functions here are model-aware:
// they consume and produce state rather than isolated buffers.
//
// This is also the natural home for alternative attention implementations
// (e.g. sliding window, flash attention, cross-attention, convolution layers)
// as the project grows.

// ── forward_layer ──
// Runs one complete transformer layer: attention block + FFN block.
// Reads quantized weights from llama->layers[layer], dequantizes on the fly.
//
// On entry:  s->x holds the residual stream for the current token.
// On exit:   s->x holds the updated residual after this layer.
//
// pa_* — ProbeAccum buckets owned by the caller (forward_token).
//         Each stage appends one sample; the caller prints the summary.
//
void forward_layer(LlamaModel *llama, InferState *s, int layer, int pos,
                   ProbeAccum *pa_norm,
                   ProbeAccum *pa_qkv,
                   ProbeAccum *pa_rope,
                   ProbeAccum *pa_mha,
                   ProbeAccum *pa_out_proj,
                   ProbeAccum *pa_ffn);

// ── mha — Multi-Head Attention with GQA ──
// Computes scaled dot-product attention for one token at position `pos`.\
//
// Reads:   s->q, s->k, s->v        (set by the caller before this call)
// Writes:  s->xb                   (attention output, [n_embd])
//          s->k_cache, s->v_cache  (current K and V stored at layer/pos slot)
//
// GQA grouping: kv_head = h / (n_heads / n_kv_heads)
//   For TinyLlama: 32 Q heads / 4 KV heads = 8 Q heads per KV group.
//
void mha(InferState *s, int layer, int pos);
void mha_batch(InferState *s, int layer, int start_pos, int n_tokens);

// ── forward_token / forward_batch ──
// Runs a full forward pass for 1 token (forward_token) or a batch of n_tokens
// contiguous tokens starting at sequence position `start_pos` (forward_batch):
//   1. Embedding lookup: token ids -> s->x
//   2. Run through all N transformer layers (updates KV cache at start_pos..start_pos+n_tokens-1)
//   3. If need_logits is true:
//        - all_logits == false (default): Final RMSNorm + LM head on the last token -> s->logits[0..n_vocab-1]
//        - all_logits == true:            Final RMSNorm + batched LM head across all n_tokens -> s->logits[t * n_vocab]
//
// If need_logits is false (e.g. during prompt prefill before the final chunk),
// step 3 is skipped to avoid evaluating the output weight matrix.
void forward_token(LlamaModel *llama, InferState *s, int token, int pos, bool need_logits);
void forward_batch(LlamaModel *llama, InferState *s, const int *tokens, int n_tokens, int start_pos,
                   bool need_logits, bool all_logits = false);

// ── prefill ──
// Feeds a sequence of prompt tokens into the model to populate the KV cache.
// Loops pos from 0 to n_tokens - 1.
// Only computes logits on the final token (pos == n_tokens - 1).
void prefill(LlamaModel *llama, InferState *s, const int *tokens, int n_tokens);

// ── argmax_token ──
// Returns the index of the highest value in s->logits + token_idx * n_vocab (greedy sampling).
int argmax_token(const InferState *s, int token_idx = 0);

// ── generate ──
// Greedy autoregressive generation:
//   1. prefill(prompt)                -> logits for the first new token
//   2. loop: pick argmax, emit it, feed it back at the next pos, repeat
// Stops when max_new tokens were produced, EOS is sampled, or the context is full.
//
// on_token (optional) is called for every generated token (EOS excluded),
// which lets the caller stream output. Returns the number of tokens generated.
typedef void (*TokenCallback)(int token, void *user);
int generate(LlamaModel *llama, InferState *s,
             const int *prompt, int n_prompt,
             int max_new, int eos_id,
             TokenCallback on_token, void *user);

#endif // FORWARD_H
