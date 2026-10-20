#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model_loader.h"
#include <string>
#include <vector>

// ── tokenize ──
// Text -> token IDs for SentencePiece-style ("llama") vocabularies:
//   1. spaces become "▁" (U+2581) and one is prepended (add_space_prefix)
//   2. the text is split into UTF-8 characters
//   3. repeatedly merge the adjacent pair whose concatenation is a vocab token
//      with the highest score (ties: leftmost) until no merge is possible
//   4. any piece still not in the vocab falls back to byte tokens "<0xXX>"
//   5. BOS (tokenizer.ggml.bos_token_id, 1 for llama) is prepended if add_bos
// Returns an empty vector if the model has no vocab.
std::vector<int> tokenize(const Model *model, const std::string &text, bool add_bos = true);

// ── tokenizer.h ──
// Text <-> token ID conversion for SentencePiece-style ("llama") vocabularies.
// The vocab itself lives in Model (loaded from GGUF metadata).

// ── token_to_piece ──
// Converts one token ID to the raw bytes it represents:
//   - "▁" (U+2581, SentencePiece's space marker) becomes ' '
//   - byte-fallback tokens "<0xXX>" become the single byte 0xXX
//     (multi-byte UTF-8 characters arrive as several consecutive byte tokens,
//      so print/append bytes as-is and the terminal reassembles them)
//   - control / unknown / unused tokens (<s>, </s>, <unk>) become ""
//   - out-of-range IDs become ""
std::string token_to_piece(const Model *model, int token);

// ── detokenize ──
// Concatenates token_to_piece() over a whole sequence.
std::string detokenize(const Model *model, const int *tokens, int n_tokens);

#endif // TOKENIZER_H
