# Real Model Analysis: Llama 3.2 3B Instruct (Q4_K_M)

Analysis of the Llama 3.2 3B Instruct model quantized to Q4_K_M, as distributed by Ollama.

## Summary

```
GGUF Model Summary
==================================================
  Model name:       Llama 3.2 3B Instruct
  Architecture:     llama
  GGUF version:     3
  Tensor count:     255
  Parameters:       3.21B
  Context length:   131072
  Embedding size:   3072
  Layers:           28
  Attention heads:  24
  KV heads:         8
  Vocab size:       128256
  Quantization:     Q4_K_M
  File size:        1.88 GB
  Tensor data size: 1.87 GB
  Est. memory:      2.06 GB (tensors + ~10% overhead)
```

Key observations:
- 3.21B parameters compressed into 1.88 GB on disk via mixed quantization.
- The estimated memory footprint (2.06 GB) fits comfortably on a 4 GB device.
- 131072 context length, though actual usable context depends on available KV-cache memory.

## Architecture Metadata

| Key | Value |
|-----|-------|
| general.architecture | llama |
| general.name | Llama 3.2 3B Instruct |
| general.basename | Llama-3.2 |
| general.size_label | 3B |
| llama.block_count | 28 |
| llama.context_length | 131072 |
| llama.embedding_length | 3072 |
| llama.feed_forward_length | 8192 |
| llama.attention.head_count | 24 |
| llama.attention.head_count_kv | 8 |
| llama.attention.key_length | 128 |
| llama.attention.value_length | 128 |
| llama.rope.freq_base | 500000 |
| llama.vocab_size | 128256 |
| general.file_type | 15 (Q4_K_M) |
| tokenizer.ggml.model | gpt2 |
| tokenizer.ggml.pre | llama-bpe |

## Quantization Distribution

The Q4_K_M file type uses a mixed quantization strategy. Not all tensors get the same bit width:

| Type | Count | Role |
|------|-------|------|
| Q4_K | 168 | Bulk of the weights (attention projections, FFN gates/ups) |
| F32 | 58 | Normalization layers (RMSNorm), rope frequencies |
| Q6_K | 29 | Token embeddings, FFN down projections, attention V projections |

This reveals the Q4_K_M strategy: keep most weights at ~4.5 bits per element, but use higher precision (Q6_K at ~6.5 bits) for the most sensitive tensors. Normalization weights stay at full F32 since they are tiny (one vector per layer) and quantizing them hurts quality disproportionately.

## Block 0 Tensor Layout

Each transformer block follows this pattern:

| Tensor | Shape | Type | Size |
|--------|-------|------|------|
| blk.0.attn_q.weight | 3072 x 3072 | Q4_K | 5.06 MB |
| blk.0.attn_k.weight | 3072 x 1024 | Q4_K | 1.69 MB |
| blk.0.attn_v.weight | 3072 x 1024 | Q6_K | 2.46 MB |
| blk.0.attn_output.weight | 3072 x 3072 | Q4_K | 5.06 MB |
| blk.0.attn_norm.weight | 3072 | F32 | 12.00 KB |
| blk.0.ffn_gate.weight | 3072 x 8192 | Q4_K | 13.50 MB |
| blk.0.ffn_up.weight | 3072 x 8192 | Q4_K | 13.50 MB |
| blk.0.ffn_down.weight | 8192 x 3072 | Q6_K | 19.69 MB |
| blk.0.ffn_norm.weight | 3072 | F32 | 12.00 KB |

The per-block weight budget is roughly 61 MB. Across 28 blocks, that is about 1.7 GB, with the remaining ~170 MB in the token embedding and output layers.

## What GQA Looks Like in Tensor Shapes

Grouped Query Attention (GQA) reduces KV-cache memory by sharing key/value heads across multiple query heads. This model uses 24 query heads and 8 KV heads (a 3:1 ratio).

The shapes make this visible:

- **Q projection**: 3072 x 3072 (24 heads * 128 dim per head = 3072)
- **K projection**: 3072 x 1024 (8 heads * 128 dim per head = 1024)
- **V projection**: 3072 x 1024 (8 heads * 128 dim per head = 1024)
- **Output projection**: 3072 x 3072 (maps concatenated heads back to embedding dim)

The K and V projections are 3x smaller than the Q projection. This means each KV head serves 3 query heads. During inference, the KV-cache stores 1024-dimensional vectors per token instead of 3072, cutting cache memory by 3x compared to standard multi-head attention.

At this model's 128K context length, the KV-cache savings from GQA are substantial: storing 128K tokens of KV state at 1024 dimensions (FP16) requires ~512 MB, versus ~1.5 GB without GQA.

## Per-Tensor Size Breakdown

The largest tensors in the model:

1. **token_embd.weight** (3072 x 128256, Q6_K): 308.23 MB. The token embedding table. Large because the vocabulary has 128K entries.
2. **blk.*.ffn_down.weight** (8192 x 3072, Q6_K): 19.69 MB each, 28 of them = 551.3 MB total. These are quantized at Q6_K for better quality.
3. **blk.*.ffn_gate.weight** and **blk.*.ffn_up.weight** (3072 x 8192, Q4_K): 13.50 MB each, 56 of them = 756.0 MB total.
4. **blk.*.attn_q.weight** and **blk.*.attn_output.weight** (3072 x 3072, Q4_K): 5.06 MB each.
5. **blk.*.attn_k.weight** and **blk.*.attn_v.weight** (3072 x 1024): 1.69 MB (Q4_K) / 2.46 MB (Q6_K) each.

The FFN layers dominate, accounting for roughly 70% of the total model weight. This is typical for transformer architectures where the feed-forward network is 2-3x wider than the embedding dimension.
