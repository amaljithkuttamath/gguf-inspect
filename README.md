[![CI](https://github.com/amaljithkuttamath/gguf-inspect/actions/workflows/ci.yml/badge.svg)](https://github.com/amaljithkuttamath/gguf-inspect/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

# gguf-inspect

A fast Rust CLI that reads GGUF model files and prints useful metadata about the model architecture, tensors, and quantization.

## Example output

```
GGUF Model Summary
==================================================
  Model name:       Llama 3.1 8B Instruct
  Architecture:     llama
  GGUF version:     3
  Tensor count:     291
  Parameters:       8.03B
  Context length:   131072
  Embedding size:   4096
  Layers:           32
  Attention heads:  32
  KV heads:         8
  Vocab size:       128256
  Quantization:     Q4_K_M
  File size:        4.92 GB
  Tensor data size: 4.58 GB
  Est. memory:      5.04 GB (tensors + ~10% overhead)
```

## Install

### From source

```bash
cargo install --path .
```

### From releases

Download a prebuilt binary from the [releases page](https://github.com/amaljithkuttamath/gguf-inspect/releases).

## Usage

```bash
# Print model summary
gguf-inspect model.gguf

# Show all metadata key-value pairs
gguf-inspect model.gguf --metadata

# List all tensors with shapes and sizes
gguf-inspect model.gguf --tensors

# JSON output for scripting
gguf-inspect model.gguf --json

# Combine flags
gguf-inspect model.gguf --metadata --tensors
```

## What's in a GGUF file?

GGUF (GPT-Generated Unified Format) is the standard binary format for quantized LLM weights, used by llama.cpp, Ollama, and similar inference tools. A single `.gguf` file contains everything needed to load a model:

1. **Header**: magic number (`GGUF`), format version, and counts for metadata and tensors.
2. **Metadata**: key-value pairs describing the model architecture (layer count, embedding size, context length, vocab, etc.) and tokenizer configuration.
3. **Tensor info**: a directory of all weight tensors, listing each tensor's name, shape, data type (F16, Q4_K, Q8_0, etc.), and byte offset into the file.
4. **Tensor data**: the raw weight data, packed according to each tensor's quantization type.

Quantization reduces model size by storing weights in lower precision. For example, Q4_K uses roughly 4.5 bits per weight instead of 16 (F16) or 32 (F32), cutting file size by 3-7x with minimal quality loss. This is what makes it practical to run large language models on consumer hardware.

## License

MIT
