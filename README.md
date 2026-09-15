# EmbeddingGemma → Core ML

Standalone experiment to convert Google's official `embeddinggemma-300m`
checkpoint into a Core ML text embedder for Cairn on iOS. It is intentionally
independent from the app: this repository establishes whether the model can be
converted correctly before any product integration or device optimization.

## The Idea

The experiment treats the original model as the source of truth. It captures
known-good embeddings from that model, converts the full embedding pipeline to
Core ML, then checks that the saved Core ML package reproduces those embeddings.

```text
original model → baseline → Core ML package → parity assessment
```

The Core ML package receives already-tokenized IDs and an attention mask. The
caller remains responsible for tokenization and query/document prompts; the
package returns one normalized 768-dimensional embedding.

## Scope

The current work is about conversion correctness on macOS. It does not yet make
claims about iOS deployment, latency, memory use, Neural Engine placement, or
the best input length for product content.

## Further Reading

- [CONTEXT.md](CONTEXT.md): workflow, current evidence, artifacts, and limits.
- `AGENTS.md`: preservation rules for contributors and coding agents.
- `model-source.json`: the pinned upstream model revision and local-file hashes.

## Sources

- [Official checkpoint](https://huggingface.co/google/embeddinggemma-300m)
- [Google inference guidance](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
- [Core ML Tools](https://github.com/apple/coremltools)
