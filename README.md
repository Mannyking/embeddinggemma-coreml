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

## Current Result

The current 512-token mixed-FP16 package is about **592 MB**, compared with
about **1.2 GB** for the float32 package.

The retained iOS 18 post-training 4-bit (linear-int4/int8-attention) package
is about 216 MB.

## Platform Compatibility

The existing F32 and mixed-FP16 Core ML packages require iOS/iPadOS 15 or
newer, or macOS 12 or newer. The F32 iOS 18 source package and the int4
candidate require iOS/iPadOS 18 or newer, or macOS 15 or newer. The same model
package works across those Apple platforms; the newer target is required for
the int4 compression feature.

The conversion and validation details are recorded in `CONTEXT.md`.

## Further Reading

- [CONTEXT.md](CONTEXT.md): workflow, current evidence, artifacts, and limits.
- `AGENTS.md`: preservation rules for the agent.
- `model-source.json`: the pinned upstream model revision and local-file hashes.

## Sources

- [Official checkpoint](https://huggingface.co/google/embeddinggemma-300m)
- [Google inference guidance](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
- [Core ML Tools](https://github.com/apple/coremltools)
