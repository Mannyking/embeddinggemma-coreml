# EmbeddingGemma → Core ML

Standalone experiment to convert Google's official `embeddinggemma-300m`
checkpoint into a Core ML text embedder for Tinkeroom on iOS. It is intentionally
independent from the app: this repository establishes whether the model can be
converted correctly before any product integration or device optimization.

## The Idea

Figuring out whether the model can be converted correctly before it is
integrated into any products/applications.

The experiment treats the original model as the source of truth. It captures
known-good embeddings from that model, converts the full embedding pipeline to
Core ML, then checks that the saved Core ML package reproduces those embeddings.

```text
original model → baseline → Core ML package → parity assessment
```

The Core ML package receives already-tokenized IDs and an attention mask. The
caller remains responsible for tokenization and query/document prompts; the
package returns one normalized 768-dimensional embedding.

## Results and Evaluation

All variants below use a fixed 512-token input. Sizes are saved model packages.

| Variant | Saved size |
| --- | ---: |
| Core ML F32 (iOS 18) | 1,235.3 MB |
| Core ML mixed FP16 | 620.4 MB |
| Core ML int4 / attention int8 (iOS 18) | 216.5 MB |
| LiteRT mixed precision | 179.1 MB |

### BEIR SciFact Retrieval

The F32, int4/int8 Core ML, and LiteRT variants were evaluated across 252
eligible claims and 4,799 abstracts. Records above the fixed 512-token limit
were excluded.

| Model | Saved size | Recall@1 | Recall@10 | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core ML F32 | 1,235.3 MB | 0.5742 | 0.8878 | 0.7488 | 0.7102 |
| Core ML int4 / attention int8 | 216.5 MB | 0.5671 | 0.8882 | 0.7414 | 0.7037 |
| LiteRT mixed precision | 179.1 MB | 0.5603 | 0.8910 | 0.7373 | 0.6958 |

LiteRT's embeddinggemma version was included as it's the current model used on the Android side.

## Platform Compatibility

The normal F32 and mixed-FP16 Core ML packages require iOS/iPadOS 15 or
newer, or macOS 12 or newer. The F32 iOS 18 source package and the int4
candidate require iOS/iPadOS 18 or newer, or macOS 15 or newer. The same model
package works across those Apple platforms; the newer target is required for
the int4 compression feature.

The conversion and validation details are recorded in `CONTEXT.md`.

## Reproduce the Findings and Packages

This requires macOS, Python 3.11, `uv`, and access to the gated Google
checkpoint on Hugging Face. Model binaries, generated packages, baselines,
and assessments are not in Git.

```sh
uv sync --locked --group dev
# After accepting the Google checkpoint terms on Hugging Face:
uv run hf auth login
uv run hf download google/embeddinggemma-300m \
  --revision 57c266a740f537b4dc058e1b0cda161fd15afa75 \
  --local-dir models/embeddinggemma-300m

uv run python scripts/original/smoke_test_f32.py
uv run python scripts/original/capture_baseline_f32.py

uv run python scripts/coreml/export_f32_512.py \
  --output-dir artifacts/coreml/f32-512
uv run python scripts/coreml/assess_f32.py \
  artifacts/coreml/f32-512/EmbeddingGemmaF32.mlpackage \
  --sequence-length 512 \
  --output-dir artifacts/coreml/f32-512/assessments/initial
```

The above scripts ensure you download the tested OG-model after which a quick test
and baseline capture follows. A 512 input-token model is created and then assessed
with the final command. Rest of the scripts work similarly.

## Further Reading

- [CONTEXT.md](CONTEXT.md): workflow, current evidence, artifacts, and limits.
- `AGENTS.md`: preservation rules for the agent.
- `models/receipts/`: pinned source and local-file hashes for the original and
  LiteRT comparison models.

## Sources

- [Official checkpoint](https://huggingface.co/google/embeddinggemma-300m)
- [Google inference guidance](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
- [Core ML Tools](https://github.com/apple/coremltools)
