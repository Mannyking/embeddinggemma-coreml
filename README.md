# EmbeddingGemma → Core ML

Standalone experiment to convert Google's official `embeddinggemma-300m`
checkpoint into a Core ML text embedder for Cairn on iOS.
Created 2026-09-12. Dependencies are recorded in `pyproject.toml` and `uv.lock`;

## OG GemmaEmbedding (f32) check

Run from this directory using the prepared environment:

```sh
.venv/bin/python scripts/assess_embeddinggemma_f32.py
```

This reusable offline check loads the local model on CPU in float32, checks
embedding shape, finite values and unit norms, and verifies a simple retrieval
ranking. Model revision and artifact hashes are recorded separately in `provenance/model-manifest.json`; dependencies are locked in `uv.lock`.

## Context and goal

The existing app is at `../../ios-dev/CairnSpike`; read its
`EMBEDDING_SCOPE.md` for background. Keep this experiment independent of that
app. Generation there remains LiteRT-LM. EmbeddingGemma is the primary embedding
candidate; Apple's Natural Language `NLEmbedding` is a secondary alternative.

A community Core ML conversion exists from john-rocky / mlboydaisuke. Source
inspection found explicit bidirectional attention, but its parity test compares
an unrescaled custom PyTorch model rather than the exported Core ML artifact.
The converter applies additional residual rescaling and can continue after
missing/mismatched weights. Its published configuration specifies 128-token
inputs and INT8 weights; documentation has conflicting precision and OS claims.
These are reasons to validate independently, not proof that its artifact fails.
We can use its source as a reference without adopting its library or conversion.

## Steps and acceptance

1. **Reference:** pin Google's official checkpoint revision, license/terms,
   tokenizer, module configuration, and dependency versions. Record artifact
   hashes. Use an isolated Python environment; keep weights out of Git.
2. **Expected results:** run the supported reference implementation and save
   token IDs, masks, and full 768-dimensional embeddings for a small fixture set.
   Include query/document prefixes, padding, short/long text, and multilingual
   examples. Define empty-input and over-limit behavior.
3. **Baseline export:** attempt coremltools conversion of the complete backbone,
   masked pooling, dense projections, and normalization. Start with one fixed
   input length. Tokenization and prefixes may remain outside Core ML. No
   retraining; avoid custom numerical rewrites until an observed failure warrants
   them. Do not assume conversion or Neural Engine placement will work directly.
4. **Actual artifact parity:** run the exported Core ML model on the Mac with
   identical reference inputs. Check shape, finite values, nonzero/unit norm,
   per-example vector agreement, and retrieval rankings. Set numerical tolerances
   explicitly. Validate bidirectional global/sliding masks, including window
   boundaries when testing longer inputs. Test the exported artifact after every
   precision or numerical change; a semantic smoke test alone is insufficient.
5. **Device evaluation:** only after parity, manually measure loading, latency,
   memory, and actual hardware execution on iPhone. Optimize precision and input
   length buckets incrementally. Integrate into Cairn only in a separate step.

First milestone: **a correct Core ML embedding on the Mac**, not maximum Neural
Engine utilization. Record exact tools and OS requirements
from the successful experiment rather than assuming a deployment floor.

## Sources

- [Official checkpoint](https://huggingface.co/google/embeddinggemma-300m)
- [Google inference guidance](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
- [Core ML Tools](https://github.com/apple/coremltools)
- [Community converter reviewed at 5ef6b301](https://github.com/john-rocky/CoreML-LLM/tree/5ef6b301d3a3d628e25c0605479f59dbf3a7d955/conversion)
- [Community artifact](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-coreml)
