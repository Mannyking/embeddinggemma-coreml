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

## Float32 reference fixtures

The initial assessment passed with Mars ranked first (0.645413), followed by
Venus (0.259444) and Saturn (0.232551), as reported from the manual run.

```sh
.venv/bin/python scripts/capture_reference_f32.py
```

This captures token IDs, attention masks and full embeddings in
`artifacts/reference-f32/tensors.npz`, with text, prefixes, exact environment,
hashes and retrieval rankings in `metadata.json`. It covers multilingual inputs,
511/512/513-token boundaries, the 2048-token limit, and a batch padded to 128
tokens for the first fixed-length export. It checks the full module pipeline
against the public encode API and compares padded with unpadded results.

Our input policy rejects empty/whitespace-only content and inputs longer than
2048 tokens including the prefix and special tokens, without truncation.
The script verifies model hashes before inference and refuses to overwrite an
existing fixture directory. The saved fixture set has been inspected: all 12
cases are present, embeddings are finite, and the tensor hash matches. These
reference checks do not establish Core ML parity or validate attention masks
independently; those remain part of artifact evaluation.

## First Core ML export

```sh
.venv/bin/python scripts/export_embeddinggemma_f32.py
```

This attempts a complete float32 ML Program with two int32 inputs of shape
`(1, 128)` (token IDs and attention mask), producing one `(1, 768)` embedding.
It checks attention masks, the wrapper and the trace against the reference,
then saves the Core ML package. Artifact comparison is a separate step. Tokenization and task prefixes remain outside the artifact.

Outputs go to `artifacts/coreml-f32-128/`. `report.json` records the stage,
errors, versions and package hashes. Successful export reports `exported`,
which does not indicate artifact parity. Existing outputs are
never overwritten.

## Assess the saved Core ML artifact

Run independently of conversion, using the existing verified package:

```sh
.venv/bin/python scripts/assess_embeddinggemma_coreml_f32.py \
  artifacts/coreml-f32-128-verified/EmbeddingGemmaF32.mlpackage
```

For a new export, pass its `.mlpackage` path instead. The assessor loads the
saved artifact on CPU and compares all eight short fixtures, including vector
agreement, normalization and retrieval ranking. It does not load the PyTorch
reference model or rerun conversion. Results and embeddings go to
`artifacts/coreml-f32-128-assessment/`; use `--output-dir` to choose a fresh
location on subsequent runs. Existing reports are preserved.

Initial artifact thresholds are elementwise `atol=1e-4`,
`rtol=1e-3`, cosine agreement at least `0.9999`, and norm error at most `1e-4`;
the three-document retrieval ranking must also match. Thresholds are provisional
experiment criteria, not established quality guarantees.

The existing `coreml-f32-128-verified` artifact passed the original combined
export/assessment run, with maximum absolute error approximately `3.73e-7`. Passing these short fixtures does not validate longer inputs,
sliding-window boundaries, iOS deployment or Neural Engine placement.

## 512-token variant

Export separately, then assess the saved package:

```sh
.venv/bin/python scripts/export_embeddinggemma_f32_512.py
.venv/bin/python scripts/assess_embeddinggemma_coreml_f32.py \
  artifacts/coreml-f32-512/EmbeddingGemmaF32.mlpackage --sequence-length 512
```

The exporter uses the same original weights and produces fixed `(1, 512)`
inputs. It prepares ten fixtures (eight short examples plus 511 and 512 tokens),
right-pads their saved token IDs, and runs the original pipeline on those exact
inputs. These reference vectors are saved in `reference_512.npz` alongside the
package, with their hash and ordering in the export report. The assessor verifies
these records before comparing Core ML outputs. Keep the export directory intact.

Transformers converts the configured bidirectional window of 512 into an
exclusive distance bound of 257 (up to 256 positions in each direction).
The 512 exporter therefore uses distinct full and sliding attention masks,
and checks both against upstream for every fixture before tracing.
Inputs longer than 512 are
excluded, never truncated to fit. The original 128 exporter remains available;
the assessor defaults to 128 for existing commands.

The current saved 512-token package has been exported and independently passed
the CPU-only artifact-parity assessment, including the 511- and 512-token
fixtures. Assessment outputs default to `artifacts/coreml-f32-512-assessment/`.
Both scripts accept `--output-dir` for fresh output locations and refuse to
overwrite existing directories.

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
