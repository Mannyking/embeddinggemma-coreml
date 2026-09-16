# Working Context

## Purpose

This repository is an isolated experiment for turning Google's
`embeddinggemma-300m` checkpoint into a Core ML text embedder. The immediate
goal is numerical agreement with the original model on macOS. iOS deployment,
latency, memory use, and Neural Engine placement are later work.

Tokenization and EmbeddingGemma's query/document prompts remain outside the
Core ML package. The package accepts already-tokenized `input_ids` and an
`attention_mask`, then returns one 768-dimensional normalized embedding.

## Terms

- **Original model:** The local official SentenceTransformers/PyTorch model.
- **Smoke test:** A short, terminal-only sanity check of the original model.
- **Baseline:** Saved known-good original-model inputs and embeddings. It is
  the answer key for later Core ML comparisons.
- **Export:** Conversion of a checked PyTorch wrapper into an `.mlpackage`.
  An exported package is not yet proven equivalent to the original model.
- **Assessment:** Loading a saved `.mlpackage`, running its baseline inputs,
  and comparing its actual outputs to expected embeddings.
- **f32:** Float32 computation and model weights for this experiment.
- **mixed f16:** A Core ML conversion that uses FP16 for `gather` and `linear`
  operations only. Attention, softmax, normalization, residual arithmetic, and
  pooling remain float32. Inputs remain int32, embeddings are returned as
  float32, and the reference pipeline remains float32.

## Evidence Flow

```text
official local model
  ├─ smoke test: quick behavior check; prints only
  └─ baseline capture: saves expected inputs and embeddings
       ├─ 128-token export: creates a fixed-shape package
       └─ 512-token export: creates a fixed-shape package and padded fixtures
            └─ assessment: runs the saved package and records parity results
```

Each stage answers a different question. A passing original-model check does
not prove that a converted Core ML package behaves the same way. Likewise, a
successful export does not prove artifact parity until the saved package has
been assessed.

## Current Verified State

- The original model baseline is pinned to revision
  `57c266a740f537b4dc058e1b0cda161fd15afa75`.
- The 128-token package passed CPU-only parity assessment for eight short
  fixtures. Its largest recorded elementwise error is approximately `3.73e-7`.
- The 512-token package passed CPU-only parity assessment for ten fixtures,
  including exact 511- and 512-token cases. Its largest recorded elementwise
  error is approximately `4.34e-7`.

These results establish parity for the recorded fixtures on macOS CPU only.
They do not establish behavior beyond the selected input length, iOS behavior,
performance, memory use, or Neural Engine execution.

## Mixed-FP16 Result

All-FP16 conversion is known to be unstable for this model: the original
blocked-mask sentinel overflows to negative infinity in FP16, and a
finite-sentinel all-FP16 experiment still became non-finite by the final RMS
norm. The failed trial artifacts and one-off diagnostic tools were removed once
they established that the mixed policy is required.

The accepted candidate uses the `mixed` policy: only Core ML `gather` and
`linear` operations are selected for FP16 conversion; attention, softmax, RMS
normalization, residual arithmetic, and pooling remain float32. It also uses a
finite blocked-attention sentinel of `-65504`, preserving the upstream allowed
and blocked positions without creating FP16 negative infinity.

Its saved package is in `artifacts/coreml/f16-512-mixed/`. CPU assessment over
all ten fixtures found finite, unit-normalized embeddings, unchanged Mars
retrieval ranking, cosine similarity from `0.999911` to `0.999960`, and maximum
absolute error from `0.00136` to `0.00311`. The package is about 592 MB, versus
about 1.2 GB for F32.

The strict `f32-parity` profile deliberately rejects that artifact because it
requires F32-level numerical agreement. The separately approved `mixed-f16`
profile retains cosine at least `0.9999`, norm error at most `1e-4`, unchanged
retrieval ranking, and adds a maximum absolute error limit of `0.01`. Use it
when assessing the mixed candidate:

```sh
.venv/bin/python scripts/coreml/assess_f32.py \
  artifacts/coreml/f16-512-mixed/EmbeddingGemmaF16.mlpackage \
  --sequence-length 512 \
  --quality-profile mixed-f16
```

The export layout is intentionally explicit:

- `scripts/coreml/export_512_common.py` contains the checked model wrapper,
  finite-mask rule, reference capture, trace validation, and Core ML conversion.
- `scripts/coreml/export_f32_512.py` selects F32 conversion.
- `scripts/coreml/export_f16_512.py` selects the accepted mixed-FP16 conversion.

## Input-Length Variants

The original model accepts up to 2,048 tokens. Core ML exports are intentionally
fixed shape:

| Variant | Inputs | Output | Intended validation |
| --- | --- | --- | --- |
| 128 | `(1, 128)` IDs and mask | `(1, 768)` embedding | Short fixtures |
| 512 | `(1, 512)` IDs and mask | `(1, 768)` embedding | Short plus 511/512-token fixtures |

The token count includes the chosen prompt and special tokens. Shorter inputs
are right-padded. Inputs longer than a package's fixed length must be rejected
or chunked by the caller; they must never be silently truncated for evaluation.

## Source Model Identity

`model-source.json` is the model-source receipt. It records
the upstream repository, immutable revision, and hashes of every downloaded
model, configuration, and tokenizer file. The baseline and exports verify those
hashes before inference.

## Current Artifacts

- `artifacts/baseline-f32/`: the saved original-model baseline. `fixtures.npz`
  contains many named NumPy arrays; `metadata.json` describes their origin.
- `artifacts/coreml/f32-128/`: the 128 package and its initial assessment in
  `assessments/initial/`. The original run predates a separately retained
  export report.
- `artifacts/coreml/f32-512/`: the 512 package, `export-report.json`,
  `padded-fixtures.npz`, and its initial assessment in `assessments/initial/`.
- `artifacts/coreml/f16-512-mixed/`: the accepted mixed-FP16 package and its
  assessment evidence.

Every new assessment belongs in a fresh subdirectory beneath the package's
`assessments/` directory. The export report records conversion; the assessment
report records results from loading and predicting with the saved package.

## Documentation Roles

- `README.md` is the concise project overview and user-facing quick start.
- This file is the working map: terminology, evidence, artifact roles, and
  known limits.
- `AGENTS.md` contains compact operational rules for people and coding agents
  changing the experiment.
