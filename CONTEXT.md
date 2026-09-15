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

Every new assessment belongs in a fresh subdirectory beneath the package's
`assessments/` directory. The export report records conversion; the assessment
report records results from loading and predicting with the saved package.

## Documentation Roles

- `README.md` is the concise project overview and user-facing quick start.
- This file is the working map: terminology, evidence, artifact roles, and
  known limits.
- `AGENTS.md` contains compact operational rules for people and coding agents
  changing the experiment.
