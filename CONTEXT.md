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

## iOS 18 4-Bit Exploration

Core ML Tools requires an iOS 18 ML Program for int4 weight compression. The
separate F32 iOS 18 source package in `artifacts/coreml/f32-512-ios18/` passed
the ten-fixture CPU-only F32-parity assessment; its maximum elementwise error
was approximately `4.34e-7`.

The retained post-training 4-bit candidate is a lossy comparison candidate;
its product acceptability requires a broader retrieval evaluation:

- `artifacts/coreml/int4-512-attention-int8/` uses signed, symmetric,
  per-block (32) linear int4 weights except for every Q/K/V/output attention
  projection, which remains int8. It is about 216 MB; the 511- and 512-token
  fixtures had cosine near `0.901` against the F32 baseline.

The compression script requires an assessed F32 iOS 18 source and records
source package, baseline, padded-fixture, and assessment hashes.

## LiteRT Comparison

The local `models/embeddinggemma-300M_seq512_mixed-precision.tflite` model is
assessed with the locked `ai-edge-litert==2.1.6` runtime through `uv`. Its
single `(1, 512)` int32 input receives the exact prompted, right-padded token
IDs used by the F32 source export; it returns one `(1, 768)` float32 embedding.

Its locked assessment is in
`artifacts/litert/tflite-512-mixed-precision/assessments/initial-uv-2-1-6/`.
Compared with the F32 baseline, short-fixture cosine is approximately
`0.967`–`0.975`; the exact 511- and 512-token fixtures are approximately
`0.859` and `0.855`. Despite that drift, the recorded Mars retrieval ordering
is unchanged.

## SciFact Retrieval Evaluation

`scripts/eval/assess_beir_scifact.py` compares the saved iOS 18 F32 package
with the int4/int8-attention candidate on BEIR's normalized SciFact test set.
It downloads data to Git-ignored `datasets/beir/` and records SHA-256 hashes
for the corpus, queries, and qrels files plus a deterministic aggregate
checksum. It writes a fresh assessment directory containing embeddings, metrics,
and per-query ranking differences.

It preserves EmbeddingGemma's query/document prompts and right padding. Empty
records and records above 512 prompted tokens are rejected. It reports Recall@1/5/10, nDCG@1/5/10, MRR@10, metric deltas, and top-ten ranking overlap.

The initial run embedded 4,799 eligible abstracts and evaluated 252 eligible
test claims. The int4/int8-attention candidate scored `0.8882` Recall@10,
`0.7414` nDCG@10, and `0.7037` MRR@10; the F32 source scored `0.8878`,
`0.7488`, and `0.7102`, respectively. This supports the candidate for this
public retrieval comparison, but is not an F32-parity or device-performance
claim.

The export layout is intentionally explicit:

- `scripts/coreml/export_512_common.py` contains the checked model wrapper,
  finite-mask rule, reference capture, trace validation, and Core ML conversion.
- `scripts/coreml/export_f32_512.py` exports F32 conversion.
- `scripts/coreml/export_f16_512.py` exports the accepted mixed-FP16 conversion.
- `scripts/coreml/export_f32_512_ios18.py` exports the assessed F32 source
  required by iOS 18 compression candidates.
- `scripts/coreml/export_int4_512_attention_int8.py` exports the compressed
  version of `f32-512-ios18` above.
- `scripts/eval/assess_beir_scifact.py` compares the saved F32 and quantized
  packages on BEIR SciFact.

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

`models/receipts/embeddinggemma-300m.json` is the original-model receipt. It
records the upstream repository, immutable revision, and hashes of every
downloaded model, configuration, and tokenizer file. Separate receipts identify
externally supplied comparison models. In particular,
`models/receipts/litert-512-mixed-precision.json` records a mutable upstream
ref but pins the assessed local file by SHA-256. The baseline and exports verify
the original-model receipt before inference, and the LiteRT fixture assessment
verifies its own receipt.

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
- `artifacts/coreml/f32-512-ios18/`: the F32 iOS 18 source package and its
  saved F32-parity assessment.
- `artifacts/coreml/int4-512-attention-int8/`: the iOS 18 linear-int4,
  attention-int8 comparison candidate and its descriptive assessment evidence.
- `artifacts/litert/tflite-512-mixed-precision/`: assessment evidence for the
  supplied mixed-precision LiteRT model; it is separate from Core ML artifacts.

Every new assessment belongs in a fresh subdirectory beneath the package's
`assessments/` directory. The export report records conversion; the assessment
report records results from loading and predicting with the saved package.

## Documentation Roles

- `README.md` is the concise project overview and user-facing quick start.
- This file is the working map: terminology, evidence, artifact roles, and
  known limits.
- `AGENTS.md` contains compact operational rules for people and coding agents
  changing the experiment.
