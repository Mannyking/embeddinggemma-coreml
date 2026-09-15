# Working Rules

Read `CONTEXT.md` before changing scripts, artifacts, or documentation. This
file is, by and large, an operational checklist.

## Preserve the Evidence Chain

- Do not edit `.npz`, `.npy`, `.mlpackage`, or JSON reports by hand.
- Do not overwrite an existing baseline, package, or assessment directory.
  Create a new named output directory for a new run.
- Keep package hashes, baseline hashes, and their linking report fields intact.
- If an artifact layout changes, move files deliberately and preserve their
  contents and recorded hashes. Explain the old-to-new mapping in documentation (commits).

## Keep Stages Distinct

- The original-model smoke test only checks that local PyTorch Gemma works.
- Baseline capture records expected original-model results.
- Export checks the wrapper and creates a Core ML package.
- Assessment must load the saved package and compare it with baseline values.

Do not describe an export as parity-verified unless a saved-artifact assessment
has passed. Do not use a smoke test as Core ML evidence.

## Model Identity and Inputs

- Verify the pinned model-source file and local model hashes before producing
  or refreshing baseline/export evidence.
- Keep model weights and generated artifacts out of Git.
- Preserve query/document prompts, token IDs, attention masks, padding side,
  and fixed input length in any comparison.
- Reject empty input and inputs above the selected model limit; do not silently
  truncate them during validation.

## Change Discipline

- Keep original-model work separate from Core ML conversion work.
- Name fixed-length variants explicitly, for example `f32-128` and `f32-512`.
- Update `CONTEXT.md` when evidence, layout, terminology, or limitations change.
- Update `README.md` only with stable, user-facing commands and conclusions.
- Run proportionate checks after edits: at minimum syntax/diff validation for
  documentation or path-only edits, and the relevant export/assessment path for
  numerical or conversion changes.
