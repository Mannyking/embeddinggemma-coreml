"""Export a fixed (1, 512) float32 Core ML artifact with checked tracing."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import argparse
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"

import coremltools as ct
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/coreml/f32-512"
BASELINE = ROOT / "artifacts/baseline-f32"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Embedder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model[0].auto_model
        self.head = torch.nn.ModuleList(list(model.children())[1:])
        if not self.backbone.config.use_bidirectional_attention:
            raise ValueError("This wrapper requires bidirectional attention")
        positions = torch.arange(512)
        self.register_buffer("local_allowed", (
            (positions[:, None] - positions[None, :]).abs()
            < self.backbone.config.sliding_window
        )[None, None, :, :])

    def attention_masks(self, attention_mask):
        # Transformers converts the configured bidirectional window of 512 to
        # an exclusive distance bound of 257. Preserve that local restriction.
        allowed = attention_mask[:, None, None, :].to(torch.bool).expand(-1, 1, 512, -1)
        mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        local_mask = torch.where(allowed & self.local_allowed, 0.0, torch.finfo(torch.float32).min)
        return {"full_attention": mask, "sliding_attention": local_mask}

    def forward(self, input_ids, attention_mask):
        tokens = self.backbone(input_ids=input_ids, attention_mask=self.attention_masks(attention_mask),
                               use_cache=False, return_dict=False)[0]
        features = {"token_embeddings": tokens, "attention_mask": attention_mask}
        for module in self.head:
            features = module(features)
        return features["sentence_embedding"]


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Move the existing output directory before rerunning: {OUTPUT}")
    report = {
        "status": "running", "stage": "verify_inputs",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "sentence-transformers", "coremltools", "numpy")},
        "script_sha256": sha256(Path(__file__)),
        "input_shape": [1, 512], "precision": "float32", "compute_units": "CPU_ONLY",
        "limitations": "Short, 511- and 512-token fixtures; no sequences beyond 512 tokens or device validation.",
    }
    OUTPUT.mkdir(parents=True)
    try:
        run(report)
        report["status"] = "exported"
        report["artifact_parity"] = "Not assessed; run assess_embeddinggemma_coreml_f32.py"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (OUTPUT / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")


def run(report):
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    model_source_path = ROOT / "model-source.json"
    expected_source_hash = metadata.get("model_source_sha256", metadata.get("model_manifest_sha256"))
    if sha256(model_source_path) != expected_source_hash:
        raise ValueError("Baseline model-source record has changed")
    fixtures_path = BASELINE / "fixtures.npz"
    expected_fixtures_hash = metadata.get("fixtures_sha256", metadata.get("tensors_sha256"))
    if sha256(fixtures_path) != expected_fixtures_hash:
        raise ValueError("Baseline fixture hash mismatch")
    model_source = json.loads(model_source_path.read_text())
    for item in model_source["files"]:
        if sha256(ROOT / "models/embeddinggemma-300m" / item["path"]) != item["sha256"]:
            raise ValueError(f"Model hash mismatch: {item['path']}")
    report["revision"] = model_source["revision"]
    report["baseline_metadata_sha256"] = sha256(BASELINE / "metadata.json")
    cases = [case for case in metadata["cases"] if case["token_count"] <= 512]
    names = [case["id"] for case in cases]
    if metadata["padding_side"] != "right":
        raise ValueError("This fixture preparation expects right padding")
    config = json.loads((ROOT / "models/embeddinggemma-300m/config.json").read_text())
    ids = np.full((len(names), 512), config["pad_token_id"], dtype=np.int32)
    masks = np.zeros_like(ids)
    unpadded = []
    with np.load(fixtures_path, allow_pickle=False) as data:
        for i, name in enumerate(names):
            tokens = data[f"{name}__input_ids"][0]
            ids[i, :len(tokens)] = tokens
            masks[i, :len(tokens)] = data[f"{name}__attention_mask"][0]
            unpadded.append(data[f"{name}__embedding"][0])
    unpadded = np.stack(unpadded)
    inputs = [(torch.from_numpy(ids[i:i+1]), torch.from_numpy(masks[i:i+1])) for i in range(len(names))]
    report["fixture_order"] = names

    report["stage"] = "load_reference"
    print("Loading the complete float32 reference pipeline", flush=True)
    model = SentenceTransformer(
        str(ROOT / "models/embeddinggemma-300m"), device="cpu", local_files_only=True,
        model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"},
    )
    model.eval()
    model[0].auto_model.config.use_cache = False
    wrapper = Embedder(model).eval()
    with torch.no_grad():
        # Capture the unmodified upstream pipeline on precisely the padded
        # inputs used by Core ML, before testing the export wrapper.
        report["stage"] = "capture_padded_fixtures"
        expected = np.concatenate([
            model({"input_ids": token_ids, "attention_mask": mask})["sentence_embedding"].numpy()
            for token_ids, mask in inputs
        ])
        np.testing.assert_allclose(expected, unpadded, atol=1e-5, rtol=1e-4)
        if not np.isfinite(expected).all():
            raise ValueError("Non-finite reference embeddings")
        np.testing.assert_allclose(np.linalg.norm(expected, axis=1), 1.0, atol=1e-5, rtol=0)
        padded_fixtures_path = OUTPUT / "padded-fixtures.npz"
        np.savez_compressed(padded_fixtures_path, input_ids=ids, attention_mask=masks, embedding=expected)
        report["padded_fixtures_sha256"] = sha256(padded_fixtures_path)
        report["stage"] = "check_masks"
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        from transformers.models.gemma3.modeling_gemma3 import _bidirectional_window_overlay
        for _, mask in inputs:
            kwargs = dict(config=model[0].auto_model.config, input_embeds=torch.zeros(1, 512, 768),
                          attention_mask=mask, cache_position=torch.arange(512), past_key_values=None)
            original = {
                "full_attention": create_causal_mask(**kwargs, or_mask_function=lambda *args: torch.tensor(True)),
                "sliding_attention": create_sliding_window_causal_mask(
                    **kwargs, or_mask_function=_bidirectional_window_overlay(model[0].auto_model.config.sliding_window)),
            }
            for key, value in wrapper.attention_masks(mask).items():
                torch.testing.assert_close(value, original[key], atol=0, rtol=0)
        report["mask_equality"] = "Exact equality with upstream full/sliding masks for all selected fixtures"
        report["stage"] = "check_wrapper"
        for i, args in enumerate(inputs):
            np.testing.assert_allclose(wrapper(*args).numpy(), expected[i:i+1], atol=1e-5, rtol=1e-4)

        report["stage"] = "trace"
        print("Tracing fixed-shape inputs with verified tensor attention masks", flush=True)
        traced = torch.jit.trace(wrapper, inputs[0], check_trace=False)
        # Verify distinct texts and padding masks, not just the tracing example.
        report["stage"] = "check_trace"
        for i, args in enumerate(inputs):
            np.testing.assert_allclose(traced(*args).numpy(), expected[i:i+1], atol=1e-5, rtol=1e-4)

    report["stage"] = "convert"
    print("Converting to a float32 ML Program", flush=True)
    converted = ct.convert(
        traced, convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=(1, 512), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, 512), dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    package = OUTPUT / "EmbeddingGemmaF32.mlpackage"
    converted.save(str(package))
    report["package_hashes"] = {p.relative_to(package).as_posix(): sha256(p)
                                for p in sorted(package.rglob("*")) if p.is_file()}
    report["specification_version"] = converted.get_spec().specificationVersion

    print(f"Export saved: {package}. Run assess_embeddinggemma_coreml_f32.py to assess it.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output_dir.resolve()
    main()
