"""Assess the local LiteRT 512-token model against the pinned F32 baseline.

The LiteRT graph accepts only a padded int32 token-ID tensor. This assessment
feeds the exact prompted, right-padded IDs captured by the assessed F32 iOS 18
source export; no tokenizer, prompt, attention mask, or truncation behavior is
introduced here.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback

from ai_edge_litert.interpreter import Interpreter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts/baseline-f32"
F32_SOURCE = ROOT / "artifacts/coreml/f32-512-ios18"
DEFAULT_MODEL = ROOT / "models/embeddinggemma-300M_seq512_mixed-precision.tflite"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual, expected):
    if actual.shape != (1, 768):
        raise ValueError(f"Expected embedding shape (1, 768), got {actual.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("LiteRT returned a non-finite embedding")
    norm = float(np.linalg.norm(actual))
    cosine = float(np.sum(actual * expected) / (norm * np.linalg.norm(expected))) if norm else 0.0
    return {
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
        "cosine": cosine,
        "norm": norm,
    }


def load_fixtures(report):
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    fixtures_path = BASELINE / "fixtures.npz"
    expected_fixtures_hash = metadata.get("fixtures_sha256", metadata.get("tensors_sha256"))
    if sha256(fixtures_path) != expected_fixtures_hash:
        raise ValueError("Baseline fixture hash mismatch")
    source_report_path = F32_SOURCE / "export-report.json"
    padded_path = F32_SOURCE / "padded-fixtures.npz"
    source_report = json.loads(source_report_path.read_text())
    if source_report.get("status") != "exported" or source_report.get("input_shape") != [1, 512]:
        raise ValueError("Expected completed iOS 18 F32 512-token source export")
    if source_report.get("baseline_metadata_sha256") != sha256(BASELINE / "metadata.json"):
        raise ValueError("F32 source export used different baseline metadata")
    if sha256(padded_path) != source_report.get("padded_fixtures_sha256"):
        raise ValueError("F32 source padded fixtures differ from their recorded hash")
    with np.load(padded_path, allow_pickle=False) as data:
        ids = data["input_ids"].astype(np.int32)
        expected = data["embedding"].copy()
    names = source_report["fixture_order"]
    required = [case["id"] for case in metadata["cases"] if case["token_count"] <= 512]
    if names != required or ids.shape != (len(names), 512) or expected.shape != (len(names), 768):
        raise ValueError("Unexpected or incomplete padded baseline fixtures")
    report.update({
        "baseline_metadata_sha256": sha256(BASELINE / "metadata.json"),
        "baseline_revision": metadata["revision"],
        "f32_source_export_report_sha256": sha256(source_report_path),
        "padded_fixtures_sha256": source_report["padded_fixtures_sha256"],
        "fixture_order": names,
        "input_provenance": "Exact prompted, right-padded input_ids from F32 source padded-fixtures.npz",
    })
    return metadata, names, ids, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/litert/tflite-512-mixed-precision/assessments/initial")
    args = parser.parse_args()
    model = args.model.resolve()
    output = args.output_dir.resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if output.exists():
        raise FileExistsError(f"Choose a new assessment output directory: {output}")
    report = {
        "status": "running", "stage": "verify_inputs", "model": str(model),
        "model_sha256": sha256(model), "model_size_bytes": model.stat().st_size,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("ai-edge-litert", "numpy")},
        "script_sha256": sha256(Path(__file__)),
        "comparison": "LiteRT output versus pinned F32 baseline; metrics are descriptive, not an acceptance claim.",
        "limitations": "Ten fixtures through 512 tokens; no device benchmark or broader retrieval evaluation.",
    }
    output.mkdir(parents=True)
    try:
        metadata, names, ids, expected = load_fixtures(report)
        report["stage"] = "load_model"
        interpreter = Interpreter(model_path=str(model))
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        report["input_details"] = [{"name": item["name"], "shape": item["shape"].tolist(),
                                    "dtype": np.dtype(item["dtype"]).name}
                                   for item in inputs]
        report["output_details"] = [{"name": item["name"], "shape": item["shape"].tolist(),
                                     "dtype": np.dtype(item["dtype"]).name}
                                    for item in outputs]
        if len(inputs) != 1 or tuple(inputs[0]["shape"]) != (1, 512) or inputs[0]["dtype"] != np.int32:
            raise ValueError(f"Unexpected LiteRT input contract: {report['input_details']}")
        if len(outputs) != 1 or tuple(outputs[0]["shape"]) != (1, 768) or outputs[0]["dtype"] != np.float32:
            raise ValueError(f"Unexpected LiteRT output contract: {report['output_details']}")
        report["stage"] = "compare"
        embeddings = []
        report["cases"] = {}
        for index, name in enumerate(names):
            interpreter.set_tensor(inputs[0]["index"], ids[index:index + 1])
            interpreter.invoke()
            actual = np.asarray(interpreter.get_tensor(outputs[0]["index"]))
            report["cases"][name] = metrics(actual, expected[index:index + 1])
            embeddings.append(actual[0])
            print(f"{name}: {report['cases'][name]}", flush=True)
        embeddings = np.stack(embeddings)
        np.save(output / "litert_embeddings.npy", embeddings)
        query = names.index("query_mars")
        documents = metadata["retrieval"]["documents"]
        scores = embeddings[[names.index(name) for name in documents]] @ embeddings[query]
        ranking = [documents[index] for index in np.argsort(-scores)]
        report["retrieval"] = {
            "scores": scores.tolist(), "ranking": ranking,
            "matches_f32_baseline_ranking": ranking == metadata["retrieval"]["ranking"],
        }
        report["status"] = "assessed"
        print(f"LiteRT assessment saved: {output}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
