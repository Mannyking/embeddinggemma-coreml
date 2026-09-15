"""Assess an existing fixed (1, 128) or (1, 512) Core ML artifact against float32 fixtures."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback

import coremltools as ct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/reference-f32"
# Initial acceptance thresholds, fixed before evaluating the exported artifact.
ATOL = 1e-4
RTOL = 1e-3
MIN_COSINE = 0.9999
NORM_ATOL = 1e-4


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual, expected):
    if actual.shape != (1, 768) or not np.isfinite(actual).all():
        raise ValueError(f"Invalid embedding shape or values: {actual.shape}")
    norm = float(np.linalg.norm(actual))
    cosine = float(np.sum(actual * expected) / (norm * np.linalg.norm(expected))) if norm else 0.0
    metrics = {"max_absolute_error": float(np.max(np.abs(actual - expected))),
               "cosine": cosine, "norm": norm}
    metrics["passed"] = bool(
        np.allclose(actual, expected, atol=ATOL, rtol=RTOL)
        and abs(norm - 1.0) <= NORM_ATOL and cosine >= MIN_COSINE
    )
    return metrics


def run(package, output, report, sequence_length):
    if not package.is_dir():
        raise FileNotFoundError(package)
    metadata = json.loads((REFERENCE / "metadata.json").read_text())
    if sha256(REFERENCE / "tensors.npz") != metadata["tensors_sha256"]:
        raise ValueError("Reference tensor hash mismatch")
    report["reference_metadata_sha256"] = sha256(REFERENCE / "metadata.json")
    report["reference_revision"] = metadata["revision"]
    report["package_hashes"] = {p.relative_to(package).as_posix(): sha256(p)
                                for p in sorted(package.rglob("*")) if p.is_file()}
    if sequence_length == 128:
        with np.load(REFERENCE / "tensors.npz", allow_pickle=False) as data:
            ids = data["padded_128__input_ids"].astype(np.int32)
            masks = data["padded_128__attention_mask"].astype(np.int32)
            expected = data["padded_128__embedding"].copy()
        names = metadata["padded_128_order"]
    else:
        export = json.loads((package.parent / "report.json").read_text())
        reference = package.parent / "reference_512.npz"
        if export["status"] != "exported" or export["input_shape"] != [1, 512]:
            raise ValueError("Expected a completed 512-token export")
        if export["reference_metadata_sha256"] != report["reference_metadata_sha256"]:
            raise ValueError("Export used different reference metadata")
        if export["package_hashes"] != report["package_hashes"]:
            raise ValueError("Package differs from the recorded export")
        if sha256(reference) != export["padded_reference_sha256"]:
            raise ValueError("512-token reference hash mismatch")
        with np.load(reference, allow_pickle=False) as data:
            ids = data["input_ids"].astype(np.int32)
            masks = data["attention_mask"].astype(np.int32)
            expected = data["embedding"].copy()
        names = export["fixture_order"]
        required = [case["id"] for case in metadata["cases"] if case["token_count"] <= 512]
        if names != required:
            raise ValueError("Incomplete or reordered 512-token reference cases")
        report["padded_reference_sha256"] = export["padded_reference_sha256"]
    if (ids.shape != (len(names), sequence_length) or masks.shape != ids.shape
            or expected.shape != (len(names), 768)):
        raise ValueError("Unexpected fixture shapes")
    if not np.isfinite(expected).all():
        raise ValueError("Reference embeddings contain non-finite values")
    report["stage"] = "artifact_parity"
    print(f"Reloading the saved Core ML artifact and checking {len(names)} fixtures", flush=True)
    artifact = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    shapes = {item.name: list(item.type.multiArrayType.shape) for item in artifact.get_spec().description.input}
    if shapes != {"input_ids": [1, sequence_length], "attention_mask": [1, sequence_length]}:
        raise ValueError(f"Model input shapes do not match requested length: {shapes}")
    outputs = []
    report["cases"] = {}
    for i, name in enumerate(names):
        actual = np.asarray(artifact.predict({"input_ids": ids[i:i+1],
                                              "attention_mask": masks[i:i+1]})["embedding"])
        metrics = compare(actual, expected[i:i+1])
        report["cases"][name] = metrics
        outputs.append(actual[0])
        print(f"{name}: {metrics}", flush=True)
    outputs = np.stack(outputs)
    np.save(output / "coreml_embeddings.npy", outputs)
    query = names.index("query_mars")
    documents = metadata["retrieval"]["documents"]
    scores = outputs[[names.index(name) for name in documents]] @ outputs[query]
    ranking = [documents[i] for i in np.argsort(-scores)]
    report["retrieval"] = {"scores": scores.tolist(), "ranking": ranking}
    if ranking != metadata["retrieval"]["ranking"] or not all(m["passed"] for m in report["cases"].values()):
        raise ValueError("Exported artifact failed reference parity; see report.json")
    print(f"Core ML artifact parity passed. Output: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Path to the saved .mlpackage")
    parser.add_argument("--sequence-length", type=int, choices=(128, 512), default=128)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    output = (args.output_dir or ROOT / f"artifacts/coreml-f32-{args.sequence_length}-assessment").resolve()
    # if output.exists():
    #     raise FileExistsError(f"Choose a new assessment output directory: {output}")
    report = {
        "status": "running", "stage": "verify_inputs", "package": str(package),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("coremltools", "numpy")},
        "script_sha256": sha256(Path(__file__)), "compute_units": "CPU_ONLY",
        "thresholds": {"atol": ATOL, "rtol": RTOL, "minimum_cosine": MIN_COSINE,
                       "unit_norm_atol": NORM_ATOL},
        "limitations": "Only fixtures fitting the selected length; no beyond-window or device validation.",
    }
    output.mkdir(parents=True)
    try:
        report["input_shape"] = [1, args.sequence_length]
        run(package, output, report, args.sequence_length)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
