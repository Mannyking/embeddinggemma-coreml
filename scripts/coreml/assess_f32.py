"""Assess a Core ML artifact against float32 baseline fixtures."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback

import coremltools as ct
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts/baseline-f32"
# F32 parity is the strict control. Mixed FP16 has a separately approved
# quality target; do not loosen the F32 profile to accommodate it. Lossy
# candidates can instead use --comparison-only to record metrics without an
# invented acceptance threshold.
QUALITY_PROFILES = {
    "f32-parity": {
        "atol": 1e-4,
        "rtol": 1e-3,
        "minimum_cosine": 0.9999,
        "unit_norm_atol": 1e-4,
    },
    "mixed-f16": {
        "maximum_absolute_error": 0.01,
        "minimum_cosine": 0.9999,
        "unit_norm_atol": 1e-4,
    },
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual, expected, profile=None):
    if actual.shape != (1, 768):
        raise ValueError(f"Expected embedding shape (1, 768), got {actual.shape}")
    finite = np.isfinite(actual)
    if not finite.all():
        nan_count = int(np.isnan(actual).sum())
        inf_count = int(np.isinf(actual).sum())
        raise ValueError(
            f"Non-finite embedding values: {nan_count} NaN, {inf_count} infinity "
            f"out of {actual.size} values"
        )
    norm = float(np.linalg.norm(actual))
    cosine = float(np.sum(actual * expected) / (norm * np.linalg.norm(expected))) if norm else 0.0
    metrics = {"max_absolute_error": float(np.max(np.abs(actual - expected))),
               "cosine": cosine, "norm": norm}
    if profile is None:
        return metrics
    if "maximum_absolute_error" in profile:
        close_enough = metrics["max_absolute_error"] <= profile["maximum_absolute_error"]
    else:
        close_enough = np.allclose(actual, expected, atol=profile["atol"], rtol=profile["rtol"])
    metrics["passed"] = bool(
        close_enough
        and abs(norm - 1.0) <= profile["unit_norm_atol"]
        and cosine >= profile["minimum_cosine"]
    )
    return metrics


def run(package, output, report, sequence_length, profile=None):
    if not package.is_dir():
        raise FileNotFoundError(package)
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    fixtures_path = BASELINE / "fixtures.npz"
    expected_fixtures_hash = metadata.get("fixtures_sha256", metadata.get("tensors_sha256"))
    if sha256(fixtures_path) != expected_fixtures_hash:
        raise ValueError("Baseline fixture hash mismatch")
    report["baseline_metadata_sha256"] = sha256(BASELINE / "metadata.json")
    report["baseline_revision"] = metadata["revision"]
    report["package_hashes"] = {p.relative_to(package).as_posix(): sha256(p)
                                for p in sorted(package.rglob("*")) if p.is_file()}
    report["package_size_bytes"] = sum(p.stat().st_size for p in package.rglob("*") if p.is_file())
    if sequence_length == 128:
        with np.load(fixtures_path, allow_pickle=False) as data:
            ids = data["padded_128__input_ids"].astype(np.int32)
            masks = data["padded_128__attention_mask"].astype(np.int32)
            expected = data["padded_128__embedding"].copy()
        names = metadata["padded_128_order"]
    else:
        export = json.loads((package.parent / "export-report.json").read_text())
        padded_fixtures_path = package.parent / "padded-fixtures.npz"
        if export["status"] != "exported" or export["input_shape"] != [1, 512]:
            raise ValueError("Expected a completed 512-token export")
        export_baseline_hash = export.get("baseline_metadata_sha256", export.get("reference_metadata_sha256"))
        if export_baseline_hash != report["baseline_metadata_sha256"]:
            raise ValueError("Export used different baseline metadata")
        if export["package_hashes"] != report["package_hashes"]:
            raise ValueError("Package differs from the recorded export")
        report["export_precision"] = export["precision"]
        report["fp16_policy"] = export.get("fp16_policy")
        report["export_report_sha256"] = sha256(package.parent / "export-report.json")
        expected_padded_hash = export.get("padded_fixtures_sha256", export.get("padded_reference_sha256"))
        if sha256(padded_fixtures_path) != expected_padded_hash:
            raise ValueError("512-token padded-fixture hash mismatch")
        with np.load(padded_fixtures_path, allow_pickle=False) as data:
            ids = data["input_ids"].astype(np.int32)
            masks = data["attention_mask"].astype(np.int32)
            expected = data["embedding"].copy()
        names = export["fixture_order"]
        required = [case["id"] for case in metadata["cases"] if case["token_count"] <= 512]
        if names != required:
            raise ValueError("Incomplete or reordered 512-token baseline cases")
        report["padded_fixtures_sha256"] = expected_padded_hash
    if (ids.shape != (len(names), sequence_length) or masks.shape != ids.shape
            or expected.shape != (len(names), 768)):
        raise ValueError("Unexpected fixture shapes")
    if not np.isfinite(expected).all():
        raise ValueError("Baseline embeddings contain non-finite values")
    report["stage"] = "artifact_parity" if profile else "artifact_comparison"
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
        metrics = compare(actual, expected[i:i+1], profile)
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
    if profile and (ranking != metadata["retrieval"]["ranking"]
                    or not all(m["passed"] for m in report["cases"].values())):
        raise ValueError(f"Exported artifact failed {report['quality_profile']} acceptance; see report.json")
    if profile:
        print(f"Core ML artifact passed {report['quality_profile']} acceptance. Output: {output}", flush=True)
    else:
        print(f"Core ML artifact comparison recorded. Output: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Path to the saved .mlpackage")
    parser.add_argument("--sequence-length", type=int, choices=(128, 512), default=128)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quality-profile", choices=tuple(QUALITY_PROFILES),
                        help="Acceptance target; defaults to f32-parity unless --comparison-only is used")
    parser.add_argument("--comparison-only", action="store_true",
                        help="Record metrics and retrieval ranking without applying an acceptance threshold")
    args = parser.parse_args()
    if args.comparison_only and args.quality_profile:
        parser.error("--comparison-only cannot be used with --quality-profile")
    if not args.comparison_only and args.quality_profile is None:
        args.quality_profile = "f32-parity"
    package = args.package.resolve()
    output = (args.output_dir or package.parent / "assessments/manual").resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new assessment output directory: {output}")
    report = {
        "status": "running", "stage": "verify_inputs", "package": str(package),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("coremltools", "numpy")},
        "script_sha256": sha256(Path(__file__)), "compute_units": "CPU_ONLY",
        "comparison_mode": "descriptive" if args.comparison_only else "acceptance",
        "quality_profile": args.quality_profile,
        "thresholds": QUALITY_PROFILES.get(args.quality_profile),
        "limitations": "Only fixtures fitting the selected length; no longer-sequence or device validation.",
    }
    output.mkdir(parents=True)
    try:
        report["input_shape"] = [1, args.sequence_length]
        profile = None if args.comparison_only else QUALITY_PROFILES[args.quality_profile]
        run(package, output, report, args.sequence_length, profile)
        report["status"] = "assessed" if args.comparison_only else "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
