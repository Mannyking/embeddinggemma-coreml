"""Compress an assessed iOS 18 F32 package to int4 with int8 attention weights.

The source package must be an independently assessed F32 export because this
stage changes only saved model weights; it does not re-run the PyTorch wrapper.
All Q/K/V/output projections remain int8, matching the inspected LiteRT policy.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import traceback

import coremltools as ct
import coremltools.optimize.coreml as cto

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts/baseline-f32"
MODEL_RECEIPT = ROOT / "models/receipts/embeddinggemma-300m.json"
DEFAULT_SOURCE = ROOT / "artifacts/coreml/f32-512-ios18/EmbeddingGemmaF32.mlpackage"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_hashes(package):
    return {path.relative_to(package).as_posix(): sha256(path)
            for path in sorted(package.rglob("*")) if path.is_file()}


def verify_model_source():
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    model_source_path = MODEL_RECEIPT
    expected_source_hash = metadata.get("model_source_sha256", metadata.get("model_manifest_sha256"))
    if sha256(model_source_path) != expected_source_hash:
        raise ValueError("Baseline model-source record has changed")
    model_source = json.loads(model_source_path.read_text())
    for item in model_source["files"]:
        path = ROOT / "models/embeddinggemma-300m" / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Model hash mismatch: {item['path']}")
    return metadata, model_source


def _attribute_string(operation, attribute):
    return operation.attributes[attribute].immediateValue.tensor.strings.values[0]


def compression_ops(package):
    """Return target ops and shared constants that must remain uncompressed."""
    spec = ct.models.utils.load_spec(str(package))
    if spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("Expected an ML Program source package")
    found_attention = {}
    target_kinds = {}
    outputs = {}
    consumers = {}
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            for operation in block.operations:
                for output in operation.outputs:
                    outputs[output.name] = operation
                op_name = _attribute_string(operation, "name")
                if operation.type == "linear":
                    weight = operation.inputs["weight"].arguments[0].name
                    target_kinds[op_name] = "int8" if "_self_attn_" in weight else "int4"
                    if "_self_attn_" in weight and weight.endswith("_weight"):
                        found_attention[weight] = op_name
                elif operation.type == "gather":
                    target_kinds[op_name] = "int4"
                for binding in operation.inputs.values():
                    for argument in binding.arguments:
                        consumers.setdefault(argument.name, []).append(op_name)
    expected = {
        f"backbone_layers_{layer}_self_attn_{projection}_proj_weight"
        for layer in range(24) for projection in ("q", "k", "v", "o")
    }
    if set(found_attention) != expected:
        missing = sorted(expected - set(found_attention))
        unexpected = sorted(set(found_attention) - expected)
        raise ValueError(f"Unexpected attention linear mapping; missing={missing}, unexpected={unexpected}")
    if len([name for name in target_kinds if name.startswith("linear_")]) != 170:
        raise ValueError("Unexpected number of Core ML linear operations")
    shared_constants = []
    for output_name, operation in outputs.items():
        if operation.type != "const":
            continue
        kinds = {target_kinds[name] for name in consumers.get(output_name, []) if name in target_kinds}
        if len(kinds) > 1:
            shared_constants.append(operation.outputs[0].name)
    return dict(sorted(found_attention.items())), target_kinds, sorted(shared_constants)


def verify_source(source, assessment, report):
    if not source.is_dir():
        raise FileNotFoundError(source)
    source_report_path = source.parent / "export-report.json"
    source_fixtures = source.parent / "padded-fixtures.npz"
    if not source_report_path.is_file() or not source_fixtures.is_file():
        raise ValueError("Source package must retain export-report.json and padded-fixtures.npz")
    source_report = json.loads(source_report_path.read_text())
    source_hashes = package_hashes(source)
    if source_report.get("status") != "exported":
        raise ValueError("Source export is not complete")
    if source_report.get("precision") != "float32" or source_report.get("input_shape") != [1, 512]:
        raise ValueError("Source must be a 512-token F32 export")
    if source_report.get("minimum_deployment_target") != "iOS18":
        raise ValueError("Source must be exported with minimum deployment target iOS18")
    if source_report.get("package_hashes") != source_hashes:
        raise ValueError("Source package differs from its recorded export")
    if sha256(source_fixtures) != source_report.get("padded_fixtures_sha256"):
        raise ValueError("Source padded fixtures differ from their recorded export")
    if not assessment.is_file():
        raise FileNotFoundError(f"Missing required source assessment: {assessment}")
    assessment_report = json.loads(assessment.read_text())
    if assessment_report.get("status") != "passed" or assessment_report.get("quality_profile") != "f32-parity":
        raise ValueError("Source must pass the f32-parity assessment before compression")
    if assessment_report.get("package_hashes") != source_hashes:
        raise ValueError("Source assessment did not assess this exact package")
    metadata, model_source = verify_model_source()
    if source_report.get("baseline_metadata_sha256") != sha256(BASELINE / "metadata.json"):
        raise ValueError("Source export used different baseline metadata")
    report.update({
        "revision": model_source["revision"],
        "baseline_metadata_sha256": sha256(BASELINE / "metadata.json"),
        "source_package": str(source),
        "source_package_hashes": source_hashes,
        "source_export_report_sha256": sha256(source_report_path),
        "source_assessment": str(assessment),
        "source_assessment_sha256": sha256(assessment),
        "source_assessment_profile": assessment_report["quality_profile"],
        "padded_fixtures_sha256": source_report["padded_fixtures_sha256"],
        "fixture_order": source_report["fixture_order"],
    })
    return source_fixtures, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-assessment", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/coreml/int4-512-attention-int8")
    args = parser.parse_args()
    source = args.source_package.resolve()
    assessment = (args.source_assessment or source.parent / "assessments/initial/report.json").resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")

    report = {
        "status": "running", "stage": "verify_source",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("coremltools", "numpy")},
        "script_sha256": sha256(Path(__file__)), "compute_units": "CPU_ONLY",
        "input_shape": [1, 512], "precision": "int4-attention-int8",
        "minimum_deployment_target": "iOS18",
        "compression": {
            "method": "linear_symmetric", "granularity": "per_block", "block_size": 32,
            "default_weight_dtype": "int4", "attention_weight_dtype": "int8",
            "attention_policy": "Q/K/V/output projections in all 24 transformer layers",
        },
        "limitations": "Saved-artifact assessment required; no sequences beyond 512 tokens or device validation.",
    }
    output.mkdir(parents=True)
    try:
        source_fixtures, _ = verify_source(source, assessment, report)
        attention, target_kinds, shared_constants = compression_ops(source)
        report["attention_linear_ops"] = attention
        report["shared_constants_left_uncompressed"] = shared_constants
        report["stage"] = "compress"
        print("Loading assessed iOS 18 F32 source package without compiling it", flush=True)
        source_model = ct.models.MLModel(
            str(source), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True
        )
        int4 = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32
        )
        int8 = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_block", block_size=32
        )
        op_name_configs = {
            op_name: int8 if kind == "int8" else int4
            for op_name, kind in target_kinds.items()
        }
        # The converted graph reuses a scalar zero bias across all linears. It
        # must not inherit different configs from its int8-attention and
        # int4-nonattention consumers.
        op_name_configs.update({name: None for name in shared_constants})
        config = cto.OptimizationConfig(op_name_configs=op_name_configs)
        compressed = cto.linear_quantize_weights(source_model, config)
        package = output / "EmbeddingGemmaInt4AttentionInt8.mlpackage"
        compressed.save(str(package))
        shutil.copy2(source_fixtures, output / "padded-fixtures.npz")
        report["package_hashes"] = package_hashes(package)
        report["package_size_bytes"] = sum(path.stat().st_size for path in package.rglob("*") if path.is_file())
        report["specification_version"] = compressed.get_spec().specificationVersion
        report["artifact_assessment"] = "Not assessed; run scripts/coreml/assess_f32.py with --comparison-only"
        report["status"] = "exported"
        print(f"Compressed package saved: {package}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
