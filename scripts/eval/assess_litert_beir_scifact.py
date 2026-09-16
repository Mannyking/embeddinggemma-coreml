"""Compare LiteRT with a saved Core ML SciFact retrieval assessment."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback

from ai_edge_litert.interpreter import Interpreter
import numpy as np

from assess_beir_scifact import (
    DATASET,
    DATASET_URL,
    LENGTH,
    MODEL,
    load_dataset,
    prepare_records,
    ranking_comparison,
    score,
    sha256,
    verify_model,
)
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LITERT = ROOT / "models" / "embeddinggemma-300M_seq512_mixed-precision.tflite"
DEFAULT_COREML = ROOT / "artifacts" / "evaluations" / "scifact" / "initial"


def load_coreml_assessment(path):
    report_path = path / "report.json"
    rankings_path = path / "rankings.jsonl"
    report = json.loads(report_path.read_text())
    if report["status"] != "assessed" or report["dataset"]["name"] != DATASET:
        raise ValueError(f"Expected a completed SciFact Core ML assessment: {path}")
    if not rankings_path.is_file():
        raise FileNotFoundError(rankings_path)
    f32, candidate = {}, {}
    with rankings_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            f32[row["query_id"]] = row["f32"]
            candidate[row["query_id"]] = row["int4_attention_int8"]
    expected = report["input_filtering"]
    for name, shape in (
        ("f32_document_embeddings.npy", (expected["documents"]["accepted"], 768)),
        ("f32_query_embeddings.npy", (expected["queries"]["evaluated"], 768)),
        ("int4_attention_int8_document_embeddings.npy", (expected["documents"]["accepted"], 768)),
        ("int4_attention_int8_query_embeddings.npy", (expected["queries"]["evaluated"], 768)),
    ):
        if np.load(path / name, mmap_mode="r").shape != shape:
            raise ValueError(f"Unexpected saved Core ML embedding shape: {name}")
    return report, f32, candidate


def predict(model_path, records, label):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or tuple(inputs[0]["shape"]) != (1, LENGTH) or inputs[0]["dtype"] != np.int32:
        raise ValueError(f"Unexpected LiteRT input contract: {inputs}")
    if len(outputs) != 1 or tuple(outputs[0]["shape"]) != (1, 768) or outputs[0]["dtype"] != np.float32:
        raise ValueError(f"Unexpected LiteRT output contract: {outputs}")
    embeddings = []
    for index, identifier in enumerate(records["ids"], start=1):
        interpreter.set_tensor(inputs[0]["index"], records["input_ids"][index - 1:index])
        interpreter.invoke()
        embedding = np.asarray(interpreter.get_tensor(outputs[0]["index"]), dtype=np.float32)
        if embedding.shape != (1, 768) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid LiteRT embedding for {identifier}: {embedding.shape}")
        embeddings.append(embedding[0])
        if index % 100 == 0 or index == len(records["ids"]):
            print(f"{label}: {index}/{len(records['ids'])}", flush=True)
    embeddings = np.stack(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    return embeddings, {"minimum": float(norms.min()), "maximum": float(norms.max())}, inputs, outputs


def write_rankings(path, queries, f32, candidate, litert):
    with path.open("w") as stream:
        for query_id, detail in f32.items():
            stream.write(json.dumps({
                "query_id": query_id,
                "query": queries[query_id],
                "relevant_document_ids": detail["relevant_document_ids"],
                "f32": detail,
                "int4_attention_int8": candidate[query_id],
                "litert": litert[query_id],
            }, ensure_ascii=False) + "\n")


def run(args, output, report):
    report["model_source"] = verify_model()
    coreml_report, f32_details, candidate_details = load_coreml_assessment(args.coreml_assessment)
    report["coreml_assessment"] = {
        "path": str(args.coreml_assessment),
        "report_sha256": sha256(args.coreml_assessment / "report.json"),
        "f32_package": coreml_report["models"]["f32"],
        "int4_attention_int8_package": coreml_report["models"]["int4_attention_int8"],
    }
    report["stage"] = "load_dataset"
    dataset_path, corpus, queries, qrels = load_dataset(args.dataset_dir)
    report["dataset"] = {
        "name": DATASET, "url": DATASET_URL, "path": str(dataset_path), "split": "test",
        "corpus_records": len(corpus), "query_records": len(queries),
        "checksum": "not recorded by request",
    }
    config = json.loads((MODEL / "config_sentence_transformers.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    if tokenizer.padding_side != "right":
        raise ValueError("This evaluator requires right-padding")
    documents = prepare_records(tokenizer, config["prompts"]["document"], {
        identifier: f"{value.get('title', '').strip()}\n\n{value['text'].strip()}".strip()
        for identifier, value in corpus.items()
    })
    raw_queries = prepare_records(tokenizer, config["prompts"]["query"], queries)
    eligible_documents = set(documents["ids"])
    query_ids, filtered_qrels = [], {}
    for query_id in raw_queries["ids"]:
        relevance = {doc_id: score_value for doc_id, score_value in qrels.get(query_id, {}).items()
                     if doc_id in eligible_documents and score_value > 0}
        if relevance:
            query_ids.append(query_id)
            filtered_qrels[query_id] = relevance
    positions = [raw_queries["ids"].index(query_id) for query_id in query_ids]
    query_inputs = {"ids": query_ids, "input_ids": raw_queries["input_ids"][positions],
                    "attention_mask": raw_queries["attention_mask"][positions]}
    expected = coreml_report["input_filtering"]
    if len(documents["ids"]) != expected["documents"]["accepted"] or len(query_ids) != expected["queries"]["evaluated"]:
        raise ValueError("Current dataset preparation differs from the referenced Core ML assessment")
    if set(query_ids) != set(f32_details) or set(query_ids) != set(candidate_details):
        raise ValueError("Current test queries differ from the referenced Core ML assessment")
    report["input_filtering"] = {
        "policy": "Reject empty records and records above 512 prompted tokens; do not truncate or chunk.",
        "documents": {"accepted": len(documents["ids"]), "rejected": len(documents["rejected"])},
        "queries": {"accepted_before_qrels": len(raw_queries["ids"]), "rejected": len(raw_queries["rejected"]),
                    "evaluated": len(query_ids), "excluded_without_eligible_relevance": len(raw_queries["ids"]) - len(query_ids)},
    }
    report["stage"] = "embed_litert"
    litert_documents, document_norms, inputs, outputs = predict(args.litert_model, documents, "LiteRT documents")
    litert_queries, query_norms, _, _ = predict(args.litert_model, query_inputs, "LiteRT queries")
    np.save(output / "litert_document_embeddings.npy", litert_documents)
    np.save(output / "litert_query_embeddings.npy", litert_queries)
    report["litert_model"] = {
        "path": str(args.litert_model), "sha256": sha256(args.litert_model),
        "size_bytes": args.litert_model.stat().st_size,
        "input_details": [{"name": item["name"], "shape": item["shape"].tolist(), "dtype": np.dtype(item["dtype"]).name} for item in inputs],
        "output_details": [{"name": item["name"], "shape": item["shape"].tolist(), "dtype": np.dtype(item["dtype"]).name} for item in outputs],
        "embedding_norms": {"documents": document_norms, "queries": query_norms},
    }
    report["stage"] = "score_retrieval"
    litert_metrics, litert_details = score(litert_queries, litert_documents, query_ids, documents["ids"], filtered_qrels)
    write_rankings(output / "rankings.jsonl", queries, f32_details, candidate_details, litert_details)
    f32_metrics = coreml_report["retrieval"]["f32"]
    candidate_metrics = coreml_report["retrieval"]["int4_attention_int8"]
    report["retrieval"] = {
        "f32": f32_metrics,
        "int4_attention_int8": candidate_metrics,
        "litert": litert_metrics,
        "delta_litert_minus_f32": {key: litert_metrics[key] - f32_metrics[key] for key in litert_metrics},
        "delta_litert_minus_int4_attention_int8": {key: litert_metrics[key] - candidate_metrics[key] for key in litert_metrics},
        "ranking_comparison": {
            "int4_attention_int8_vs_f32": ranking_comparison(f32_details, candidate_details),
            "litert_vs_f32": ranking_comparison(f32_details, litert_details),
            "litert_vs_int4_attention_int8": ranking_comparison(candidate_details, litert_details),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-assessment", type=Path, default=DEFAULT_COREML)
    parser.add_argument("--litert-model", type=Path, default=DEFAULT_LITERT)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets" / "beir")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.coreml_assessment = args.coreml_assessment.resolve()
    args.litert_model = args.litert_model.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new assessment output directory: {output}")
    if not args.litert_model.is_file():
        raise FileNotFoundError(args.litert_model)
    report = {
        "status": "running", "stage": "verify_inputs", "python": platform.python_version(),
        "platform": platform.platform(), "sequence_length": LENGTH,
        "packages": {name: importlib.metadata.version(name) for name in ("ai-edge-litert", "numpy", "transformers")},
        "script_sha256": sha256(Path(__file__)),
        "comparison": "LiteRT versus the saved F32 and int4/int8-attention SciFact retrieval assessment.",
        "limitations": "SciFact test only; no device benchmark or product-corpus evaluation.",
    }
    output.mkdir(parents=True)
    try:
        run(args, output, report)
        report["status"] = "assessed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
