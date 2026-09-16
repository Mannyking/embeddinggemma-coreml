"""Compare saved 512-token Core ML packages on BEIR SciFact retrieval."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "embeddinggemma-300m"
MODEL_RECEIPT = ROOT / "models" / "receipts" / "embeddinggemma-300m.json"
LENGTH = 512
DATASET = "scifact"
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
K_VALUES = (1, 5, 10)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files_sha256(directory):
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }


def dataset_hashes(path):
    required = (path / "corpus.jsonl", path / "queries.jsonl", path / "qrels" / "test.tsv")
    hashes = {item.relative_to(path).as_posix(): sha256(item) for item in required}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return hashes, identity


def verify_model():
    source_path = MODEL_RECEIPT
    source = json.loads(source_path.read_text())
    for item in source["files"]:
        if sha256(MODEL / item["path"]) != item["sha256"]:
            raise ValueError(f"Model hash mismatch: {item['path']}")
    return {"revision": source["revision"], "model_source_sha256": sha256(source_path)}


def verify_package(package):
    if not package.is_dir():
        raise FileNotFoundError(package)
    export_path = package.parent / "export-report.json"
    export = json.loads(export_path.read_text())
    hashes = files_sha256(package)
    if export["status"] != "exported" or export["input_shape"] != [1, LENGTH]:
        raise ValueError(f"Expected a completed fixed-{LENGTH} export: {package}")
    if export["package_hashes"] != hashes:
        raise ValueError(f"Package differs from its recorded export: {package}")
    return {
        "package": str(package), "package_hashes": hashes,
        "package_size_bytes": sum(path.stat().st_size for path in package.rglob("*") if path.is_file()),
        "export_precision": export["precision"], "export_report_sha256": sha256(export_path),
    }


def load_dataset(root):
    path = root / DATASET
    expected = (path / "corpus.jsonl", path / "queries.jsonl", path / "qrels" / "test.tsv")
    if not all(item.is_file() for item in expected):
        print(f"Downloading BEIR {DATASET} to {root}", flush=True)
        path = Path(util.download_and_unzip(DATASET_URL, str(root)))
    expected = (path / "corpus.jsonl", path / "queries.jsonl", path / "qrels" / "test.tsv")
    if not all(item.is_file() for item in expected):
        raise ValueError(f"Expected BEIR SciFact layout under {path}")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(path)).load(split="test")
    return path, corpus, queries, qrels


def prepare(tokenizer, prompt, text):
    if not text.strip():
        raise ValueError("BEIR record is empty")
    token_ids = tokenizer((prompt + text).strip(), truncation=False)["input_ids"]
    if len(token_ids) > LENGTH:
        return None, len(token_ids)
    ids = np.full(LENGTH, tokenizer.pad_token_id, dtype=np.int32)
    mask = np.zeros(LENGTH, dtype=np.int32)
    ids[:len(token_ids)] = token_ids
    mask[:len(token_ids)] = 1
    return (ids, mask), len(token_ids)


def prepare_records(tokenizer, prompt, texts):
    accepted, ids, masks, lengths, rejected = [], [], [], [], []
    for identifier, text in texts.items():
        prepared, length = prepare(tokenizer, prompt, text)
        if prepared is None:
            rejected.append({"id": identifier, "token_count": length, "reason": "above_fixed_limit"})
            continue
        input_ids, attention_mask = prepared
        accepted.append(identifier)
        ids.append(input_ids)
        masks.append(attention_mask)
        lengths.append(length)
    if not accepted:
        raise ValueError("No records fit the selected fixed input length")
    return {"ids": accepted, "input_ids": np.stack(ids), "attention_mask": np.stack(masks),
            "lengths": lengths, "rejected": rejected}


def predict(package, records, label):
    print(f"Loading {label}", flush=True)
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    shapes = {item.name: list(item.type.multiArrayType.shape)
              for item in model.get_spec().description.input}
    if shapes != {"input_ids": [1, LENGTH], "attention_mask": [1, LENGTH]}:
        raise ValueError(f"Unexpected Core ML input contract: {shapes}")
    embeddings = []
    for index, identifier in enumerate(records["ids"], start=1):
        embedding = np.asarray(model.predict({
            "input_ids": records["input_ids"][index - 1:index],
            "attention_mask": records["attention_mask"][index - 1:index],
        })["embedding"], dtype=np.float32)
        if embedding.shape != (1, 768) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid embedding for {label} {identifier}: {embedding.shape}")
        embeddings.append(embedding[0])
        if index % 100 == 0 or index == len(records["ids"]):
            print(f"{label}: {index}/{len(records['ids'])}", flush=True)
    embeddings = np.stack(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    return embeddings, {"minimum": float(norms.min()), "maximum": float(norms.max())}


def score(query_embeddings, document_embeddings, query_ids, document_ids, qrels):
    metrics = {**{f"recall@{k}": 0.0 for k in K_VALUES},
               **{f"ndcg@{k}": 0.0 for k in K_VALUES}, "mrr@10": 0.0}
    details = {}
    scores = query_embeddings @ document_embeddings.T
    for position, query_id in enumerate(query_ids):
        relevance = qrels[query_id]
        relevant = set(relevance)
        ranking = [document_ids[index] for index in np.argsort(-scores[position], kind="stable")]
        ranks = [rank + 1 for rank, identifier in enumerate(ranking) if identifier in relevant]
        top10 = ranking[:10]
        details[query_id] = {"relevant_document_ids": sorted(relevant), "top10": top10,
                             "first_relevant_rank": ranks[0] if ranks else None}
        for k in K_VALUES:
            top = ranking[:k]
            metrics[f"recall@{k}"] += len(set(top) & relevant) / len(relevant)
            gains = [relevance.get(identifier, 0) for identifier in top]
            dcg = sum((2 ** gain - 1) / np.log2(rank + 2) for rank, gain in enumerate(gains))
            ideal = sorted(relevance.values(), reverse=True)[:k]
            idcg = sum((2 ** gain - 1) / np.log2(rank + 2) for rank, gain in enumerate(ideal))
            metrics[f"ndcg@{k}"] += dcg / idcg if idcg else 0.0
        metrics["mrr@10"] += 1 / ranks[0] if ranks and ranks[0] <= 10 else 0.0
    return {name: value / len(query_ids) for name, value in metrics.items()}, details


def ranking_comparison(f32, candidate):
    overlap = [len(set(f32[qid]["top10"]) & set(candidate[qid]["top10"])) / 10 for qid in f32]
    f32_only = candidate_only = 0
    for query_id, f32_detail in f32.items():
        candidate_detail = candidate[query_id]
        f32_hit = f32_detail["top10"][0] in f32_detail["relevant_document_ids"]
        candidate_hit = candidate_detail["top10"][0] in candidate_detail["relevant_document_ids"]
        f32_only += f32_hit and not candidate_hit
        candidate_only += candidate_hit and not f32_hit
    return {"mean_top10_overlap": float(np.mean(overlap)),
            "identical_top10_rankings": sum(f32[qid]["top10"] == candidate[qid]["top10"] for qid in f32),
            "f32_top1_relevant_candidate_not": f32_only,
            "candidate_top1_relevant_f32_not": candidate_only}


def write_rankings(path, queries, f32, candidate):
    with path.open("w") as stream:
        for query_id, detail in f32.items():
            stream.write(json.dumps({"query_id": query_id, "query": queries[query_id],
                                     "relevant_document_ids": detail["relevant_document_ids"],
                                     "f32": detail, "int4_attention_int8": candidate[query_id]},
                                    ensure_ascii=False) + "\n")


def run(args, output, report):
    report["model_source"] = verify_model()
    report["models"] = {"f32": verify_package(args.f32_package),
                        "int4_attention_int8": verify_package(args.candidate_package)}
    report["stage"] = "load_dataset"
    path, corpus, queries, qrels = load_dataset(args.dataset_dir)
    hashes, identity = dataset_hashes(path)
    report["dataset"] = {"name": DATASET, "url": DATASET_URL, "path": str(path), "split": "test",
                         "corpus_records": len(corpus), "query_records": len(queries),
                         "files_sha256": hashes, "content_sha256": identity}
    config = json.loads((MODEL / "config_sentence_transformers.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    if tokenizer.padding_side != "right":
        raise ValueError("This evaluator requires right-padding")
    documents = prepare_records(tokenizer, config["prompts"]["document"], {
        identifier: f"{value.get('title', '').strip()}\n\n{value['text'].strip()}".strip()
        for identifier, value in corpus.items()
    })
    raw_queries = prepare_records(tokenizer, config["prompts"]["query"], queries)
    allowed_documents = set(documents["ids"])
    query_ids, filtered_qrels = [], {}
    for query_id in raw_queries["ids"]:
        relevance = {doc_id: score for doc_id, score in qrels.get(query_id, {}).items()
                     if doc_id in allowed_documents and score > 0}
        if relevance:
            query_ids.append(query_id)
            filtered_qrels[query_id] = relevance
    positions = [raw_queries["ids"].index(query_id) for query_id in query_ids]
    if not query_ids:
        raise ValueError("No test queries retain an eligible relevant document")
    query_inputs = {"ids": query_ids, "input_ids": raw_queries["input_ids"][positions],
                    "attention_mask": raw_queries["attention_mask"][positions]}
    report["input_filtering"] = {
        "policy": "Reject empty records and records above 512 prompted tokens; do not truncate or chunk.",
        "documents": {"accepted": len(documents["ids"]), "rejected": len(documents["rejected"]),
                      "token_length_minimum": min(documents["lengths"]), "token_length_maximum": max(documents["lengths"])},
        "queries": {"accepted_before_qrels": len(raw_queries["ids"]), "rejected": len(raw_queries["rejected"]),
                    "evaluated": len(query_ids), "excluded_without_eligible_relevance": len(raw_queries["ids"]) - len(query_ids),
                    "token_length_minimum": min(raw_queries["lengths"]), "token_length_maximum": max(raw_queries["lengths"])}
    }
    report["stage"] = "embed_f32"
    f32_documents, f32_doc_norms = predict(args.f32_package, documents, "F32 documents")
    f32_queries, f32_query_norms = predict(args.f32_package, query_inputs, "F32 queries")
    np.save(output / "f32_document_embeddings.npy", f32_documents)
    np.save(output / "f32_query_embeddings.npy", f32_queries)
    report["stage"] = "embed_int4_attention_int8"
    candidate_documents, candidate_doc_norms = predict(args.candidate_package, documents, "int4/int8 documents")
    candidate_queries, candidate_query_norms = predict(args.candidate_package, query_inputs, "int4/int8 queries")
    np.save(output / "int4_attention_int8_document_embeddings.npy", candidate_documents)
    np.save(output / "int4_attention_int8_query_embeddings.npy", candidate_queries)
    report["stage"] = "score_retrieval"
    f32_metrics, f32_details = score(f32_queries, f32_documents, query_ids, documents["ids"], filtered_qrels)
    candidate_metrics, candidate_details = score(candidate_queries, candidate_documents, query_ids, documents["ids"], filtered_qrels)
    write_rankings(output / "rankings.jsonl", queries, f32_details, candidate_details)
    report["embedding_norms"] = {"f32": {"documents": f32_doc_norms, "queries": f32_query_norms},
                                 "int4_attention_int8": {"documents": candidate_doc_norms, "queries": candidate_query_norms}}
    report["retrieval"] = {"f32": f32_metrics, "int4_attention_int8": candidate_metrics,
                           "delta_int4_minus_f32": {name: candidate_metrics[name] - f32_metrics[name] for name in f32_metrics},
                           "ranking_comparison": ranking_comparison(f32_details, candidate_details)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f32-package", type=Path,
                        default=ROOT / "artifacts/coreml/f32-512-ios18/EmbeddingGemmaF32.mlpackage")
    parser.add_argument("--candidate-package", type=Path,
                        default=ROOT / "artifacts/coreml/int4-512-attention-int8/EmbeddingGemmaInt4AttentionInt8.mlpackage")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/beir")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.f32_package, args.candidate_package = args.f32_package.resolve(), args.candidate_package.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new assessment output directory: {output}")
    report = {"status": "running", "stage": "verify_inputs", "python": platform.python_version(),
              "platform": platform.platform(), "compute_units": "CPU_ONLY", "sequence_length": LENGTH,
              "packages": {name: importlib.metadata.version(name) for name in ("beir", "coremltools", "numpy", "transformers")},
              "script_sha256": sha256(Path(__file__)),
              "comparison": "Saved F32 iOS 18 package versus saved int4/int8-attention package.",
              "limitations": "SciFact test only; no device benchmark or product-corpus evaluation."}
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
