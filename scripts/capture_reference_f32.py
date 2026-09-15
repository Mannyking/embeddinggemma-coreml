"""Save offline CPU float32 fixtures for later Core ML artifact comparisons."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/embeddinggemma-300m"
OUTPUT = ROOT / "artifacts/reference-f32"
LIMIT = 2048


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(model, text, role):
    # Reject empty content before adding a prompt. Preserve nonempty content;
    # Sentence Transformers strips the resulting prompted string at tokenization.
    if not text.strip():
        raise ValueError("Empty or whitespace-only input is rejected")
    prompted = model.prompts[role] + text
    ids = model.tokenizer(prompted.strip(), truncation=False)["input_ids"]
    if len(ids) > LIMIT:
        raise ValueError(f"Input has {len(ids)} tokens; limit is {LIMIT}, including prompt and special tokens")
    return prompted, ids


def exact_length_text(model, target):
    # Build real text, then verify its length with the actual tokenizer.
    for count in range(max(1, target - 64), target + 1):
        text = "planet " * count
        ids = model.tokenizer((model.prompts["document"] + text).strip(), truncation=False)["input_ids"]
        if len(ids) == target:
            return text
    raise ValueError(f"Could not construct a {target}-token fixture")


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve or move the existing fixture directory before rerunning: {OUTPUT}")
    manifest_path = ROOT / "provenance/model-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        if sha256(MODEL / item["path"]) != item["sha256"]:
            raise ValueError(f"Model hash mismatch: {item['path']}")

    model = SentenceTransformer(
        str(MODEL), device="cpu", local_files_only=True,
        model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"},
    )
    model.eval()
    config = model[0].auto_model.config
    config.use_cache = False
    if not config.use_bidirectional_attention or not model[1].include_prompt:
        raise ValueError("Expected bidirectional attention and prompt-inclusive pooling")

    cases = [
        ("query_mars", "query", "Which planet is known as the Red Planet?"),
        ("doc_mars", "document", "Mars is known as the Red Planet."),
        ("doc_venus", "document", "Venus has a thick atmosphere."),
        ("doc_saturn", "document", "Saturn has prominent rings."),
        ("short", "document", "Hi"),
        ("query_french", "query", "Quelle planète est surnommée la planète rouge ?"),
        ("doc_arabic", "document", "يُعرف المريخ بالكوكب الأحمر."),
        ("doc_japanese", "document", "火星は赤い惑星として知られています。"),
    ]
    for length in (511, 512, 513, 2048):
        cases.append((f"tokens_{length}", "document", exact_length_text(model, length)))

    rejected = []
    for name, text in [("empty", ""), ("whitespace", " \t\n"),
                       ("over_limit", exact_length_text(model, LIMIT + 1))]:
        try:
            prepare(model, text, "document")
        except ValueError as error:
            rejected.append({"id": name, "reason": str(error)})
        else:
            raise ValueError(f"Policy failed to reject {name}")

    arrays = {}
    records = []
    vectors = {}
    with torch.inference_mode():
        for name, role, text in cases:
            prompted, expected_ids = prepare(model, text, role)
            # Batch size one avoids encode's length sorting. Save precisely the
            # inputs passed to the complete official module pipeline.
            features = model.tokenize([prompted])
            if features["input_ids"][0].tolist() != expected_ids:
                raise ValueError(f"Unexpected tokenization/truncation for {name}")
            for key, value in features.items():
                arrays[f"{name}__{key}"] = value.numpy().copy()
            embedding = model(features)["sentence_embedding"].numpy().copy()
            public = model.encode([text], prompt_name=role, show_progress_bar=False)
            np.testing.assert_allclose(embedding, public, atol=1e-6, rtol=1e-5)
            validate(embedding)
            arrays[f"{name}__embedding"] = embedding
            vectors[name] = embedding[0]
            records.append({"id": name, "role": role, "text": text,
                            "prompt": model.prompts[role], "prompted_text": prompted,
                            "token_count": len(expected_ids)})
            print(f"Captured {name}: {len(expected_ids)} tokens", flush=True)

        # A fixed 128-token padded batch provides inputs for the first export.
        short_cases = cases[:8]
        features = model.tokenizer(
            [prepare(model, text, role)[0].strip() for _, role, text in short_cases],
            padding="max_length", max_length=128, truncation=False, return_tensors="pt",
        )
        for key, value in features.items():
            arrays[f"padded_128__{key}"] = value.numpy().copy()
        padded = model(dict(features))["sentence_embedding"].numpy().copy()
        validate(padded)
        unpadded = np.stack([vectors[name] for name, _, _ in short_cases])
        np.testing.assert_allclose(padded, unpadded, atol=1e-5, rtol=1e-4)
        arrays["padded_128__embedding"] = padded

    documents = ["doc_mars", "doc_venus", "doc_saturn"]
    scores = np.stack([vectors[name] for name in documents]) @ vectors["query_mars"]
    ranking = [documents[i] for i in np.argsort(-scores)]
    if ranking[0] != "doc_mars":
        raise ValueError(f"Unexpected retrieval ranking: {ranking}")
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OUTPUT / "tensors.npz", **arrays)
    metadata = {
        "revision": manifest["revision"], "model_manifest_sha256": sha256(manifest_path),
        "script_sha256": sha256(Path(__file__)), "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "device": "cpu", "dtype": "float32", "attention": "eager", "use_cache": False,
        "policy": "Reject empty/whitespace-only content and inputs above 2048 tokens including prompts and special tokens; never truncate.",
        "cases": records, "rejected": rejected,
        "padded_128_order": [name for name, _, _ in short_cases],
        "padding_side": model.tokenizer.padding_side,
        "validation_tolerances": {"unit_norm_atol": 1e-5, "encode_atol": 1e-6,
                                  "encode_rtol": 1e-5, "padding_atol": 1e-5, "padding_rtol": 1e-4},
        "retrieval": {"documents": documents, "scores": scores.tolist(), "ranking": ranking},
        "tensors_sha256": sha256(OUTPUT / "tensors.npz"),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved validated reference fixtures to {OUTPUT}")


def validate(embedding):
    if embedding.ndim != 2 or embedding.shape[1] != 768 or not np.isfinite(embedding).all():
        raise ValueError("Expected finite 768-dimensional embeddings")
    np.testing.assert_allclose(np.linalg.norm(embedding, axis=1), 1.0, atol=1e-5, rtol=0)


if __name__ == "__main__":
    main()
