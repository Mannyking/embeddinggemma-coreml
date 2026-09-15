"""Assess local float32 reference embeddings; this does not validate Core ML parity."""

import os
from pathlib import Path

# Set before importing Hugging Face libraries to keep this check offline.
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    model_path = Path(__file__).resolve().parents[2] / "models/embeddinggemma-300m"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Download the pinned checkpoint to {model_path} first")

    model = SentenceTransformer(
        str(model_path),
        device="cpu",
        local_files_only=True,
        model_kwargs={
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
        },
    )
    model.eval()
    model[0].auto_model.config.use_cache = False

    query = model.encode_query("Which planet is known as the Red Planet?")
    docs = model.encode_document([
        "Mars is known as the Red Planet.",
        "Venus has a thick atmosphere.",
        "Saturn has prominent rings.",
    ])

    vectors = np.vstack([query, docs])
    if vectors.shape != (4, 768):
        raise ValueError(f"Expected shape (4, 768), got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"Embeddings are not unit normalized: {norms}")

    scores = docs @ query
    labels = ["Mars", "Venus", "Saturn"]
    print("\nQuery: Which planet is known as the Red Planet?")
    print(f"docs @ query: {docs.shape} @ {query.shape} -> {scores.shape}")
    print("Each document row is compared with the same query vector.")
    print("Unit norms [query, Mars, Venus, Saturn]:", norms)

    print("\nEmbedding preview (first 6 of 768 dimensions):")
    with np.printoptions(precision=5, suppress=True):
        print("Query: ", query[:6])
        for label, doc in zip(labels, docs):
            print(f"{label:6}: {doc[:6]}")

    print("\nInside the Mars dot product (first 6 dimensions):")
    print(" dimension     document        query      product")
    for i in range(6):
        print(f"{i:10d} {docs[0, i]:12.6f} {query[i]:12.6f} "
              f"{docs[0, i] * query[i]:12.6f}")
    products = docs * query
    print(f"Elementwise multiplication: docs * query -> {products.shape}")
    print("Sum all 768 products per row to get the dot product:")
    for label, row, score in zip(labels, products, scores):
        print(f"  {label:6}: sum(products) = {row.sum():.6f}; @ score = {score:.6f}")

    print("\nRetrieval ranking (cosine similarity; higher is better):")
    for rank, index in enumerate(np.argsort(-scores), start=1):
        print(f"  {rank}. {labels[index]:6} {scores[index]:+.6f}")
    if int(np.argmax(scores)) != 0:
        raise ValueError("Expected Mars to rank first")
    print("Reference smoke check passed")


if __name__ == "__main__":
    main()
