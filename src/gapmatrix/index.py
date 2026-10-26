"""Chunking, local embedding, and the Chroma vector store."""

import json

import chromadb
import numpy as np
from fastembed import TextEmbedding

from .config import repo_path

_model = None


def embed(texts: list[str], cfg: dict) -> np.ndarray:
    """Local CPU embedding via ONNX Runtime (fastembed); no torch, no cloud API.

    Vectors are L2-normalised so cosine distance in Chroma behaves as expected.
    """
    global _model
    if _model is None:
        _model = TextEmbedding(cfg["embedding"]["model"])
    vecs = np.array(list(_model.embed(texts, batch_size=cfg["embedding"]["batch_size"])))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character chunks with overlap (as in the class notebook)."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def collection(cfg: dict, reset: bool = False):
    client = chromadb.PersistentClient(path=str(repo_path(cfg["vector_store"]["dir"])))
    name = cfg["vector_store"]["collection"]
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def build_index(papers_sources: dict[str, list[dict]], cfg: dict, rebuild: bool = False) -> int:
    """Chunk each paper separately, embed locally, and store in Chroma.

    Incremental by default: papers already in the collection are skipped, because CPU
    embedding is slow. Use rebuild=True after changing chunking or the embedding model.
    """
    coll = collection(cfg, reset=rebuild)
    indexed = {m["paper_id"] for m in coll.get(include=["metadatas"])["metadatas"]}
    size, overlap = cfg["chunking"]["size"], cfg["chunking"]["overlap"]
    ids, docs, metas = [], [], []
    for paper_id, sources in papers_sources.items():
        if paper_id in indexed:
            continue
        for src in sources:
            for i, chunk in enumerate(chunk_text(src["text"], size, overlap)):
                ids.append(f"{paper_id}:{src['source']}:{i}")
                docs.append(chunk)
                metas.append({"paper_id": paper_id, "source": src["source"]})

    if docs:
        vectors = embed(docs, cfg)
        coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors.tolist())

    stats = {}
    for m in coll.get(include=["metadatas"])["metadatas"]:
        stats[m["paper_id"]] = stats.get(m["paper_id"], 0) + 1
    out = repo_path(cfg["data"]["cache_dir"]) / "index_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return len(ids)


def retrieve(paper_id: str, query: str, cfg: dict) -> list[dict]:
    """Top-k chunks for one paper only, so cells never mix evidence across papers."""
    coll = collection(cfg)
    qvec = embed([query], cfg).tolist()
    res = coll.query(query_embeddings=qvec, n_results=cfg["retrieval"]["top_k"],
                     where={"paper_id": paper_id})
    return [
        {"id": i, "text": d, "distance": round(dist, 4)}
        for i, d, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])
    ]
