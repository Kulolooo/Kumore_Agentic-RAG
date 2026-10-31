"""双塔向量构建：文本 bge-m3 与视觉 bge-visualized 使用两个独立索引。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.embed.local import LocalEmbedder
from app.embed.store import VectorIndex, reuse_by_content
from app.ingest.store import load_images, load_index
from app.schemas import Chunk, ImageRecord


def _text_vectors(
    settings: Settings,
    chunks: list[Chunk],
    *,
    batch_size: int,
) -> dict[str, Any]:
    chunks = [c for c in chunks if not c.is_parent]
    if not chunks:
        return {"chunks": 0, "encoded": 0, "seconds": 0.0, "reused": 0}
    started = time.perf_counter()
    old: VectorIndex | None = None
    if settings.vector_index_path.is_file():
        try:
            candidate = VectorIndex.load(settings.vector_index_path)
            if candidate.model_name == settings.embedding_model:
                old = candidate
        except Exception:
            old = None
    reused = reuse_by_content(chunks, old, settings.embedding_model) if old else {}
    missing = {
        c.content_hash: c.text
        for c in chunks
        if c.content_hash not in reused
    }
    fresh: dict[str, np.ndarray] = {}
    if missing:
        embedder = LocalEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=max(1, batch_size),
            model_dir=settings.model_dir,
        )
        vectors = embedder.encode(
            list(missing.values()),
            batch_size=max(1, batch_size),
            show_progress=True,
        )
        fresh = dict(zip(missing.keys(), vectors))
    by_hash = {**reused, **fresh}
    matrix = np.stack([by_hash[c.content_hash] for c in chunks]).astype(np.float32)
    index = VectorIndex.build(
        [c.chunk_id for c in chunks],
        matrix,
        settings.embedding_model,
        [c.content_hash for c in chunks],
    )
    index.save(settings.vector_index_path)
    return {
        "chunks": len(chunks),
        "encoded": len(fresh),
        "reused": len(reused),
        "seconds": round(time.perf_counter() - started, 1),
        "path": str(settings.vector_index_path),
        "dim": index.dim,
    }


def _visual_vectors(
    settings: Settings,
    images: list[ImageRecord],
    *,
    batch_size: int = 1,
) -> dict[str, Any]:
    if not images:
        return {"images": 0, "encoded": 0, "seconds": 0.0, "reused": 0}
    started = time.perf_counter()
    old: VectorIndex | None = None
    if settings.visual_vector_index_path.is_file():
        try:
            candidate = VectorIndex.load(settings.visual_vector_index_path)
            if candidate.model_name == settings.visual_embedding_model:
                old = candidate
        except Exception:
            old = None
    old_by_sha = {}
    if old is not None and old.content_hashes:
        old_by_sha = {
            digest: old.vectors[i]
            for i, digest in enumerate(old.content_hashes)
        }
    missing = [image for image in images if image.sha256 not in old_by_sha]
    fresh: dict[str, np.ndarray] = {}
    if missing:
        embedder = LocalEmbedder(
            model_name=settings.visual_embedding_model,
            device=settings.embedding_device,
            batch_size=1,
            model_dir=settings.model_dir,
            visual_weight=settings.embedding_visual_weight,
        )
        for image in missing:
            path = settings.assets_dir / image.image_file
            vectors = embedder.encode_images([path])
            fresh[image.sha256] = vectors[0]
    by_sha = {**old_by_sha, **fresh}
    matrix = np.stack([by_sha[image.sha256] for image in images]).astype(np.float32)
    index = VectorIndex.build(
        [image.image_id for image in images],
        matrix,
        settings.visual_embedding_model,
        [image.sha256 for image in images],
    )
    index.save(settings.visual_vector_index_path)
    return {
        "images": len(images),
        "encoded": len(fresh),
        "reused": len(old_by_sha),
        "seconds": round(time.perf_counter() - started, 1),
        "path": str(settings.visual_vector_index_path),
        "dim": index.dim,
    }


def build_vectors(
    settings: Settings,
    *,
    limit: int | None = None,
    text_batch: int = 32,
    tower: str = "text",
) -> dict[str, Any]:
    """一次只加载一座塔；两塔必须在不同进程，避免 4GB 权重同时驻留。"""
    docs, chunks = load_index(settings)
    if limit is not None:
        chunks = chunks[:limit]
    if tower == "text":
        return {"text": _text_vectors(settings, chunks, batch_size=text_batch)}
    if tower == "visual":
        return {"visual": _visual_vectors(settings, load_images(settings))}
    raise ValueError(f"tower 只能是 text/visual，当前 {tower!r}")
