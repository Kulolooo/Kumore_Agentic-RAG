"""视觉索引的懒编码队列/状态；视觉向量与文本向量完全分开。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from app.config import Settings
from app.embed.store import VectorIndex
from app.ingest.store import load_images


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def status(settings: Settings) -> dict[str, Any]:
    total = len(load_images(settings))
    visual_ids: set[str] = set()
    if settings.visual_vector_index_path.is_file():
        try:
            visual_ids = set(VectorIndex.load(settings.visual_vector_index_path).chunk_ids)
        except Exception:
            visual_ids = set()
    queue = _read(settings.visual_queue_path, [])
    ready = sum(1 for i in load_images(settings) if i.image_id in visual_ids)
    return {
        "total": total,
        "ready": ready,
        "queued": len(queue),
        "percent": round(100 * ready / total, 1) if total else 100.0,
        "state": "ready" if ready >= total else ("building" if queue else "partial"),
    }


def enqueue_image_ids(settings: Settings, image_ids: set[str]) -> list[str]:
    if not image_ids:
        return []
    visual_ids: set[str] = set()
    if settings.visual_vector_index_path.is_file():
        try:
            visual_ids = set(VectorIndex.load(settings.visual_vector_index_path).chunk_ids)
        except Exception:
            visual_ids = set()
    missing = sorted(image_ids - visual_ids)
    if not missing:
        return []
    queue = set(_read(settings.visual_queue_path, []))
    queue.update(missing)
    _write(settings.visual_queue_path, sorted(queue))
    return missing


def enqueue_from_chunks(settings: Settings, chunk_ids: list[str]) -> list[str]:
    # 只加载图片映射，不重建 Retriever。
    from app.ingest.store import load_index

    _, chunks = load_index(settings)
    wanted = {
        c.image_id
        for c in chunks
        if c.chunk_id in set(chunk_ids) and c.image_id
    }
    return enqueue_image_ids(settings, wanted)


def claim_queue(settings: Settings) -> list[str]:
    """队列认领；worker 用锁文件保证单写者。"""
    queue = list(_read(settings.visual_queue_path, []))
    if queue:
        _write(settings.visual_queue_path, [])
    return queue


def mark_status(settings: Settings, **updates: Any) -> None:
    data = _read(settings.visual_status_path, {})
    data.update(updates)
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(settings.visual_status_path, data)


def failed(settings: Settings, image_ids: list[str], error: str) -> None:
    mark_status(settings, failed=image_ids, error=error[:500])
