"""独立进程：懒编码视觉向量，写新索引后原子替换。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from app.config import Settings
from app.embed.local import LocalEmbedder
from app.embed.store import VectorIndex
from app.ingest.store import load_images
from app.kb import resolve_settings
from app.visual_index import claim_queue, failed, mark_status, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", required=True)
    args = parser.parse_args(argv)
    base = Settings.from_env()
    settings, kb = resolve_settings(args.kb, base)
    lock_path = settings.visual_queue_path.with_name("visual_worker.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 0
    try:
        queue = claim_queue(settings)
        images = {i.image_id: i for i in load_images(settings)}
        wanted = [images[i] for i in queue if i in images]
        if not wanted:
            mark_status(settings, state="ready")
            return 0
        old: VectorIndex | None = None
        if settings.visual_vector_index_path.is_file():
            try:
                old = VectorIndex.load(settings.visual_vector_index_path)
            except Exception:
                old = None
        old_by_sha = {}
        if old is not None and old.content_hashes:
            old_by_sha = {
                h: old.vectors[i] for i, h in enumerate(old.content_hashes)
            }
        embedder = LocalEmbedder(
            model_name=settings.visual_embedding_model,
            device=settings.embedding_device,
            batch_size=1,
            model_dir=settings.model_dir,
            visual_weight=settings.embedding_visual_weight,
        )
        mark_status(settings, state="building", total=len(images), queued=len(wanted))
        done = 0
        new_by_sha = dict(old_by_sha)
        for image in wanted:
            vector = embedder.encode_images([settings.assets_dir / image.image_file])[0]
            new_by_sha[image.sha256] = vector
            done += 1
            mark_status(
                settings,
                state="building",
                processed=done,
                total=len(images),
                ready=status(settings)["ready"] + done,
                percent=round(100 * (status(settings)["ready"] + done) / max(1, len(images)), 1),
            )
        # 以全部图片为行，缺向量的（尚未排到）沿用旧索引；没有的行先跳过，
        # 保证视觉索引永远只包含真实存在的图。
        ids: list[str] = []
        vectors = []
        hashes = []
        for image in images.values():
            vec = new_by_sha.get(image.sha256)
            if vec is None:
                continue
            ids.append(image.image_id)
            hashes.append(image.sha256)
            vectors.append(vec)
        index = VectorIndex.build(
            ids, __import__("numpy").stack(vectors).astype("float32"),
            settings.visual_embedding_model, hashes,
        )
        index.save(settings.visual_vector_index_path)
        mark_status(settings, state="ready" if len(ids) >= len(images) else "partial",
                    queued=0, processed=done, ready=len(ids),
                    percent=round(100 * len(ids) / max(1, len(images)), 1))
        return 0
    except Exception as exc:  # noqa: BLE001
        failed(settings, queue, f"{type(exc).__name__}: {exc}")
        print(f"visual worker failed: {exc}", file=sys.stderr)
        return 3
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
