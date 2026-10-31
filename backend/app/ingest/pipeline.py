"""入库流程编排：扫描 → 去重 → 解析 → 分块 → 落盘。

刻意线性、无并发：M0 要先看清"每个文件花了多少时间、失败在哪一步"。
并发会把失败原因搅在一起，等流程稳定（M1）再谈加速。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.ingest.chunkers import chunk_document, config_from_settings
from app.ingest.images import save_image_asset, save_image_file
from app.ingest.ocr import render_pdf_page
from app.ingest.parse import IMAGE_EXTS, PARSER_VERSION, parse_document
from app.ingest.scan import rel_path_of, scan_corpus
from app.ingest.store import load_images, load_index, save_index
from app.schemas import Chunk, Doc, ImageRecord, make_doc_id

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    seen_files: int = 0
    candidates: int = 0
    skipped_by_ext: int = 0
    skipped_dirs: int = 0
    skipped_temp: int = 0
    duplicates: int = 0
    parsed_ok: int = 0
    parsed_failed: int = 0
    #: 因 --limit 根本没尝试解析的文档数。**不能算进"解析失败"**，
    #: 否则试跑会显示一堆假失败，掩盖真实失败率。
    skipped_limit: int = 0
    docs_indexed: int = 0
    chunks: int = 0
    chars: int = 0
    #: 增量：未变文档/切片直接复用，不重新解析/编码。
    reused_docs: int = 0
    reused_chunks: int = 0
    parsed_new: int = 0
    removed_docs: int = 0
    images: int = 0
    #: M1 解析质量：表格转 Markdown 的个数 / 丢掉的页眉页脚行数 / 判为双栏的文档数
    n_tables: int = 0
    n_dropped_furniture: int = 0
    n_two_column_docs: int = 0
    #: 实际走 OCR 的页数；0 表示没有扫描页/图片页。
    n_ocr_pages: int = 0
    duration_s: float = 0.0
    by_ext: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = dict(self.__dict__)
        data["failures"] = self.failures[:200]  # 报告里只留前 200 条
        data["n_failures"] = len(self.failures)
        return data


def _file_sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(block)
            if not buf:
                break
            digest.update(buf)
    return digest.hexdigest()


def ingest(
    settings: Settings,
    *,
    limit: int | None = None,
    verbose: bool = True,
    previous_storage_dir: Path | None = None,
) -> IngestStats:
    """重建一代索引；提供 previous_storage_dir 时按内容 hash 复用未变文档。

    ``limit`` 仍是试跑语义（不提升 current）。
    """
    started = time.perf_counter()
    stats = IngestStats()

    old_docs: list[Doc] = []
    old_chunks: list[Chunk] = []
    old_images: list[ImageRecord] = []
    if previous_storage_dir is not None and previous_storage_dir.is_dir():
        old_settings = dataclasses.replace(settings, storage_dir=previous_storage_dir)
        try:
            old_docs, old_chunks = load_index(old_settings)
            old_images = load_images(old_settings)
        except Exception as exc:  # noqa: BLE001 - 旧代际坏了就退化为全量
            log.warning("旧代际读取失败，本次全量：%s", exc)
    old_doc_by_id = {d.doc_id: d for d in old_docs}
    old_chunks_by_doc: dict[str, list[Chunk]] = {}
    for chunk in old_chunks:
        old_chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)
    old_images_by_doc: dict[str, list[ImageRecord]] = {}
    for image in old_images:
        old_images_by_doc.setdefault(image.doc_id, []).append(image)

    scan = scan_corpus(
        settings.corpus_root,
        settings.ext_whitelist,
        max_files=settings.max_files,
    )
    stats.seen_files = scan.seen_files
    stats.candidates = len(scan.files)
    stats.skipped_by_ext = scan.skipped_by_ext
    stats.skipped_dirs = scan.skipped_dirs
    stats.skipped_temp = scan.skipped_temp
    stats.by_ext = scan.by_ext()

    if verbose:
        log.info(
            "扫描完成：遍历 %d 个文件，命中 %d 个（跳过非白名单 %d 个），按扩展名 %s",
            scan.seen_files, len(scan.files), scan.skipped_by_ext, stats.by_ext,
        )

    # ---- 1. 建立文档记录（含内容 hash，用于去重）----
    # 配置指纹：同样的文件 + 同样的参数才会切出同样的 chunk 集。
    # 增量重建靠它判断"这份文档的 chunk 是不是旧的"（比状态枚举更直接）。
    chunk_config = config_from_settings(settings)
    config_fp = (
        f"{chunk_config.fingerprint},"
        f"parser={settings.parser_backend}/{settings.parse_strategy}/{PARSER_VERSION}"
    )
    docs: list[Doc] = []
    for path in scan.files:
        rel = rel_path_of(path, settings.corpus_root)
        try:
            stat = path.stat()
            sha = _file_sha256(path)
        except OSError as exc:
            stats.parsed_failed += 1
            stats.failures.append({"path": rel, "stage": "stat", "error": str(exc)})
            continue
        docs.append(
            Doc(
                doc_id=make_doc_id(rel),
                rel_path=rel,
                title=path.stem,
                ext=path.suffix.lower(),
                sha256=sha,
                size=stat.st_size,
                mtime=stat.st_mtime,
                chunk_config=config_fp,
            )
        )

    # ---- 2. 内容完全相同的文件只索引一份 ----
    # 语料里有「钻"拾"」4 个副本这类情况，重复索引会让同一段文本占掉多个 top-k 名额。
    first_by_hash: dict[str, str] = {}
    for doc in docs:
        keeper = first_by_hash.get(doc.sha256)
        if keeper is None:
            first_by_hash[doc.sha256] = doc.doc_id
        else:
            doc.status = "duplicate"
            stats.duplicates += 1
    if verbose and stats.duplicates:
        log.info("内容重复、跳过索引的文档 %d 个", stats.duplicates)

    active = [d for d in docs if d.status == "active"]
    chunks: list[Chunk] = []
    images: list[ImageRecord] = []

    # ---- 2.5 内容 hash 增量：未变文档直接复用旧 Doc/Chunk/Image ----
    reused_ids: set[str] = set()
    for doc in active:
        old = old_doc_by_id.get(doc.doc_id)
        if (
            old is not None
            and old.sha256 == doc.sha256
            and old.chunk_config == config_fp
            and old.parse_ok
            and old_chunks_by_doc.get(doc.doc_id)
            # 旧代际缺少“本应存在”的图片资产时，才重解析补图；
            # 纯文字 PDF 没有 image_id，不需要为了视觉索引重跑。
            and not (
                not old_images_by_doc.get(doc.doc_id)
                and any(c.image_id for c in old_chunks_by_doc.get(doc.doc_id, []))
            )
        ):
            doc.parse_ok = True
            doc.parse_error = None
            doc.n_chunks = old.n_chunks
            chunks.extend(old_chunks_by_doc[doc.doc_id])
            images.extend(old_images_by_doc.get(doc.doc_id, []))
            stats.reused_docs += 1
            stats.reused_chunks += old.n_chunks
            reused_ids.add(doc.doc_id)
    if verbose and stats.reused_docs:
        log.info("增量复用未变文档 %d 份 / %d chunks", stats.reused_docs, stats.reused_chunks)
    stats.removed_docs = len({d.doc_id for d in old_docs} - {d.doc_id for d in docs})
    active = [d for d in active if d.doc_id not in reused_ids]

    if limit is not None:
        # 没尝试解析的文档必须标成"未解析"，不能沿用 Doc 的 parse_ok=True 默认值——
        # 那会让 stats 显示"解析失败 0"，等于把试跑伪装成全量成功（P32）。
        for doc in active[limit:]:
            doc.parse_ok = False
            doc.parse_error = f"未解析：--limit {limit} 试跑"
            stats.skipped_limit += 1
        active = active[:limit]

    # ---- 3. 解析 + 分块 ----
    total = len(active)
    for i, doc in enumerate(active, start=1):
        path = settings.corpus_root / doc.rel_path
        result = parse_document(
            path,
            backend=settings.parser_backend,
            strategy=settings.parse_strategy,
            max_bytes=settings.max_file_bytes,
        )
        stats.n_ocr_pages += result.n_ocr_pages
        if not result.ok or not result.blocks:
            doc.parse_ok = False
            doc.parse_error = result.error or ("; ".join(result.warnings) or "无文本")
            doc.n_chunks = 0
            stats.parsed_failed += 1
            stats.failures.append(
                {"path": doc.rel_path, "stage": "parse", "error": doc.parse_error or ""}
            )
        else:
            stats.n_tables += result.n_tables
            stats.n_dropped_furniture += result.n_dropped_furniture
            if result.two_column:
                stats.n_two_column_docs += 1
            doc_chunks = chunk_document(doc.doc_id, result.blocks, chunk_config)
            if not doc_chunks:
                # 解析出了 block 但切出来是空的（整页空白 / 只有图片）：这不是成功。
                # 之前这里写成 parse_ok=True, n_chunks=0，于是 stats 里有 7 份文档
                # 既不算失败也不在索引里，对不上账（P32）。
                doc.parse_ok = False
                doc.parse_error = "解析后无文本（可能是图片型文档）"
                doc.n_chunks = 0
                stats.parsed_failed += 1
                stats.failures.append(
                    {"path": doc.rel_path, "stage": "chunk", "error": doc.parse_error}
                )
            else:
                doc.parse_ok = True
                doc.n_chunks = len(doc_chunks)
                doc.parse_error = None
                chunks.extend(doc_chunks)
                stats.parsed_ok += 1
                stats.parsed_new += 1
                stats.chars += result.chars

                # 图片/扫描页：保存页面图，文本 chunk 指向 image_id；文字层页面不建图。
                try:
                    if doc.ext in IMAGE_EXTS:
                        image = save_image_file(
                            settings, doc_id=doc.doc_id, rel_path=doc.rel_path, source=path
                        )
                        image.text_chunk_ids = [c.chunk_id for c in doc_chunks]
                        for chunk in doc_chunks:
                            chunk.image_id = image.image_id
                        images.append(image)
                        stats.images += 1
                    elif doc.ext == ".pdf" and result.ocr_pages:
                        for page_no in result.ocr_pages:
                            rendered = render_pdf_page(path, page_no - 1)
                            image = save_image_asset(
                                settings, doc_id=doc.doc_id, rel_path=doc.rel_path,
                                page=page_no, image=rendered,
                            )
                            page_chunks = [c for c in doc_chunks if c.page == page_no]
                            image.text_chunk_ids = [c.chunk_id for c in page_chunks]
                            for chunk in page_chunks:
                                chunk.image_id = image.image_id
                            images.append(image)
                            stats.images += 1
                except Exception as exc:  # noqa: BLE001 - 图片落盘失败不应毁掉文本索引
                    stats.warnings.append(f"{doc.rel_path}: 图片资产保存失败 {exc}")

                if result.warnings:
                    note = f"{doc.rel_path}: {'; '.join(result.warnings)}"
                    stats.warnings.append(note)
        if verbose and (i % 10 == 0 or i == total):
            log.info("解析进度 %d/%d，累计 chunk %d", i, total, len(chunks))

    indexed = [d for d in docs if d.parse_ok and d.n_chunks > 0]
    stats.docs_indexed = len(indexed)
    stats.chunks = len(chunks)
    stats.images = len(images)

    # ---- 4. 落盘 ----
    save_index(settings, docs, chunks, images)
    stats.duration_s = round(time.perf_counter() - started, 2)

    report_path = settings.storage_dir / "ingest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if verbose:
        log.info(
            "入库完成：索引 %d 份文档 / %d 个 chunk，失败 %d，耗时 %.1fs",
            stats.docs_indexed, stats.chunks, stats.parsed_failed, stats.duration_s,
        )
    return stats
