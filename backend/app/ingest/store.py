"""索引落盘：JSONL（供人看/供 git diff）+ SQLite（供程序查）。

两份都写是有意的：
  - JSONL 是纯文本，出问题时能直接 grep、能进版本库做小规模样本；
  - SQLite 提供按 doc_id 的快速过滤（版本冲突题要按 status 过滤）。
M0 每次入库全量重建，不做增量合并——增量留到 M1，因为要先有版本号才能判断"哪份更新"。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from app.config import Settings
from app.schemas import Chunk, Doc, ImageRecord, load_chunks, load_docs, write_jsonl

log = logging.getLogger(__name__)


def _parser_version() -> str:
    """解析器版本；单独成函数是为了避免模块级循环导入。"""
    from app.ingest.parse import PARSER_VERSION

    return PARSER_VERSION

_SCHEMA = """PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS docs (
    doc_id      TEXT PRIMARY KEY,
    rel_path    TEXT NOT NULL,
    title       TEXT NOT NULL,
    ext         TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    parse_ok    INTEGER NOT NULL DEFAULT 1,
    parse_error TEXT,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    chunk_config TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    page       INTEGER,
    page_end   INTEGER,
    section    TEXT,
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end   INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    is_parent  INTEGER NOT NULL DEFAULT 0,
    mom_id     TEXT,
    image_id   TEXT,
    FOREIGN KEY (doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

#: 索引单独执行：老库可能缺新列，必须先 ALTER 再加索引，
#: 否则 `CREATE INDEX ... ON chunks(content_hash)` 会在缺列的老库上直接报错。
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_docs_rel  ON docs(rel_path);
CREATE INDEX IF NOT EXISTS idx_docs_status ON docs(status);
"""


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path) -> None:
    with connect(path) as conn:
        conn.executescript(_SCHEMA)
        # 老库缺列：补上，避免用户必须删库重来。新列默认空，
        # 载入时会按正文补算（见 load_index），下一次 ingest 落盘。
        chunk_cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "page_end" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN page_end INTEGER")
        if "content_hash" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
        if "is_parent" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN is_parent INTEGER NOT NULL DEFAULT 0")
        if "mom_id" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN mom_id TEXT")
        if "image_id" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN image_id TEXT")
        doc_cols = {row["name"] for row in conn.execute("PRAGMA table_info(docs)")}
        if "chunk_config" not in doc_cols:
            conn.execute("ALTER TABLE docs ADD COLUMN chunk_config TEXT NOT NULL DEFAULT ''")
        conn.executescript(_INDEXES)


def save_index(
    settings: Settings,
    docs: Sequence[Doc],
    chunks: Sequence[Chunk],
    images: Sequence[ImageRecord] = (),
) -> dict[str, int]:
    """全量重建索引，同时写 JSONL 与 SQLite。"""
    n_docs = write_jsonl(settings.docs_path, (d.to_dict() for d in docs))
    n_chunks = write_jsonl(settings.chunks_path, (c.to_dict() for c in chunks))
    n_images = write_jsonl(settings.images_path, (i.to_dict() for i in images))

    init_db(settings.sqlite_path)
    with connect(settings.sqlite_path) as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM docs")
        conn.executemany(
            """INSERT INTO docs
               (doc_id, rel_path, title, ext, sha256, size, mtime, status,
                parse_ok, parse_error, n_chunks, chunk_config)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    d.doc_id, d.rel_path, d.title, d.ext, d.sha256, d.size, d.mtime,
                    d.status, int(d.parse_ok), d.parse_error, d.n_chunks, d.chunk_config,
                )
                for d in docs
            ],
        )
        conn.executemany(
            """INSERT INTO chunks
               (chunk_id, doc_id, ordinal, text, page, page_end, section,
                char_start, char_end, content_hash, is_parent, mom_id, image_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (c.chunk_id, c.doc_id, c.ordinal, c.text, c.page, c.page_end, c.section,
                 c.char_start, c.char_end, c.content_hash, int(c.is_parent), c.mom_id,
                 c.image_id)
                for c in chunks
            ],
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("built_at", json.dumps(time.strftime("%Y-%m-%dT%H:%M:%S%z"))),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (
                "config",
                json.dumps(
                    {
                        "corpus_root": str(settings.corpus_root),
                        "chunk_size": settings.chunk_size,
                        "chunk_overlap": settings.chunk_overlap,
                        "chunk_strategy": settings.chunk_strategy,
                        "parser": settings.parser_backend,
                        "parser_version": _parser_version(),
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("counts", json.dumps({"docs": n_docs, "chunks": n_chunks, "images": n_images})),
        )

    return {"docs": n_docs, "chunks": n_chunks, "images": n_images}


def load_index(settings: Settings) -> tuple[list[Doc], list[Chunk]]:
    """优先读 SQLite；没有就退回 JSONL。"""
    if settings.sqlite_path.is_file():
        with connect(settings.sqlite_path) as conn:
            doc_rows = conn.execute("SELECT * FROM docs ORDER BY rel_path").fetchall()
            chunk_rows = conn.execute(
                "SELECT * FROM chunks ORDER BY doc_id, ordinal"
            ).fetchall()
        docs = [
            Doc(
                doc_id=r["doc_id"], rel_path=r["rel_path"], title=r["title"], ext=r["ext"],
                sha256=r["sha256"], size=r["size"], mtime=r["mtime"], status=r["status"],
                parse_ok=bool(r["parse_ok"]), parse_error=r["parse_error"],
                n_chunks=r["n_chunks"],
                chunk_config=(r["chunk_config"] if "chunk_config" in r.keys() else ""),
            )
            for r in doc_rows
        ]
        chunks = [
            Chunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], ordinal=r["ordinal"],
                text=r["text"], page=r["page"], page_end=r["page_end"],
                section=r["section"], char_start=r["char_start"], char_end=r["char_end"],
                content_hash=(r["content_hash"] if "content_hash" in r.keys() else ""),
                is_parent=bool(r["is_parent"]) if "is_parent" in r.keys() else False,
                mom_id=(r["mom_id"] if "mom_id" in r.keys() else None),
                image_id=(r["image_id"] if "image_id" in r.keys() else None),
            )
            for r in chunk_rows
        ]
        return docs, _fill_content_hash(chunks)

    return load_docs(settings.docs_path), _fill_content_hash(load_chunks(settings.chunks_path))


def load_images(settings: Settings) -> list[ImageRecord]:
    path = settings.images_path
    if not path.is_file():
        return []
    rows: list[ImageRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
            rows.append(ImageRecord(**data))
        except (json.JSONDecodeError, TypeError):
            continue
    return rows


def _fill_content_hash(chunks: Sequence[Chunk]) -> list[Chunk]:
    """老库/老 JSONL 没有内容指纹时按正文补算。

    不补的话，检索层的"按内容折叠"会把所有空指纹当成同一段话，
    把结果折成一条——这是必须防住的静默错。
    """
    from app.schemas import make_content_hash

    filled = 0
    for chunk in chunks:
        if not chunk.content_hash:
            chunk.content_hash = make_content_hash(chunk.text)
            filled += 1
    if filled:
        log.debug("补算 content_hash：%d 个片段（老索引）", filled)
    return list(chunks)


def read_meta(settings: Settings) -> dict[str, str]:
    if not settings.sqlite_path.is_file():
        return {}
    with connect(settings.sqlite_path) as conn:
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.OperationalError:
            return {}
    return {r["key"]: r["value"] for r in rows}
