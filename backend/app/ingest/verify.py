"""索引一致性校验：文档表 / chunk 集 / 向量索引三方对账。

为什么值得单独一层：向量索引缺 chunk 时，检索层是**静默跳过**的
（`retriever.search` 里 `_by_chunk_id.get(chunk_id)` 拿到 None 就 continue），
结果是混合检索的向量腿悄悄失效，指标下降却没有任何报错。
M1 每改一次分块参数都要重建索引，这个坑必然会踩到，所以把"对不上"变成硬失败。

依据（逆向）：
- 换 embedding 模型时旧向量**读不到**是 RAGFlow 的设计（向量维度进字段名，
  `docs/逆向/02-摄取链路.md:135`）；我们靠"文件名带模型 + 载入校验"达到同样效果，
  但文件名可能被覆盖，所以载入后还要比对；
- 写失败要删半成品（`docs/逆向/notes/A-ingestion.md:585` 的 `compensateFailedIndexWrite`）；
- 文档级 `chunk_num` 要能与索引对上（`docs/逆向/notes/D-data-api-deploy.md:65`）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.config import Settings
from app.embed.store import VectorIndex
from app.schemas import Chunk, Doc


class IndexInconsistent(RuntimeError):
    """索引对不上。CLI 捕获后 ``exit 3``，而不是继续跑出错误结论。"""


@dataclass(frozen=True)
class Problem:
    what: str
    detail: str
    hint: str

    def __str__(self) -> str:
        return f"{self.what}：{self.detail}　→ {self.hint}"


def check(
    settings: Settings,
    docs: list[Doc],
    chunks: list[Chunk],
    vector_index: VectorIndex | None = None,
) -> list[Problem]:
    """返回所有对不上的地方；空列表表示一致。"""
    problems: list[Problem] = []
    counts = Counter(c.doc_id for c in chunks)
    doc_ids = {d.doc_id for d in docs}

    # 1. 文档表声称的 chunk 数 vs 索引里实际条数
    for doc in docs:
        if doc.status != "active" or not doc.parse_ok:
            continue
        actual = counts.get(doc.doc_id, 0)
        if doc.n_chunks == 0:
            problems.append(
                Problem(
                    "文档状态",
                    f"{doc.rel_path} 标为已解析，却没有 chunk",
                    "这是半成品入库的典型症状；重新运行 python -m app ingest",
                )
            )
        elif actual != doc.n_chunks:
            problems.append(
                Problem(
                    "chunk 数",
                    f"{doc.rel_path}：文档表 {doc.n_chunks}，索引里 {actual}",
                    "重新运行 python -m app ingest（不要用 --limit）",
                )
            )

    # 2. 孤儿 chunk（指向不存在的文档）
    orphans = sorted({c.doc_id for c in chunks if c.doc_id not in doc_ids})
    if orphans:
        problems.append(
            Problem(
                "孤儿 chunk",
                f"{len(orphans)} 个 doc_id 在文档表里不存在，例如 {orphans[0]}",
                "重新运行 python -m app ingest",
            )
        )

    # 3. 内容指纹缺失（老库没迁移完就上检索，会把结果折成一条）
    missing_hash = sum(1 for c in chunks if not c.content_hash)
    if missing_hash:
        problems.append(
            Problem(
                "内容指纹",
                f"{missing_hash}/{len(chunks)} 个 chunk 没有 content_hash",
                "重新运行 python -m app ingest",
            )
        )

    # 4. 父子块关系：子块的 mom_id 必须指向同文档的父块；父块不能被当成检索单元。
    by_id = {c.chunk_id: c for c in chunks}
    bad_mom = [
        c.chunk_id
        for c in chunks
        if c.mom_id and (c.mom_id not in by_id or not by_id[c.mom_id].is_parent)
    ]
    if bad_mom:
        problems.append(
            Problem(
                "父子块",
                f"{len(bad_mom)} 个子块的 mom_id 指不到父块，例如 {bad_mom[0]}",
                "重新运行 python -m app ingest（父块与子块必须同批写入）",
            )
        )
    known_children = {c.mom_id for c in chunks if c.mom_id}
    lonely_parents = [
        c.chunk_id
        for c in chunks
        if c.is_parent and c.chunk_id not in known_children
    ]
    if lonely_parents:
        problems.append(
            Problem(
                "父子块",
                f"{len(lonely_parents)} 个父块没有子块，永远不会被命中",
                "重新运行 python -m app ingest；若持续出现，检查 child_tokens 是否过大",
            )
        )

    # 5. 向量索引覆盖情况（bm25 模式不需要）。父块不参与检索，也不该出现在向量索引里。
    if settings.retrieval_mode != "bm25":
        corpus_ids = {c.chunk_id for c in chunks if not c.is_parent}
        if vector_index is None:
            problems.append(
                Problem(
                    "向量索引",
                    "检索模式是 "
                    f"{settings.retrieval_mode}，但没有加载体向量索引",
                    "python -m app embed",
                )
            )
        else:
            vec_ids = set(vector_index.chunk_ids)
            missing = corpus_ids - vec_ids
            dead = vec_ids - corpus_ids
            if missing:
                problems.append(
                    Problem(
                        "向量覆盖",
                        f"{len(missing)}/{len(corpus_ids)} 个 chunk 没有向量，这些片段向量腿永远召不回",
                        "python -m app embed --rebuild",
                    )
                )
            if dead:
                problems.append(
                    Problem(
                        "向量失效",
                        f"{len(dead)} 条向量的 chunk 已不在索引里（重建过索引但没重建向量）",
                        "python -m app embed --rebuild",
                    )
                )
            if vector_index.model_name != settings.embedding_model:
                problems.append(
                    Problem(
                        "向量模型",
                        f"索引是 {vector_index.model_name}，配置是 {settings.embedding_model}",
                        "换模型必须重建：python -m app embed --rebuild",
                    )
                )

    return problems


def raise_if_any(
    settings: Settings,
    docs: list[Doc],
    chunks: list[Chunk],
    vector_index: VectorIndex | None = None,
) -> None:
    problems = check(settings, docs, chunks, vector_index)
    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise IndexInconsistent(f"索引不一致，先修好再检索：\n{detail}")


def corpus_drift(settings: Settings, docs: list[Doc]) -> dict[str, list[str]]:
    """语料目录 vs 索引：新增 / 修改 / 删除。

    只比 `size + mtime`，不重算 sha256（1.4 s 扫完 49 万文件，204 份命中）。
    这个检查回答的是"索引是不是旧的"，与 ``check`` 回答的"索引内部自洽吗"是两件事：
    实测踩过一次——语料在 10-27 加了 3 份文档，而索引还是 10-26 的，
    `stats` 一直显示 204，直到重新 ingest 才发现是 207。
    """
    from app.ingest.scan import rel_path_of, scan_corpus

    scan = scan_corpus(
        settings.corpus_root, settings.ext_whitelist, max_files=settings.max_files
    )
    current = {
        rel_path_of(path, settings.corpus_root): (path.stat().st_size, path.stat().st_mtime)
        for path in scan.files
    }
    indexed = {doc.rel_path: (doc.size, doc.mtime) for doc in docs}

    added = sorted(set(current) - set(indexed))
    removed = sorted(set(indexed) - set(current))
    changed = sorted(
        rel
        for rel in set(current) & set(indexed)
        if current[rel][0] != indexed[rel][0]
        or abs(current[rel][1] - indexed[rel][1]) > 1.0
    )
    return {"added": added, "removed": removed, "changed": changed}
