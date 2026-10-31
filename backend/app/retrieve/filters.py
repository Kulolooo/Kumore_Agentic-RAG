"""元数据过滤：把条件下推到检索层，而不是"取回 top-100 再在 Python 里筛"。

**为什么这是一个独立问题**：先取 top-k 再过滤，最坏情况是 k 条全部被滤光——
用户拿到空结果，而库里明明有符合条件的内容。RAGFlow 在 ES 里走 bool 子句、
在 Infinity 里走 ``JSON_CONTAINS``，都是**下推**（`03-检索与排序.md` §7.1）。
我们的向量腿是 numpy 暴力检索，下推就是给打分矩阵加一个布尔掩码；
BM25 腿则是先按条件缩候选、再截断 top-k。

**默认行为与 M0 保持一致**：只检索 ``status == "active"`` 的文档。
区别在于现在这个条件发生在打分/截断**之前**，所以废旧版本再多也不会挤掉
当前版本的候选位。

过滤条件刻意做成"文档级"的：扩展名、路径前缀、doc_id、mtime 区间、状态。
不支持按 chunk 内容过滤——那是检索打分该做的事，混进来会让两件事互相掩盖。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from app.schemas import Doc

#: 默认只搜当前版本。``superseded``（被更新版本取代）与 ``duplicate``（字节重复）
#: 都不该占用候选位；要看旧版本时显式放开。
ACTIVE_ONLY = ("active",)


def _norm_rel(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def _norm_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        return ext
    return ext if ext.startswith(".") else f".{ext}"


def parse_time(value: str | float | None) -> float | None:
    """把 ``2026-01-01`` / ``2026-01-01T12:00:00`` / epoch 秒转成 epoch 秒。

    只写日期时按**本地时区当天 00:00** 解释。测试里传 epoch，避免时区依赖。
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError as exc:
        raise ValueError(
            f"时间过滤条件无法解析：{value!r}；支持 2026-01-01 或 2026-01-01T12:00:00"
        ) from exc


@dataclass(frozen=True)
class DocFilter:
    """文档级过滤条件。所有字段为空 = 不附加额外条件（但默认仍只看 active）。"""

    exts: tuple[str, ...] = ()
    path_prefixes: tuple[str, ...] = ()
    doc_ids: frozenset[str] = frozenset()
    #: mtime >= since（epoch 秒）
    since: float | None = None
    #: mtime <= until（epoch 秒）
    until: float | None = None
    #: 允许的文档状态；空元组表示"不按状态过滤"。
    statuses: tuple[str, ...] = ACTIVE_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "exts", tuple(_norm_ext(e) for e in self.exts if e and e.strip())
        )
        object.__setattr__(
            self,
            "path_prefixes",
            tuple(_norm_rel(p) for p in self.path_prefixes if p and p.strip()),
        )
        object.__setattr__(self, "statuses", tuple(s.strip() for s in self.statuses if s.strip()))

    @property
    def is_empty(self) -> bool:
        """没有任何条件（含状态）时才可以跳过构造允许集合。"""
        return not (
            self.exts
            or self.path_prefixes
            or self.doc_ids
            or self.since is not None
            or self.until is not None
            or self.statuses
        )

    def with_statuses(self, statuses: tuple[str, ...]) -> "DocFilter":
        return replace(self, statuses=statuses)

    def matches(self, doc: Doc) -> bool:
        if self.statuses and doc.status not in self.statuses:
            return False
        if self.exts and _norm_ext(doc.ext) not in self.exts:
            return False
        if self.doc_ids and doc.doc_id not in self.doc_ids:
            return False
        if self.path_prefixes:
            rel = _norm_rel(doc.rel_path)
            if not any(rel.startswith(prefix) for prefix in self.path_prefixes):
                return False
        if self.since is not None and (doc.mtime or 0.0) < self.since:
            return False
        if self.until is not None and (doc.mtime or 0.0) > self.until:
            return False
        return True

    def describe(self) -> str:
        """一行可写进报告的描述；空条件显示"无"。"""
        parts: list[str] = []
        if self.statuses:
            parts.append("/".join(self.statuses))
        if self.exts:
            parts.append("ext:" + ",".join(self.exts))
        if self.path_prefixes:
            parts.append("path:" + ",".join(self.path_prefixes))
        if self.doc_ids:
            parts.append(f"docs:{len(self.doc_ids)}")
        if self.since is not None:
            parts.append(f"since:{_fmt_time(self.since)}")
        if self.until is not None:
            parts.append(f"until:{_fmt_time(self.until)}")
        return " ".join(parts) if parts else "无"


def _fmt_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")


def from_args(
    *,
    ext: str | None = None,
    path_prefix: str | None = None,
    since: str | float | None = None,
    until: str | float | None = None,
    include_superseded: bool = False,
) -> DocFilter | None:
    """CLI 参数的薄封装：没有任何"额外"条件时返回 None（检索层走零成本默认路径）。

    注意：默认的 ``status=active`` 由 ``Retriever.search`` 自己保证，
    这里只为"显式放开旧版本"或"附加了扩展名/路径/时间条件"才构造对象。
    """
    exts = tuple(e for e in (ext or "").split(",") if e.strip())
    prefixes = tuple(p for p in (path_prefix or "").split(",") if p.strip())
    since_ts = parse_time(since)
    until_ts = parse_time(until)
    if not (exts or prefixes or since_ts is not None or until_ts is not None):
        return None
    return DocFilter(
        exts=exts,
        path_prefixes=prefixes,
        since=since_ts,
        until=until_ts,
        statuses=() if include_superseded else ACTIVE_ONLY,
    )


def allowed_chunk_ids(
    doc_filter: DocFilter | None,
    docs_by_id: dict[str, Doc],
    chunks: list,
) -> set[str] | None:
    """把文档级条件展开成 chunk 白名单。

    ``None`` = 不限制（检索层可跳过掩码计算）。百份级语料上这个集合是
    毫秒级开销；真到十万 chunk 再考虑把它预计算进索引。
    """
    if doc_filter is None:
        return None
    if not docs_by_id:
        # 没有文档元数据（单元测试里的裸 chunk 集）时无从判断，
        # 返回 None 表示"不过滤"，而不是把整个语料排除掉。
        return None
    allowed_docs = {doc_id for doc_id, doc in docs_by_id.items() if doc_filter.matches(doc)}
    return {chunk.chunk_id for chunk in chunks if chunk.doc_id in allowed_docs}
