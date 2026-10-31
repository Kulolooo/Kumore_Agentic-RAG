"""检索层：BM25 / 向量 / 混合，三种模式共用一个入口。

`agent`（后续阶段）只允许通过这一层访问检索，不直接碰索引，
这样换检索方案时上层代码不用改。

已修复的两个实测缺陷仍在这里：

1. **同族占满 top-k** —— 同一份文档的 docx/pdf 副本、或同族多版本，会把 5 个名额吃光。
   用 ``max_per_family`` 按"版本族"限流。
2. **版本冲突静默选一版** —— 冲突的一方不在上下文里时模型无从知晓。
   用 ``expand_versions`` 把同族其他版本的最佳片段作为 ``extras`` 补进上下文。

M0-1 新增：``mode = bm25 | vector | hybrid``，以及两种融合方式
（``rrf`` 免调参 / ``weighted`` 即 RAGFlow 主链用的加权和）。
每条结果带 ``source``，回答"这条是被词法还是向量召回的"。
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings
from app.embed.local import LocalEmbedder
from app.embed.store import VectorIndex
from app.ingest.store import load_images, load_index
from app.retrieve.bm25 import BM25Index
from app.retrieve.filters import DocFilter, allowed_chunk_ids, ACTIVE_ONLY
from app.retrieve.rerank import CrossEncoderReranker, Reranker
from app.retrieve.versions import VersionInfo, build_version_map
from app.schemas import AlsoIn, Chunk, Doc, Retrieved, make_content_hash

log = logging.getLogger(__name__)

VALID_MODES = ("bm25", "vector", "hybrid")


def _minmax(scores: list[tuple[str, float]]) -> dict[str, float]:
    """把一路分数压到 [0,1]。BM25 与余弦量纲差得远，不加权归一化就没法相加。"""
    if not scores:
        return {}
    values = [s for _, s in scores]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return {cid: 1.0 for cid, _ in scores}
    return {cid: (s - lo) / span for cid, s in scores}


def _zscore(scores: list[tuple[str, float]]) -> dict[str, float]:
    """按本路候选的分布做标准化（均值 0、标准差 1）。

    与 min-max 的区别是**不看极值、看分布**：min-max 会把每一路的第 1 名都抬到 1.0，
    哪怕这一路整体都很平（全是无关命中）；z-score 下"平"的腿最高也只有 2 个标准差左右，
    分不到 0.3 的满权重。这是对"融合权重被极值绑架"的一个直接修补。
    标准差为 0 时全部记 0（等价于"这一路没有区分度"）。
    """
    if not scores:
        return {}
    values = [s for _, s in scores]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    if std <= 1e-9:
        return {cid: 0.0 for cid, _ in scores}
    return {cid: (s - mean) / std for cid, s in scores}


@dataclass
class SearchResult:
    query: str
    #: 主结果，长度 <= top_k，检索指标只用这部分
    hits: list[Retrieved] = field(default_factory=list)
    #: 同族其他版本补进来的片段（最多 2 条），只为回答服务，不计入 Hit@k
    extras: list[Retrieved] = field(default_factory=list)
    #: 被版本过滤掉的命中数，版本冲突题里要看这个数字
    dropped_superseded: int = 0
    #: 因"同族限流"被跳过的 chunk 数，用于观察限流是否过严
    dropped_family: int = 0
    #: 因"与已选中的某条内容完全相同"被折叠掉的 chunk 数
    dropped_duplicate: int = 0
    #: 实际使用的检索模式与融合方式，报告里要写清楚
    mode: str = "bm25"
    fusion: str = ""
    #: 元数据过滤条件的一行描述（含默认的 status=active）；空字符串表示老行为
    doc_filter: str = ""
    #: 重排状态：off（未启用）/ ok / failed（已 fail-open 回退原序）
    rerank_status: str = "off"
    rerank_error: str = ""
    rerank_ms: float | None = None
    rerank_candidates: int = 0
    #: 文本塔未预热时，hybrid/vector 自动回退 BM25 的原因。
    fallback_reason: str = ""

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk.chunk_id for h in self.hits]

    @property
    def answer_hits(self) -> list[Retrieved]:
        """喂给模型的全部上下文：主结果 + 版本补充。"""
        return [*self.hits, *self.extras]


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        docs_by_id: dict[str, Doc],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        max_per_family: int = 3,
        vector_index: VectorIndex | None = None,
        embedder: LocalEmbedder | None = None,
        visual_index: VectorIndex | None = None,
        visual_embedder: LocalEmbedder | None = None,
        images_by_id: dict | None = None,
        mode: str = "bm25",
        fusion: str = "rrf",
        vector_weight: float = 0.3,
        rrf_k: int = 60,
        knn_top_k: int = 100,
        chunk_strategy: str = "",
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
        rerank_fail_open: bool = True,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode 只能是 {VALID_MODES}，当前 {mode!r}")
        self.chunk_strategy = chunk_strategy
        self.chunks = chunks
        self.docs_by_id = docs_by_id
        self.max_per_family = max_per_family
        self.vector_index = vector_index
        self.embedder = embedder
        self.visual_index = visual_index
        self.visual_embedder = visual_embedder
        self.images_by_id = images_by_id or {}
        self.mode = mode
        self.fusion = fusion
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.knn_top_k = knn_top_k
        self.reranker = reranker
        self.rerank_candidates = max(1, rerank_candidates)
        self.rerank_fail_open = rerank_fail_open
        # 父子块策略：父块只用于回填上下文，不参与任何一路召回。若把父块也放进
        # BM25/向量，top-k 会被粗粒度的父块占满，检索指标也就失去意义。
        self._parents = {c.chunk_id: c for c in chunks if c.is_parent}
        searchable = [c for c in chunks if not c.is_parent]
        # 老索引没有内容指纹：按正文补算。不补的话，下面的"按内容折叠"
        # 会把所有空指纹当成同一段话，把结果折成一条。
        for chunk in searchable:
            if not chunk.content_hash:
                chunk.content_hash = make_content_hash(chunk.text)

        self.index = BM25Index(k1=k1, b=b)
        # 词法腿带上文档标题：语料里大量"同名不同格式/不同版本"的副本，
        # 正文里往往不出现文件名里的信息（如 BOM 的板名、任务书版本号），
        # 只索引正文会让"问文件名里的事"这类问题召回不到。
        # 只影响 BM25 的可检索文本，不改 chunk.text / content_hash，无需重建向量。
        self.index.build((c.chunk_id, self._bm25_text(c)) for c in searchable)
        self._by_chunk_id = {c.chunk_id: c for c in searchable}
        self._chunk_doc = {c.chunk_id: c.doc_id for c in searchable}
        self._doc_by_rel = {d.rel_path: d for d in docs_by_id.values()}
        # 视觉索引存 image_id；检索返回仍必须是文本 chunk。
        self._image_to_chunk = {}
        for chunk in searchable:
            if chunk.image_id:
                self._image_to_chunk.setdefault(chunk.image_id, chunk.chunk_id)
        # 版本族是**派生信息**，不入库：只依赖文档标题与路径，随时可重算。
        self.versions: dict[str, VersionInfo] = build_version_map(docs_by_id.values())

    @classmethod
    def from_storage(cls, settings: Settings) -> "Retriever":
        from app.ingest.verify import raise_if_any

        docs, chunks = load_index(settings)
        vector_index: VectorIndex | None = None
        embedder: LocalEmbedder | None = None
        visual_index: VectorIndex | None = None
        visual_embedder: LocalEmbedder | None = None
        images_by_id: dict = {}

        if settings.retrieval_mode != "bm25":
            path = settings.vector_index_path
            if not path.is_file():
                raise FileNotFoundError(
                    f"未找到文本向量索引：{path}\n先运行：python -m app embed"
                )
            vector_index = VectorIndex.load(path)
            if vector_index.model_name and vector_index.model_name != settings.embedding_model:
                raise ValueError(
                    f"文本向量索引用的是 {vector_index.model_name}，"
                    f"但当前配置是 {settings.embedding_model}；换模型必须重建索引"
                )
            embedder = LocalEmbedder(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                model_dir=settings.model_dir,
            )

            # 视觉索引可选：没有就只用文本两路，不阻塞冷启动。
            visual_path = settings.visual_vector_index_path
            if visual_path.is_file():
                try:
                    candidate = VectorIndex.load(visual_path)
                    if (
                        candidate.model_name
                        and candidate.model_name == settings.visual_embedding_model
                    ):
                        visual_index = candidate
                        # 图找图已移除；文本→图用 bge-m3 的共享文本空间查询，
                        # 不把 1.75GB 视觉塔再加载进检索进程。
                        images_by_id = {i.image_id: i for i in load_images(settings)}
                except Exception as exc:  # noqa: BLE001 - 视觉索引坏了应回退文本
                    log.warning("视觉索引加载失败，回退文本检索：%s", exc)

        # 对不上就抛错：静默降级会让"M1 改了分块"被误读成"混合检索变差"。
        raise_if_any(settings, docs, chunks, vector_index)

        reranker: CrossEncoderReranker | None = None
        if settings.rerank:
            reranker = CrossEncoderReranker(
                model_name=settings.rerank_model,
                device=settings.rerank_device,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
                model_dir=settings.model_dir,
            )

        return cls(
            chunks,
            {d.doc_id: d for d in docs},
            k1=settings.bm25_k1,
            b=settings.bm25_b,
            max_per_family=settings.max_per_family,
            vector_index=vector_index,
            embedder=embedder,
            visual_index=visual_index,
            visual_embedder=visual_embedder,
            images_by_id=images_by_id,
            mode=settings.retrieval_mode,
            fusion=settings.fusion,
            vector_weight=settings.vector_weight,
            rrf_k=settings.rrf_k,
            knn_top_k=settings.knn_top_k,
            chunk_strategy=settings.chunk_strategy,
            reranker=reranker,
            rerank_candidates=settings.rerank_candidates,
            rerank_fail_open=settings.rerank_fail_open,
        )

    @property
    def text_ready(self) -> bool:
        """BM25 永远 ready；文本向量模式在模型预热完成前不 ready。"""
        if self.embedder is None:
            return True
        return bool(getattr(self.embedder, "is_ready", True))

    def prewarm(self) -> None:
        if self.embedder is not None:
            loader = getattr(self.embedder, "prewarm", None)
            if loader is not None:
                loader()

    def doc_of(self, chunk: Chunk) -> Doc | None:
        return self.docs_by_id.get(chunk.doc_id)

    def _bm25_text(self, chunk: Chunk) -> str:
        """BM25 实际索引的文本 = 文档标题 + 正文（标题已在正文里就不重复）。

        标题是**每文档**的元数据，把它算进词法腿是通用做法：换任何语料，
        "文件名里写着、正文没写"的信息都能被检索到；没有为某份文档写特例。
        不塞进 ``chunk.text``，所以 content_hash、引用展示、向量索引都不受影响。
        """
        doc = self.docs_by_id.get(chunk.doc_id)
        title = (doc.title or "").strip() if doc is not None else ""
        if title and title not in chunk.text:
            return f"{title}\n{chunk.text}"
        return chunk.text

    def _family_key(self, doc_id: str) -> str:
        """同族判定：有版本族就按族算，否则这份文档自成一族。"""
        info = self.versions.get(doc_id)
        return info.group_key if info is not None else f"doc:{doc_id}"

    # ------------------------------------------------------------------ 召回
    def _vector_rank(
        self, query: str, pool: int, allowed: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if self.vector_index is None or self.embedder is None:
            raise RuntimeError(
                "vector / hybrid 模式需要向量索引与本地 embedder；"
                "请先 python -m app embed，并确认 KUMORE_RETRIEVAL_MODE 设置正确"
            )
        query_vec = self.embedder.encode_query(query)
        mask = None
        if allowed is not None:
            # 掩码与向量索引的行一一对应；下推后 top_k 只在允许的文档里选，
            # 不会出现"取回 100 条、过滤后只剩 2 条"的候选饥饿。
            mask = np.fromiter(
                (cid in allowed for cid in self.vector_index.chunk_ids),
                dtype=bool,
                count=len(self.vector_index.chunk_ids),
            )
        return self.vector_index.search(query_vec, top_k=pool, mask=mask)

    def _visual_rank(
        self, query: str, pool: int, allowed: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """视觉索引返回 image_id；映射回同页的文本 chunk，避免第二套 Retrieved 协议。"""
        if self.visual_index is None or self.embedder is None:
            return []
        query_vec = self.embedder.encode_query(query)
        hits = self.visual_index.search(query_vec, top_k=pool)
        out: list[tuple[str, float]] = []
        for image_id, score in hits:
            chunk_id = self._image_to_chunk.get(image_id)
            if chunk_id is None:
                continue
            if allowed is not None and chunk_id not in allowed:
                continue
            out.append((chunk_id, score))
        return out

    def _fuse_many(
        self, rankings: list[tuple[str, list[tuple[str, float]]]]
    ) -> list[tuple[str, float, str]]:
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, list[str]] = defaultdict(list)
        for name, ranked in rankings:
            for rank, (cid, _score) in enumerate(ranked, start=1):
                scores[cid] += 1.0 / (self.rrf_k + rank)
                sources[cid].append(name)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(cid, score, "+".join(sources[cid])) for cid, score in ordered]

    def _fuse(
        self,
        bm25: list[tuple[str, float]],
        vector: list[tuple[str, float]],
    ) -> list[tuple[str, float, str]]:
        """两路合并。返回 (chunk_id, score, source)。"""
        if self.fusion in ("weighted", "zscore"):
            norm = _zscore if self.fusion == "zscore" else _minmax
            norm_bm = norm(bm25)
            norm_vec = norm(vector)
            weight = self.vector_weight
            merged: dict[str, float] = {}
            sources: dict[str, list[str]] = {}
            for cid in set(norm_bm) | set(norm_vec):
                # 缺失的那一路按 0（min-max 的下界 / z-score 的均值）参与，
                # 不能用"只算一路"的方式，否则单路命中会被系统性抬高。
                merged[cid] = (1 - weight) * norm_bm.get(cid, 0.0) + weight * norm_vec.get(cid, 0.0)
                sources[cid] = [name for name, table in (("bm25", norm_bm), ("vector", norm_vec)) if cid in table]
            ordered = sorted(merged.items(), key=lambda kv: -kv[1])
            return [(cid, score, "+".join(sources[cid])) for cid, score in ordered]

        # RRF（默认）：只用名次，不用分数，免调参
        scores: dict[str, float] = defaultdict(float)
        sources = {}
        for name, ranked in (("bm25", bm25), ("vector", vector)):
            for rank, (cid, _score) in enumerate(ranked, start=1):
                scores[cid] += 1.0 / (self.rrf_k + rank)
                sources.setdefault(cid, set()).add(name)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(cid, score, "+".join(sorted(sources[cid]))) for cid, score in ordered]

    def _rank(
        self, query: str, mode: str, pool: int, allowed: set[str] | None = None
    ) -> list[tuple[str, float, str]]:
        if mode == "bm25":
            ranked = self.index.score_all(query)
            if allowed is not None:
                # 先缩候选、再截断：顺序反过来就是"先取 top-k 再过滤"，
                # 被过滤光时结果为空（RAGFlow 在 ES 里也是下推，`03` §7.1）。
                ranked = [(cid, s) for cid, s in ranked if cid in allowed]
            return [(cid, s, "bm25") for cid, s in ranked[:pool]]
        if mode == "vector":
            text = self._vector_rank(query, pool, allowed)
            visual = self._visual_rank(query, pool, allowed)
            if visual:
                return self._fuse_many([("vector", text), ("visual", visual)])
            return [(cid, s, "vector") for cid, s in text]
        bm = self._rank(query, "bm25", pool, allowed)
        text = self._vector_rank(query, pool, allowed)
        visual = self._visual_rank(query, pool, allowed)
        if visual:
            return self._fuse_many(
                [
                    ("bm25", [(cid, s) for cid, s, _ in bm]),
                    ("vector", text),
                    ("visual", visual),
                ]
            )
        return self._fuse([(cid, s) for cid, s, _ in bm], text)

    def _maybe_rerank(
        self, query: str, raw: list[tuple[str, float, str]]
    ) -> tuple[list[tuple[str, float, str]], str, str, float | None, int]:
        """对前 ``rerank_candidates`` 条重排；失败按配置 fail-open 回退原序。

        返回 ``(排序后的原始表, 状态, 错误, 耗时 ms, 实际重排条数)``。
        **不会**因为重排而改变候选集合：只在被重排的那一段内换顺序，
        其余条目原样接在后面——这样"重排到底动了什么"可被逐题复盘。
        """
        if self.reranker is None or not raw:
            return raw, "off", "", None, 0
        head = raw[: min(self.rerank_candidates, len(raw))]
        started = time.perf_counter()
        try:
            # 取正文也放进 try：向量索引与 chunk 集万一失配，这里会 KeyError，
            # 同样应该走 fail-open，而不是把一次检索打成 500。
            texts = [self._by_chunk_id[cid].text for cid, _, _ in head]
            scores = self.reranker.rerank(query, texts)
        except Exception as exc:  # noqa: BLE001 - 任何重排故障都不该打穿检索
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            if not self.rerank_fail_open:
                raise
            log.warning("重排失败，保持检索原序：%s: %s", type(exc).__name__, exc)
            return raw, "failed", f"{type(exc).__name__}: {exc}", elapsed, 0
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        order = sorted(range(len(head)), key=lambda i: -scores[i])
        reranked = [(head[i][0], scores[i], head[i][2]) for i in order]
        return reranked + raw[len(head) :], "ok", "", elapsed, len(head)

    # ------------------------------------------------------------------ 查询
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        include_superseded: bool = False,
        relative_cutoff: float = 0.0,
        max_per_family: int | None = None,
        expand_versions: bool = True,
        sibling_min_ratio: float = 0.25,
        mode: str | None = None,
        doc_filter: DocFilter | None = None,
    ) -> SearchResult:
        """检索 + 元数据过滤下推 + 同族限流 + 版本补充（+ 可选重排）。

        ``relative_cutoff``：丢掉得分低于 (最高分 × cutoff) 的命中。中文按字切词时，
        "的/是/不"这类单字会让明显无关的文档也拿到零点几分。默认 0（不裁剪）。
        有重排时先按原分数裁剪、再重排——两种分数不可比，混着用会变成玄学。

        ``max_per_family``：同族最多取几个 chunk。``None`` 用构造时的默认值。
        ``expand_versions``：把同族其他版本的最佳片段补进 ``extras``。
        ``mode``：覆盖构造时的检索模式（bm25 / vector / hybrid）。
        ``doc_filter``：文档级过滤条件（扩展名 / 来源 / 时间 / 状态），**在打分截断前**
        生效。默认只看 active 文档；``include_superseded=True`` 会覆盖状态这一项。
        """
        use_mode = mode or self.mode
        fallback_reason = ""
        if use_mode in ("vector", "hybrid") and not self.text_ready:
            # 冷启动不阻塞：模型在后台预热；当前 turn 先用 BM25 回答。
            use_mode = "bm25"
            fallback_reason = "text_embedding_loading"
        result = SearchResult(
            query=query,
            mode=use_mode,
            fusion=self.fusion if use_mode == "hybrid" else "",
            fallback_reason=fallback_reason,
        )
        if not self.chunks or top_k <= 0:
            return result

        # 过滤条件先落成 chunk 白名单，再交给两路召回做下推。
        effective = doc_filter or DocFilter()
        if include_superseded:
            effective = effective.with_statuses(())
        result.doc_filter = effective.describe()
        allowed = allowed_chunk_ids(
            effective, self.docs_by_id, list(self._by_chunk_id.values())
        )

        pool = max(self.knn_top_k, top_k * 8)
        raw = self._rank(query, use_mode, pool, allowed)
        if not raw:
            return result

        if relative_cutoff > 0:
            floor = raw[0][1] * relative_cutoff
            raw = [item for item in raw if item[1] >= floor]
            if not raw:
                return result

        raw_orig = raw
        raw, result.rerank_status, result.rerank_error, result.rerank_ms, result.rerank_candidates = (
            self._maybe_rerank(query, raw)
        )
        # 版本补充仍按**重排前**的检索分数挑：重排分数是另一把尺子（可能为负），
        # 拿它的比例阈值去比原始分，会把同族补片补成噪声。
        raw_for_extras = raw if result.rerank_status != "ok" else raw_orig

        limit = self.max_per_family if max_per_family is None else max_per_family

        counts: dict[str, int] = defaultdict(int)
        #: 内容指纹 → 已选中的那条。同一段话只占一个名额，其余记进 also_in。
        seen_content: dict[str, Retrieved] = {}
        for chunk_id, score, source in raw:
            chunk = self._by_chunk_id.get(chunk_id)
            if chunk is None:
                continue
            doc = self.docs_by_id.get(chunk.doc_id)
            # 下推之后这里通常不会命中；保留是为了 allowed 为空集（不过滤状态）时兜底。
            if doc is not None and doc.status != "active" and not include_superseded:
                result.dropped_superseded += 1
                continue
            content = chunk.content_hash or make_content_hash(chunk.text)
            duplicate = seen_content.get(content)
            if duplicate is not None:
                duplicate.also_in.append(
                    AlsoIn(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        rel_path=doc.rel_path if doc is not None else "",
                        page=chunk.page,
                    )
                )
                result.dropped_duplicate += 1
                continue
            family = self._family_key(chunk.doc_id)
            if limit and counts[family] >= limit:
                result.dropped_family += 1
                continue
            counts[family] += 1
            hit = Retrieved(chunk=chunk, score=score, source=source)
            if chunk.mom_id:
                hit.parent = self._parents.get(chunk.mom_id)
            seen_content[content] = hit
            result.hits.append(hit)
            if len(result.hits) >= top_k:
                break

        for rank, hit in enumerate(result.hits, start=1):
            hit.rank = rank

        if expand_versions:
            result.extras = self._version_extras(
                [(cid, score) for cid, score, _ in raw_for_extras],
                result.hits,
                sibling_min_ratio,
                include_superseded,
                seen_content=set(seen_content),
                best_score=raw_for_extras[0][1] if raw_for_extras else None,
            )
            for offset, extra in enumerate(result.extras, start=len(result.hits) + 1):
                extra.rank = offset
        return result

    def _version_extras(
        self,
        raw: list[tuple[str, float]],
        hits: list[Retrieved],
        min_ratio: float,
        include_superseded: bool = False,
        seen_content: set[str] | None = None,
        best_score: float | None = None,
    ) -> list[Retrieved]:
        """给每个出现在主结果里的版本族，补上其他版本的最佳片段。

        只在补进来的片段**分数不低于主结果最高分 × min_ratio** 时才补，
        避免把不相关版本的噪声硬塞进上下文。内容与已有结果完全相同的也不补。

        ``best_score``：重排开启时主结果的分数量纲已经变了，必须由调用方传入
        **重排前**的最高分，否则阈值失去意义。
        """
        if not hits:
            return []

        represented = {h.chunk.doc_id for h in hits}
        seen_content = set(seen_content or ())
        floor = (best_score if best_score is not None else hits[0].score) * min_ratio

        best_per_doc: dict[str, tuple[str, float]] = {}
        for chunk_id, score in raw:
            if score < floor:
                break
            doc_id = self._chunk_doc.get(chunk_id)
            if doc_id is None or doc_id in represented or doc_id in best_per_doc:
                continue
            best_per_doc[doc_id] = (chunk_id, score)

        extras: list[Retrieved] = []
        seen_groups: set[str] = set()
        for hit in hits:
            info = self.versions.get(hit.chunk.doc_id)
            if info is None or not info.is_multi_version:
                continue
            if info.group_key in seen_groups:
                continue
            seen_groups.add(info.group_key)
            for sibling_rel in info.siblings:
                sibling = self._doc_by_rel.get(sibling_rel)
                if sibling is None or sibling.doc_id in represented:
                    continue
                if sibling.status != "active" and not include_superseded:
                    continue
                found = best_per_doc.get(sibling.doc_id)
                if found is None:
                    continue
                chunk = self._by_chunk_id.get(found[0])
                if chunk is None:
                    continue
                content = chunk.content_hash or make_content_hash(chunk.text)
                if content in seen_content:
                    continue
                seen_content.add(content)
                extra = Retrieved(chunk=chunk, score=found[1], source="version_sibling")
                if chunk.mom_id:
                    extra.parent = self._parents.get(chunk.mom_id)
                extras.append(extra)
                represented.add(sibling.doc_id)
                if len(extras) >= 2:
                    return extras
        return extras

    @property
    def stats(self) -> dict[str, object]:
        data: dict[str, object] = {
            "docs": len(self.docs_by_id),
            "chunks": len(self._by_chunk_id),
            "parents": len(self._parents),
            "n_terms": self.index.stats.n_terms,
            "avg_doc_len": round(self.index.stats.avg_doc_len, 1),
            "version_families": len({v.group_key for v in self.versions.values()}),
            "mode": self.mode,
            "rerank": bool(self.reranker is not None),
        }
        if self.reranker is not None:
            data["rerank_model"] = getattr(self.reranker, "name", "?")
            data["rerank_candidates"] = self.rerank_candidates
        if self.vector_index is not None:
            data["vector"] = self.vector_index.stats()
        return data
