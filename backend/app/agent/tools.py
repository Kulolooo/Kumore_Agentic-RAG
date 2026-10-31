"""四个工具的执行层。

**所有检索都走 ``Retriever``**，不直接碰索引/存储——换检索方案时工具层不用改
（这是 ``retriever.py`` 开头就定下的分工）。

三个与既有代码共享的不变量：

1. **``[ID:n]`` 由 ``EvidenceRegistry`` 发号**，而且一次 ask 只有一个注册表实例。
   M3 时它是"每次回答新建"（``answer.py:253``）；M4 改成跨步骤存活，这样第 3 步发出的
   ``[ID:7]`` 在最终回答里仍然指第 3 步读到的那一块——否则多步引用无法回溯到步。
2. **隐私边界不放松**：工具返回值里**没有文件名、没有路径**。文档用本次 ask 内稳定的
   代号 ``D0``/``D1`` 指代，模型据此做消歧（``scope=D3``），但拿不到标题。
   为什么不能给标题：实测 66/202 份文件名含 11 位学号，姓名更常见（``prompt.py:144``）。
   代号的另一个好处是模型不会把文件名当证据写进答案。
3. **失败不抛穿**：任一工具出错都返回 ``ok=False`` + 一行错误说明，由循环决定降级，
   绝不让一次工具异常把整次问答打成 500。

去重口径沿用 ``ServeText``：同一块第二次被读到时只给编号、不再重复正文
（``evidence.py:96-109``），省 token 也省模型注意力。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.agent.toolspec import LIMITS, MAX_EVIDENCE, TOOL_TEXT_MAX
from app.generate.evidence import NOT_CITABLE, EvidenceRegistry
from app.generate.prompt import _neutralize_handles
from app.retrieve.retriever import Retriever
from app.schemas import Chunk, Retrieved

_WS = re.compile(r"\s+")
_ID_NUMBER = re.compile(r"(?:ID\s*[:：]\s*)?(\d{1,3})", re.IGNORECASE)
_DOC_CODE = re.compile(r"^[Dd](\d{1,3})$")


def _flat(text: str) -> str:
    return _WS.sub("", text or "")


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ToolResult:
    """一次工具调用的结果。``text`` 会原样回灌给模型。"""

    text: str
    ok: bool = True
    #: 本步**新**发号的块数（去重后）
    n_new: int = 0
    #: 本步新发出的编号
    new_ids: list[int] = field(default_factory=list)
    #: 本步涉及的文档代号（新见的也记上，便于 trace 复盘）
    docs: list[str] = field(default_factory=list)
    #: 本步**新发号**的块来自哪条召回腿（bm25 / vector / bm25+vector / grep / read…）。
    #: M5 的归因要回答"这一步是词法还是向量召回的"，只有块号不够。
    sources: list[str] = field(default_factory=list)
    detail: str = ""
    #: 工具执行耗时（毫秒）；失败时也有值，便于区分"慢"和"错"
    ms: float = 0.0


class DocHandles:
    """本次 ask 内的文档代号（``D0``、``D1``…），首见序发号。

    与证据编号同一个哲学：模型只能引用它见过的代号，所以代号必须在文档第一次出现在
    工具结果里的那一刻定下来。``doc_id`` 是路径哈希，人不可读，也不该发给模型。
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._code_of: dict[str, str] = {}

    def code(self, doc_id: str) -> str:
        if not doc_id:
            return ""
        if doc_id not in self._code_of:
            code = f"D{len(self._order)}"
            self._order.append(doc_id)
            self._code_of[doc_id] = code
        return self._code_of[doc_id]

    def doc_id(self, code: str) -> str | None:
        """代号 → doc_id；也接受直接给 doc_id（模型可能抄了长 id）。"""
        code = (code or "").strip()
        if code in self._order:
            return code
        match = _DOC_CODE.match(code)
        if match:
            index = int(match.group(1))
            return self._order[index] if 0 <= index < len(self._order) else None
        return None

    def known(self, doc_id: str) -> bool:
        return doc_id in self._code_of

    def codes(self) -> dict[str, str]:
        return dict(self._code_of)

    def __len__(self) -> int:
        return len(self._order)


class ToolBox:
    """执行四个工具，并把命中写进共享的证据注册表。"""

    def __init__(
        self,
        retriever: Retriever,
        registry: EvidenceRegistry | None = None,
        handles: DocHandles | None = None,
        *,
        top_k: int = 5,
        include_superseded: bool = False,
    ) -> None:
        self.retriever = retriever
        self.registry = registry or EvidenceRegistry()
        self.handles = handles or DocHandles()
        self.top_k = top_k
        self.include_superseded = include_superseded
        #: chunk_id → Retrieved。最终回答的引用要用它回填页码/小节/正文。
        self.evidence: dict[str, Retrieved] = {}
        #: 发号顺序对应的 Retrieved（``[ID:n]`` → ``evidence_order[n]``）
        self.evidence_order: list[Retrieved] = []
        #: chunk_id → (步号, 工具名)：验收要求"每条 claim 能回指到某一步"
        self.served_at: dict[str, tuple[int, str]] = {}
        #: 由循环在每次工具调用前设置，用于 ``served_at`` 记账
        self.current_step: int = 0
        #: 被截断掉的证据块数（超过 MAX_EVIDENCE 之后不再发号）
        self.overflow = 0

    # ---------------------------------------------------------------- 发号
    def _serve(self, hit: Retrieved) -> tuple[int, bool]:
        """把一条命中交给注册表；返回 (编号, 是否有新正文)。"""
        chunk = hit.context_chunk
        text = chunk.text or ""
        number = self.registry.stamp(hit.chunk.chunk_id, text=text)
        if number == NOT_CITABLE:
            return NOT_CITABLE, False
        fresh = number >= len(self.evidence_order)
        if fresh:
            if len(self.evidence_order) >= MAX_EVIDENCE:
                self.overflow += 1
                return NOT_CITABLE, False
            self.evidence_order.append(hit)
            self.evidence[hit.chunk.chunk_id] = hit
        else:
            self.evidence.setdefault(hit.chunk.chunk_id, hit)
        self.handles.code(hit.chunk.doc_id)
        self.served_at.setdefault(hit.chunk.chunk_id, (self.current_step, hit.source or ""))
        return number, fresh

    def _block(self, number: int, hit: Retrieved, *, fresh: bool) -> str:
        chunk = hit.context_chunk
        head = [f"[ID:{number}]", self.handles.code(chunk.doc_id)]
        page = chunk.page
        if page is not None:
            head.append(
                f"第 {page}–{chunk.page_end} 页"
                if chunk.page_end is not None and chunk.page_end != page
                else f"第 {page} 页"
            )
        if chunk.section:
            head.append(f"小节：{chunk.section}")
        line = "｜".join(part for part in head if part)
        if not fresh:
            # 同一块已经发过正文：只给编号，避免重复占上下文（ServeText 的语义）
            return f"{line}（正文同上，见前文该编号）"
        return f"{line}\n{_neutralize_handles(_clip(chunk.text.strip(), TOOL_TEXT_MAX))}"

    # ---------------------------------------------------------------- 分发
    def run(self, name: str, args: dict[str, Any] | None, *, args_raw: str = "") -> ToolResult:
        args = dict(args or {})
        started = time.perf_counter()
        try:
            if name == "search_kb":
                result = self._search_kb(args)
            elif name == "grep_kb":
                result = self._grep_kb(args)
            elif name == "read_section":
                result = self._read_section(args)
            elif name == "list_docs":
                result = self._list_docs(args)
            else:
                result = ToolResult(
                    text=(
                        f"未知工具 {name!r}；可用：search_kb / grep_kb / read_section / list_docs"
                    ),
                    ok=False,
                    detail=f"未知工具 {name}",
                )
        except Exception as exc:  # noqa: BLE001 - 任何工具故障都不该打穿循环
            result = ToolResult(
                text=f"工具 {name} 执行失败：{type(exc).__name__}: {exc}",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        result.ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    @staticmethod
    def _bounded(args: dict[str, Any], tool: str) -> dict[str, int]:
        """把模型给的数字收进 [默认值, 上限]，并把非整数当成默认值。

        模型偶尔会写 ``"top_k": "5"``（字符串）或 ``1e9``。这些不该让调用失败——
        钳到安全区比报错更有用，也不需要为它多花一次模型调用。
        """
        limits = LIMITS[tool]
        out: dict[str, int] = {}
        for key in ("top_k", "max_hits", "limit"):
            if key not in limits:
                continue
            default, cap = limits[key]
            raw = args.get(key)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = default
            out[key] = max(1, min(value, cap))
        return out

    # ---------------------------------------------------------------- 工具
    def _search_kb(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult("search_kb 需要非空的 query", ok=False, detail="query 为空")
        bounds = self._bounded(args, "search_kb")
        top_k = bounds.get("top_k", self.top_k)

        doc_filter = None
        scope = str(args.get("scope") or "").strip()
        if scope:
            doc_id = self.handles.doc_id(scope)
            if doc_id is None:
                return ToolResult(
                    f"未知文档代号 {scope!r}；先用 list_docs 看有哪些代号",
                    ok=False,
                    detail=f"未知 scope {scope}",
                )
            from app.retrieve.filters import DocFilter

            doc_filter = DocFilter(doc_ids=frozenset({doc_id}))

        result = self.retriever.search(
            query,
            top_k=top_k,
            include_superseded=self.include_superseded,
            doc_filter=doc_filter,
        )
        if not result.hits:
            return ToolResult(
                f"search_kb({query!r}{'，scope=' + scope if scope else ''}) 没有命中",
                n_new=0,
                detail="零命中",
            )
        blocks: list[str] = []
        new_ids: list[int] = []
        docs: list[str] = []
        sources: set[str] = set()
        for hit in result.hits:
            number, fresh = self._serve(hit)
            if number == NOT_CITABLE:
                continue
            if fresh:
                new_ids.append(number)
                if hit.source:
                    sources.add(hit.source)
            docs.append(self.handles.code(hit.chunk.doc_id))
            blocks.append(self._block(number, hit, fresh=fresh))
        detail = f"命中 {len(result.hits)} 条"
        if result.dropped_family:
            detail += f"；同族限流跳过 {result.dropped_family} 条"
        return ToolResult(
            text="\n\n".join(blocks),
            n_new=len(new_ids),
            new_ids=new_ids,
            docs=sorted(set(docs)),
            sources=sorted(sources),
            detail=detail,
        )

    def _grep_kb(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern") or "").strip()
        if len(_flat(pattern)) < 2:
            return ToolResult(
                "grep_kb 需要至少 2 个字符的 pattern", ok=False, detail="pattern 太短"
            )
        bounds = self._bounded(args, "grep_kb")
        max_hits = bounds.get("max_hits", 5)
        needle = _flat(pattern).lower()

        matched: list[Chunk] = []
        for chunk in self.retriever.chunks:
            if chunk.is_parent:
                continue
            if needle in _flat(chunk.text).lower():
                matched.append(chunk)
                if len(matched) >= max_hits:
                    break
        if not matched:
            return ToolResult(f"grep_kb({pattern!r}) 没有匹配", n_new=0, detail="零匹配")

        blocks: list[str] = []
        new_ids: list[int] = []
        docs: list[str] = []
        for chunk in matched:
            hit = Retrieved(chunk=chunk, score=0.0, source="grep")
            number, fresh = self._serve(hit)
            if number == NOT_CITABLE:
                continue
            if fresh:
                new_ids.append(number)
            docs.append(self.handles.code(chunk.doc_id))
            blocks.append(self._block(number, hit, fresh=fresh))
        return ToolResult(
            text="\n\n".join(blocks),
            n_new=len(new_ids),
            new_ids=new_ids,
            docs=sorted(set(docs)),
            sources=["grep"] if new_ids else [],
            detail=f"字面匹配 {len(matched)} 条",
        )

    def _read_section(self, args: dict[str, Any]) -> ToolResult:
        code = str(args.get("doc") or "").strip()
        doc_id = self.handles.doc_id(code)
        if doc_id is None:
            return ToolResult(
                f"未知文档代号 {code!r}；先用 list_docs 看有哪些代号",
                ok=False,
                detail=f"未知 doc {code}",
            )
        bounds = self._bounded(args, "read_section")
        limit = bounds.get("limit", 3)
        chunks = sorted(
            (c for c in self.retriever.chunks if c.doc_id == doc_id and not c.is_parent),
            key=lambda c: c.ordinal,
        )
        if not chunks:
            return ToolResult(f"文档 {code} 没有可读片段", ok=False, detail="空文档")

        start = 0
        near = str(args.get("near") or "").strip()
        if near:
            chunk_id = None
            match = _ID_NUMBER.search(near)
            if match:
                chunk_id = self.registry.resolve(int(match.group(1)))
            if chunk_id is None and near in self.evidence:
                chunk_id = near
            for position, chunk in enumerate(chunks):
                if chunk.chunk_id == chunk_id:
                    start = max(0, position - 1)
                    break

        picked = chunks[start : start + limit]
        blocks: list[str] = []
        new_ids: list[int] = []
        for chunk in picked:
            hit = Retrieved(chunk=chunk, score=0.0, source="read")
            number, fresh = self._serve(hit)
            if number == NOT_CITABLE:
                continue
            if fresh:
                new_ids.append(number)
            blocks.append(self._block(number, hit, fresh=fresh))
        return ToolResult(
            text="\n\n".join(blocks),
            n_new=len(new_ids),
            new_ids=new_ids,
            docs=[self.handles.code(doc_id)],
            sources=["read"] if new_ids else [],
            detail=f"读第 {start + 1}–{start + len(picked)} 块（共 {len(chunks)} 块）",
        )

    def _list_docs(self, args: dict[str, Any]) -> ToolResult:
        bounds = self._bounded(args, "list_docs")
        limit = bounds.get("limit", 10)
        keyword = str(args.get("keyword") or "").strip()
        needle = _flat(keyword).lower()

        first_chunk: dict[str, Chunk] = {}
        for chunk in self.retriever.chunks:
            if chunk.is_parent or chunk.doc_id in first_chunk:
                continue
            if needle and needle not in _flat(chunk.text).lower():
                # 关键词可能落在后面的块里：再扫一遍该文档的其它块
                continue
            first_chunk[chunk.doc_id] = chunk

        if needle:
            # 首块不含关键词的文档，只要任意一块含就算命中
            by_doc: dict[str, list[Chunk]] = {}
            for chunk in self.retriever.chunks:
                if not chunk.is_parent:
                    by_doc.setdefault(chunk.doc_id, []).append(chunk)
            for doc_id, chunks in by_doc.items():
                if doc_id in first_chunk:
                    continue
                if any(needle in _flat(c.text).lower() for c in chunks):
                    first_chunk[doc_id] = min(chunks, key=lambda c: c.ordinal)

        rows: list[str] = []
        for doc_id, chunk in first_chunk.items():
            if len(rows) >= limit:
                break
            doc = self.retriever.docs_by_id.get(doc_id)
            code = self.handles.code(doc_id)
            total = doc.n_chunks if doc is not None else 0
            preview = _clip(_WS.sub(" ", chunk.text.strip()), 80)
            rows.append(f"{code}｜{total} 块｜开头：{_neutralize_handles(preview)}")
        if not rows:
            return ToolResult(
                f"list_docs({'keyword=' + keyword if keyword else ''}) 没有匹配的文档",
                n_new=0,
                detail="零匹配",
            )
        head = f"共 {len(first_chunk)} 份文档匹配" + (f"（关键词 {keyword!r}）" if keyword else "")
        return ToolResult(
            text=head + "\n" + "\n".join(rows),
            docs=[row.split("｜")[0] for row in rows],
            detail=f"列出 {len(rows)} 份",
        )

    # ---------------------------------------------------------------- 取证
    def hits_in_registry_order(self) -> list[Retrieved]:
        """按编号顺序排好的命中列表。

        最终合成的上下文**必须**按这个顺序渲染：``[ID:n]`` 的 n 就是它的下标，
        这样最终答案的句柄既能回解到块，也能回指到步。
        """
        return list(self.evidence_order)

    def registry_texts(self) -> dict[str, str]:
        """chunk_id → 模型读到的正文（claim 逐字校验用）。"""
        return {hit.chunk.chunk_id: hit.context_chunk.text for hit in self.evidence_order}

    def evidence_map(self) -> list[dict[str, Any]]:
        """编号 → 哪一步、哪个工具、哪一块。

        这是验收"最终答案的每条 claim 都能回指到某一步发出的 ``[ID:n]``"的数据源。
        ``rel_path`` 是相对路径（项目里 eval 的 context 也存这一列），不含绝对路径。
        """
        rows: list[dict[str, Any]] = []
        for number, hit in enumerate(self.evidence_order):
            chunk = hit.context_chunk
            doc = self.retriever.docs_by_id.get(chunk.doc_id)
            step, source = self.served_at.get(hit.chunk.chunk_id, (0, hit.source or ""))
            rows.append(
                {
                    "id": number,
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_code": self.handles._code_of.get(chunk.doc_id, ""),
                    "rel_path": doc.rel_path if doc is not None else "",
                    "page": chunk.page,
                    "section": chunk.section,
                    "step": step,
                    "source": source,
                }
            )
        return rows
