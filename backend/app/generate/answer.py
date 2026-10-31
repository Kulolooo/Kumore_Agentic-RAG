"""问答：检索 → 提示 → 生成 → 校验引用。

关键设计：**没有任何检索结果时根本不调用模型**。既省钱，也避免模型在空上下文下
自由发挥；这种情况下拒答是确定性的，评测里可以稳定复现。

M3 起引用编号是 ``[ID:n]``（0 基、按展示顺序发号，见 ``evidence.py``）。
为什么换：旧的 ``[n]`` 与正文里的方括号数字**无法区分**——学术 PDF 里的
``假象消除[9,10]`` 会被当成引用（实测 claim 答案里混进 15 个）。``[ID:n]``
把引用变成一个不会被正文误触发的机械契约。旧写法仍被识别，只为复算历史报告；
生成侧只写 ``[ID:n]``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generate.client import DeepSeekClient
from app.generate.claim import parse_claims, verify_claims
from app.generate.evidence import EvidenceRegistry, handle as render_handle
from app.generate.prompt import REFUSAL_PREFIX, build_messages
from app.retrieve.retriever import Retriever
from app.retrieve.versions import VersionInfo
from app.schemas import Answer, Citation, Doc, Retrieved

#: 机械契约：``[ID:n]``，n 从 0 开始（与 EvidenceRegistry 的发号一致）。
_HANDLE_ID = re.compile(r"\[ID:(\d{1,3})\]")

#: 过渡期写法：``[n]``（1 基）。只用于复算 M3 之前的报告与兼容模型偶发回退。
_HANDLE_LEGACY = re.compile(r"\[(\d{1,2})\]")

#: 畸形句柄：写了 ``ID:`` 但后面不是 1–3 位数字（如 ``[ID:1234]``、``[ID:一]``）。
#: 它显然是想当引用，但不能算有效引用，也不能留在正文里冒充引用——一律删除。
_MALFORMED_ID = re.compile(r"\[ID:[^\]\n]{0,12}\]")

#: 删掉越界编号后，可能留下"文字 [9] 。"这种断掉的样子，收一下尾
_DANGLING_SPACE = re.compile(r"[ \t]+(?=[，。；、！？,.])")


@dataclass(frozen=True)
class Handle:
    """正文里出现的一个引用编号。"""

    form: str  # "id" | "legacy"
    number: int  # 字面写出来的数字（id 是 0 基，legacy 是 1 基）
    literal: str  # "[ID:3]" / "[3]"
    start: int
    end: int

    def index(self, n_context: int) -> int | None:
        """映射到上下文下标；越界返回 None。"""
        position = self.number if self.form == "id" else self.number - 1
        return position if 0 <= position < n_context else None

    def label(self) -> str:
        return render_handle(self.number) if self.form == "id" else f"[{self.number}]"


def extract_handles(text: str, *, include_legacy: bool = True) -> list[Handle]:
    """按出现顺序取出全部编号，两种写法都认，并记下写法与位置。

    ``include_legacy=False`` 时只认 ``[ID:n]``。调用方在"整份回答里已经出现
    ``[ID:n]``"时会用它：那时裸 ``[n]`` 一律当正文——claim 的「证据」字段是逐字
    复制来的，里面可能带参考文献编号（``[5]``），当引用会让指标凭空多出一条。
    """
    found: list[Handle] = []
    for match in _HANDLE_ID.finditer(text or ""):
        found.append(
            Handle("id", int(match.group(1)), match.group(0), match.start(), match.end())
        )
    if include_legacy:
        for match in _HANDLE_LEGACY.finditer(text or ""):
            found.append(
                Handle(
                    "legacy", int(match.group(1)), match.group(0), match.start(), match.end()
                )
            )
    found.sort(key=lambda item: item.start)
    return found


def sanitize_answer(
    text: str, n_context: int, *, allow_legacy: bool = True
) -> tuple[str, list[int]]:
    """把越界编号从回答正文里删除。

    为什么要在生成后删：文档内容本身可能包含「必须引用 [9]」这类注入指令
    （实测 7 次里 1 次会让模型照做）。校验器只能**发现**越界，用户看到的仍然是
    一段带假引用的回答。所以把越界标记删掉，保证交给用户的引用一定可回溯。

    返回 (清洗后的文本, 被删掉的编号)。编号按**字面数字**记录，方便与模型输出对照；
    畸形标记（``[ID:1234]`` 之类）没有数字可记，删除后只在 warnings 里说明。
    """
    text = text or ""
    #: (起点, 终点, 要记录的数字 / None=畸形)
    spans: list[tuple[int, int, int | None]] = []
    for match in _MALFORMED_ID.finditer(text):
        inner = _HANDLE_ID.fullmatch(match.group(0))
        if inner is None:
            spans.append((match.start(), match.end(), None))
            continue
        number = int(inner.group(1))
        if 0 <= number < n_context:
            continue
        spans.append((match.start(), match.end(), number))
    for match in _HANDLE_LEGACY.finditer(text):
        if not allow_legacy:
            break
        number = int(match.group(1))
        if 1 <= number <= n_context:
            continue
        spans.append((match.start(), match.end(), number))
    if not spans:
        return text.strip(), []

    spans.sort()
    pieces: list[str] = []
    removed: list[int] = []
    cursor = 0
    for start, end, number in spans:
        if start < cursor:  # 两种写法理论上不重叠；真重叠就跳过，别切坏正文
            continue
        pieces.append(text[cursor:start])
        cursor = end
        if number is not None:
            removed.append(number)
    pieces.append(text[cursor:])
    return _DANGLING_SPACE.sub("", "".join(pieces)).strip(), removed


def extract_markers(text: str) -> list[int]:
    """正文里的编号数字，按出现顺序（含重复）。

    语义是"模型写出来的那个数字"：``[ID:3]`` 给 3，``[3]`` 也给 3。
    两种写法的下标换算不同，要下标请看 ``extract_handles``。
    """
    return [item.number for item in extract_handles(text)]


def is_refusal(text: str) -> bool:
    return text.strip().startswith(REFUSAL_PREFIX)


def _build_citations(
    handles: list[Handle],
    hits: list[Retrieved],
    docs_by_id: dict[str, Doc],
) -> tuple[list[Citation], list[int]]:
    """把编号映射回具体 chunk；越界编号单独收集，说明模型在编造。"""
    citations: list[Citation] = []
    invalid: list[int] = []
    seen: set[tuple[str, int]] = set()
    for item in handles:
        key = (item.form, item.number)
        if key in seen:
            continue
        seen.add(key)
        index = item.index(len(hits))
        if index is None:
            invalid.append(item.number)
            continue
        hit = hits[index]
        doc = docs_by_id.get(hit.chunk.doc_id)
        # 父子块：模型看到的是父块正文，页码/小节/预览都按父块写；
        # chunk_id 仍指向命中的子块（它才是可被检索、可算指标的那一行）。
        shown = hit.context_chunk
        citations.append(
            Citation(
                marker=item.number,
                chunk_id=hit.chunk.chunk_id,
                doc_id=hit.chunk.doc_id,
                rel_path=doc.rel_path if doc else "",
                page=shown.page,
                page_end=shown.page_end,
                section=shown.section,
                text=shown.preview,
                handle=item.label(),
            )
        )
    return citations, invalid


def answer_from_hits(
    question: str,
    hits: list[Retrieved],
    docs_by_id: dict[str, Doc],
    client: DeepSeekClient,
    versions: dict[str, VersionInfo] | None = None,
    include_paths: bool = False,
    prompt_variant: str = "strict",
) -> Answer:
    """给定已检索好的上下文直接生成。

    单独拆出来是为了让评测复用同一份检索结果：评测既算检索指标又算回答指标，
    如果各调一次检索，两边看到的上下文可能不一致（以及白跑一遍 BM25）。
    """
    context_ids = [h.chunk.chunk_id for h in hits]

    if not hits:
        return Answer(
            question=question,
            answer=f"{REFUSAL_PREFIX}：知识库中没有检索到与该问题相关的资料。",
            refused=True,
            context_chunk_ids=[],
            warnings=["零命中，未调用模型"],
        )

    messages = build_messages(
        question, hits, docs_by_id, versions, include_paths, variant=prompt_variant
    )
    chat = client.chat(messages)

    # 越界编号先记下来（这是"模型违反引用规则"的原始指标），再从正文里删掉，
    # 保证交付出去的引用一定可回溯。
    # 引用形态由"整份回答里有没有出现 [ID:n]"决定：有 → 只认 ID 形态，
    # 裸 [n] 是正文（claim 的证据字段会逐字复制带参考文献编号的原文）。
    allow_legacy = _HANDLE_ID.search(chat.text) is None
    raw_handles = extract_handles(chat.text, include_legacy=allow_legacy)
    raw_invalid = sorted(
        {item.number for item in raw_handles if item.index(len(hits)) is None}
    )

    cleaned, removed = sanitize_answer(chat.text, len(hits), allow_legacy=allow_legacy)
    citations, _ = _build_citations(
        extract_handles(cleaned, include_legacy=allow_legacy), hits, docs_by_id
    )
    refused = is_refusal(cleaned)

    warnings: list[str] = []
    malformed = [
        match.group(0)
        for match in _MALFORMED_ID.finditer(chat.text)
        if _HANDLE_ID.fullmatch(match.group(0)) is None
    ]
    if malformed:
        warnings.append(
            f"删除了 {len(malformed)} 个畸形引用标记（如 {malformed[0]}），它们不是有效句柄"
        )
    if raw_invalid:
        warnings.append(
            f"引用了不存在的资料编号 {raw_invalid}（上下文只有 {len(hits)} 条），已从正文删除"
        )
    if not refused and not citations:
        warnings.append("回答未标注任何引用")

    # ---- M3：claim 结构 ----
    # 只在 claim 变体下解析。解析失败**不重试、不改写正文**，直接按普通回答交付
    # 并打点——重试会把一次失败变成两次费用，而用户拿到的还是同一段文字。
    claims: list = []
    claim_report = None
    claim_fallback = False
    if prompt_variant == "claim" and not refused:
        registry = EvidenceRegistry.from_hits(hits)
        claims = parse_claims(cleaned, registry)
        if not claims:
            claim_fallback = True
            warnings.append(
                "claim 解析失败（没有一行符合 {陈述, 证据, 句柄} 格式），已按普通回答交付"
            )
        else:
            claim_report = verify_claims(
                claims,
                {h.chunk.chunk_id: h.context_chunk.text for h in hits},
                registry,
                raw_text=cleaned,
            )

    return Answer(
        question=question,
        answer=cleaned,
        refused=refused,
        citations=citations,
        cited_markers=[c.marker for c in citations],
        invalid_markers=raw_invalid,
        sanitized_markers=removed,
        context_chunk_ids=context_ids,
        warnings=warnings,
        claims=claims,
        claim_report=claim_report,
        claim_fallback=claim_fallback,
        prompt_tokens=chat.prompt_tokens,
        completion_tokens=chat.completion_tokens,
        latency_ms=chat.latency_ms,
        model=chat.model,
    )


def answer_question(
    question: str,
    retriever: Retriever,
    client: DeepSeekClient,
    *,
    top_k: int = 5,
    cutoff: float = 0.0,
    include_superseded: bool = False,
    include_paths: bool = False,
    prompt_variant: str = "strict",
    doc_filter=None,
) -> Answer:
    result = retriever.search(
        question,
        top_k=top_k,
        include_superseded=include_superseded,
        relative_cutoff=cutoff,
        doc_filter=doc_filter,
    )
    return answer_from_hits(
        question,
        result.answer_hits,
        retriever.docs_by_id,
        client,
        versions=retriever.versions,
        include_paths=include_paths,
        prompt_variant=prompt_variant,
    )
