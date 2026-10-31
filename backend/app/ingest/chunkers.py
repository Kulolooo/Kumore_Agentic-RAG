"""分块策略：四种，同一批解析结果只换切法。

M0 的固定窗口是**对照基线**，不是最终方案。M1 要回答的问题是
"同一批题、同一批文档，换一种切法，Recall 与证据完整性会怎么变"，
所以四种策略共用同一份 ``Block`` 输入与同一套元数据回填逻辑，只换切分规则：

============  ==============================================================
``fixed``     M0 现状：800 字符窗口 / 重叠 150，窗口末尾回退到句读。
``token``     抄 RAGFlow ``general`` 模板：512 token、8 个分隔符、**overlap 0**。
``section``   按标题层级切段，段内目标 1800 token（标题块不跨段）。
``parent_child``  small-to-big：父块 1800 token（``is_parent=True``，检索不返回），
              子块 512 token 带 ``mom_id``；命中子块后由上层回填父块正文。
============  ==============================================================

两个刻意的设计：

1. **token 是估算值**，用 ``app.text.tokens.estimate_tokens`` 的确定性字符成本，
   不是任何模型的分词器——理由写在那个模块里（可复现优先）。
2. **表格尽量整块保留**（``Block.kind == "table"``）：一张表被劈成两半后，
   后半段只有数据没有表头，检索到了也读不懂。超过预算 1.5 倍才允许拆。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.ingest.blocks import Block
from app.schemas import Chunk, make_chunk_id, make_content_hash
from app.text.tokens import _char_cost, estimate_tokens

STRATEGIES = ("fixed", "token", "section", "parent_child")

#: 8 个分隔符，顺序即优先级（抄 RAGFlow `general` 模板的 ``delimiters``）。
DEFAULT_DELIMITERS: tuple[str, ...] = ("\n\n", "\n", "。", "；", "！", "？", "!", "?")

#: 可以安全切断的位置（固定窗口用）。
_BOUNDARY_CHARS = "。！？!?\n；;"


@dataclass(frozen=True)
class ChunkStrategy:
    """一种分块配置。字段同时进 ``Doc.chunk_config`` 指纹。"""

    name: str = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 150
    token_size: int = 512
    section_tokens: int = 1800
    child_tokens: int = 512
    parent_tokens: int = 1800
    delimiters: tuple[str, ...] = DEFAULT_DELIMITERS

    def __post_init__(self) -> None:
        if self.name not in STRATEGIES:
            raise ValueError(f"未知分块策略 {self.name!r}，可选 {STRATEGIES}")
        if self.name == "fixed":
            if self.chunk_size <= 0:
                raise ValueError(f"chunk_size 必须为正数，当前 {self.chunk_size}")
            if not 0 <= self.chunk_overlap < self.chunk_size:
                raise ValueError(
                    f"chunk_overlap 必须在 [0, chunk_size) 内，当前 {self.chunk_overlap}"
                )
        for field_name in ("token_size", "section_tokens", "child_tokens", "parent_tokens"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} 必须为正数，当前 {value}")
        if self.name == "parent_child" and self.child_tokens >= self.parent_tokens:
            raise ValueError(
                f"子块预算必须小于父块：child={self.child_tokens}, parent={self.parent_tokens}"
            )

    @property
    def fingerprint(self) -> str:
        """写进 ``Doc.chunk_config``：改了参数就能一眼看出哪些文档的 chunk 是旧的。"""
        if self.name == "fixed":
            detail = f"size={self.chunk_size},overlap={self.chunk_overlap}"
        elif self.name == "token":
            detail = f"tokens={self.token_size},overlap=0"
        elif self.name == "section":
            detail = f"section_tokens={self.section_tokens}"
        else:
            detail = f"child={self.child_tokens},parent={self.parent_tokens}"
        return f"strategy={self.name},{detail}"


@dataclass
class _Unit:
    """一个 Block 在拼接文本里的区间。"""

    start: int
    end: int
    page: int | None
    section: str | None
    kind: str
    text: str


def _join_blocks(blocks: list[Block]) -> tuple[str, list[_Unit]]:
    parts: list[str] = []
    units: list[_Unit] = []
    pos = 0
    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        if pos:
            parts.append("\n")
            pos += 1
        start = pos
        parts.append(text)
        pos += len(text)
        units.append(_Unit(start, pos, block.page, block.section, block.kind, text))
    return "".join(parts), units


def _unit_at(units: list[_Unit], offset: int) -> _Unit | None:
    lo, hi = 0, len(units) - 1
    found: _Unit | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        unit = units[mid]
        if unit.end <= offset:
            lo = mid + 1
        else:
            found = unit
            hi = mid - 1
    return found or (units[-1] if units else None)


def _meta_at(units: list[_Unit], offset: int) -> tuple[int | None, str | None]:
    unit = _unit_at(units, offset)
    return (unit.page, unit.section) if unit else (None, None)


def _make_chunk(
    doc_id: str,
    ordinal: int,
    text: str,
    units: list[_Unit],
    start: int,
    end: int,
    *,
    is_parent: bool = False,
    mom_id: str | None = None,
) -> Chunk | None:
    raw = text[start:end]
    lead = len(raw) - len(raw.lstrip())
    piece = raw.strip()
    if not piece:
        return None
    real_start = start + lead
    real_end = real_start + len(piece)
    page, section = _meta_at(units, real_start)
    # 终点页按**窗口终点**算，而不是 strip 后的文本终点：窗口末尾的换行可能正好是
    # 页边界，M0 起就这么记（见 tests/test_chunk.py 的跨页用例），换了实现不能悄悄改。
    page_end, _ = _meta_at(units, max(real_start, end - 1))
    return Chunk(
        chunk_id=make_chunk_id(doc_id, ordinal),
        doc_id=doc_id,
        ordinal=ordinal,
        text=piece,
        page=page,
        page_end=page_end,
        section=section,
        char_start=real_start,
        char_end=real_end,
        content_hash=make_content_hash(piece),
        is_parent=is_parent,
        mom_id=mom_id,
    )


# ---------------------------------------------------------------- 切点
def _snap_end(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    """把窗口末尾往前挪到最近的句子边界（固定窗口策略）。"""
    if hard_end >= len(text):
        return len(text)
    floor = start + max(int(chunk_size * 0.6), 1)
    for i in range(hard_end, floor, -1):
        if text[i - 1] in _BOUNDARY_CHARS:
            return i
    return hard_end


def _split_spans(
    text: str, max_tokens: int, delimiters: tuple[str, ...]
) -> list[tuple[int, int]]:
    """按 token 预算 + 分隔符优先级切分，返回相对区间。O(n)。"""
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    while start < n:
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break

        cost = 0.0
        last_hit = {d: -1 for d in delimiters}
        i = start
        while i < n:
            cost += _char_cost(text[i])
            if cost > max_tokens:
                break
            for delimiter in delimiters:
                if text.startswith(delimiter, i):
                    last_hit[delimiter] = i + len(delimiter)
                    break
            i += 1

        if i >= n:
            cut = n
        else:
            floor = start + max(int((i - start) * 0.5), 1)
            cut = None
            for delimiter in delimiters:
                position = last_hit[delimiter]
                if position >= floor:
                    cut = position
                    break
            if cut is None:
                cut = i
        spans.append((start, cut))
        start = cut
    return spans


def _pack_units(
    units: list[_Unit],
    max_tokens: int,
    delimiters: tuple[str, ...],
) -> list[tuple[int, int]]:
    """把 Block 贪心装进 token 预算；单块超预算才拆，表格允许 1.5 倍溢出。"""
    spans: list[tuple[int, int]] = []
    buffer: list[_Unit] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            spans.append((buffer[0].start, buffer[-1].end))
        buffer = []
        buffer_tokens = 0

    for unit in units:
        tokens = estimate_tokens(unit.text)
        if buffer and buffer_tokens + tokens > max_tokens:
            flush()
        if tokens > max_tokens:
            if unit.kind == "table" and tokens <= int(max_tokens * 1.5):
                spans.append((unit.start, unit.end))
                continue
            flush()
            for offset_start, offset_end in _split_spans(unit.text, max_tokens, delimiters):
                spans.append((unit.start + offset_start, unit.start + offset_end))
            continue
        buffer.append(unit)
        buffer_tokens += tokens
    flush()
    return spans


def _section_groups(units: list[_Unit]) -> list[list[_Unit]]:
    """按 section 连续分段。section 为空也算一段（不和别的段合并）。"""
    groups: list[list[_Unit]] = []
    for unit in units:
        if groups and groups[-1][0].section == unit.section:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return groups


# ---------------------------------------------------------------- 策略
def _fixed_chunks(
    doc_id: str, text: str, units: list[_Unit], config: ChunkStrategy
) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    total = len(text)
    while start < total:
        hard_end = min(start + config.chunk_size, total)
        end = _snap_end(text, start, hard_end, config.chunk_size)
        if end <= start:
            end = hard_end
        chunk = _make_chunk(doc_id, ordinal, text, units, start, end)
        if chunk is not None:
            chunks.append(chunk)
            ordinal += 1
        if end >= total:
            break
        start = max(end - config.chunk_overlap, start + 1)
    return chunks


def _token_chunks(
    doc_id: str, text: str, units: list[_Unit], config: ChunkStrategy
) -> list[Chunk]:
    # RAGFlow general 模板是一个 chunk 装若干完整 Block，而不是在字符流上滑窗；
    # 这里照做：先按 Block 装，单块超预算才按分隔符拆，overlap = 0。
    spans = _pack_units(units, config.token_size, config.delimiters)
    return _chunks_from_spans(doc_id, text, units, spans)


def _chunks_from_spans(
    doc_id: str, text: str, units: list[_Unit], spans: list[tuple[int, int]]
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for start, end in spans:
        chunk = _make_chunk(doc_id, ordinal, text, units, start, end)
        if chunk is not None:
            chunks.append(chunk)
            ordinal += 1
    return chunks


def _section_chunks(
    doc_id: str, text: str, units: list[_Unit], config: ChunkStrategy
) -> list[Chunk]:
    spans: list[tuple[int, int]] = []
    for group in _section_groups(units):
        spans.extend(_pack_units(group, config.section_tokens, config.delimiters))
    return _chunks_from_spans(doc_id, text, units, spans)


def _parent_child_chunks(
    doc_id: str, text: str, units: list[_Unit], config: ChunkStrategy
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    parent_spans: list[tuple[int, int]] = []
    for group in _section_groups(units):
        parent_spans.extend(_pack_units(group, config.parent_tokens, config.delimiters))

    for parent_start, parent_end in parent_spans:
        parent = _make_chunk(
            doc_id, ordinal, text, units, parent_start, parent_end, is_parent=True
        )
        if parent is None:
            continue
        chunks.append(parent)
        ordinal += 1
        parent_text = text[parent.char_start : parent.char_end]
        # 子块在父块正文内再切；偏移要加回父块起点，才能对上文档全文。
        for offset_start, offset_end in _split_spans(
            parent_text, config.child_tokens, config.delimiters
        ):
            child = _make_chunk(
                doc_id,
                ordinal,
                text,
                units,
                parent.char_start + offset_start,
                parent.char_start + offset_end,
                mom_id=parent.chunk_id,
            )
            if child is not None:
                chunks.append(child)
                ordinal += 1
    return chunks


def chunk_document(
    doc_id: str, blocks: list[Block], config: ChunkStrategy
) -> list[Chunk]:
    """按给定策略切分。空输入返回空列表。"""
    text, units = _join_blocks(blocks)
    if not text.strip() or not units:
        return []

    if config.name == "fixed":
        return _fixed_chunks(doc_id, text, units, config)
    if config.name == "token":
        return _token_chunks(doc_id, text, units, config)
    if config.name == "section":
        return _section_chunks(doc_id, text, units, config)
    if config.name == "parent_child":
        return _parent_child_chunks(doc_id, text, units, config)
    raise ValueError(f"未知分块策略 {config.name!r}")


def config_from_settings(settings) -> ChunkStrategy:
    """从 ``Settings`` 生成策略配置（供 pipeline / CLI 共用）。"""
    return ChunkStrategy(
        name=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        token_size=settings.chunk_token_size,
        section_tokens=settings.section_tokens,
        child_tokens=settings.child_tokens,
        parent_tokens=settings.parent_tokens,
    )


def with_name(config: ChunkStrategy, name: str) -> ChunkStrategy:
    """换策略但保留其余参数（对照实验用）。"""
    return replace(config, name=name)
