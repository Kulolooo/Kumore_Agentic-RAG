"""claim 结构：``{陈述, 逐字证据, 句柄}`` 的提示词契约、解析与逐字校验。

抄的是 RAGFlow 的两处机制，不抄规模：

1. **claim 行的形状**（``_ref/ragflow-main/internal/agentic_rag/
   tool_navigate_structure.go:641-661``）::

       - [claim] <statement> | Evidence: "<verbatim quote>" [chunks: c1] ref=3

   即"一句原子陈述 + 它的逐字证据 + 一个可引用句柄"，一条行自带证据，
   所以可以当回答材料用，不必再回去读原文。我们把它汉化成::

       - 陈述：<一句话>｜证据：「<逐字原文>」｜句柄：[ID:0]

   ``ref=3`` 在 RAGFlow 里是工具渲染时盖的，模型只需回抄；我们的场景是模型
   在生成时就自己写句柄，所以句柄必须是**我们已经发出去的号**（见 evidence.py）。

2. **逐字证据的验收门**（``internal/ingestion/component/knowledge_compiler/
   tree/claims.go:83-86, 450-513``）：``evidence.quote`` 必须是被引 chunk 里
   **连续逐字**的一段；定位不到的引文在下游被丢弃，并计入 gate 日志
   （``verified`` / ``rejected``）。``locate_evidence`` 与 ``normalize_for_match``
   是 ``LocateEvidence`` / ``normalizeForMatch``（``:515-575``）的 Python 移植：
   匹配前把连续空白压成一个空格（容忍模型跨行复制时改写的换行），
   但偏移量仍然对**原始正文**计算，这样高亮能落回 char_start/char_end。

为什么不直接比字符串相等：PDF/PPTX 抽出来的正文常有软换行（"长 4.1\\n米"），
模型转述时会写成一行；空白归一化后同一句话才能对上。除此之外**不做**任何
同义、标点或语序的宽容——放宽就不再是"逐字"，而逐字率正是 M3 要先量出来的东西。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.generate.evidence import NOT_CITABLE, EvidenceRegistry

#: claim 行必须以列表符号开头。模型偶尔用 ``•`` 或 ``*``；不认表格行。
_BULLET = re.compile(r"^\s*[-*•·]\s+(?P<body>.+?)\s*$")

#: 三个字段。分隔符容忍全/半角竖线与 ``│``；字段缺失时该组不匹配。
_STATEMENT = re.compile(
    r"陈述\s*[:：]\s*(?P<v>.*?)(?=\s*[｜|│]\s*(?:证据|句柄)|$)", re.S
)
_EVIDENCE = re.compile(
    r"证据\s*[:：]\s*(?P<v>.*?)(?=\s*[｜|│]\s*(?:陈述|句柄)|$)", re.S
)
_HANDLE_FIELD = re.compile(r"句柄\s*[:：]\s*(?P<v>.*)$", re.S)

#: 句柄字面量：新契约 ``[ID:n]`` 优先，``[n]`` 只作过渡期兼容（语义 1 基）。
_HANDLE_ID = re.compile(r"\[ID:(\d{1,3})\]")
_HANDLE_LEGACY = re.compile(r"\[(\d{1,2})\]")

#: 引文外层的包装（提示词要求的「」/“”/""），定位前剥掉；内层一个字都不动。
_QUOTE_WRAP = re.compile(r"^[\s「『“\"']+|[\s」』”\"']+$")

_WS = re.compile(r"\s+")

#: 零宽字符与全角方括号：提示词把正文里的 ``[ID:n]`` 渲染成 ``［ID:n］``（防止照抄成
#: 引用），模型若原样抄进引文，定位时要能把两种写法认成同一个字符。
_ZW_TABLE = str.maketrans({"［": "[", "］": "]", "：": ":", "\u200b": None, "\u200c": None, "\u200d": None, "\ufeff": None})


# ---------------------------------------------------------------- 逐字定位
def normalize_for_match(text: str) -> tuple[str, list[int]]:
    """压平空白，并给出"归一化后第 j 个字符 → 原文第几个字符"的映射。

    对应 ``normalizeForMatch``（``claims.go:555-575``）。回改映射必须按**字符**
    记（Go 里按字节，因为 ``strings.Index`` 返回字节偏移）：否则中文引文会
    定位到错误的区间，甚至匹配不上。
    """
    out: list[str] = []
    index: list[int] = []
    prev_space = False
    for position, char in enumerate(text):
        mapped = char.translate(_ZW_TABLE)
        if not mapped:
            continue
        if mapped.isspace():
            if prev_space:
                continue
            out.append(" ")
            index.append(position)
            prev_space = True
        else:
            out.append(mapped)
            index.append(position)
            prev_space = False
    return "".join(out), index


def locate_evidence(quote: str, text: str) -> tuple[int, int] | None:
    """在被引正文里找逐字引文，返回原文上的 ``(start, end)``；找不到返回 None。

    ``end`` 是"匹配之后那个字符"的偏移（可直接切片），与源码一致
    （``claims.go:536-544``）：不能用"最后匹配字符 +1"——多字节字符会被切一半。
    """
    if not quote or not text:
        return None
    normalized_text, index = normalize_for_match(text)
    normalized_quote, _ = normalize_for_match(quote)
    if not normalized_quote:
        return None
    position = normalized_text.find(normalized_quote)
    if position < 0:
        return None
    last = position + len(normalized_quote) - 1
    if last >= len(index):
        return None
    end = index[last + 1] if last + 1 < len(index) else len(text)
    return index[position], end


# ---------------------------------------------------------------- 结构
@dataclass
class Claim:
    """一条 claim 行解析出来的结构。``located`` 在校验阶段才填。"""

    statement: str
    evidence: str = ""
    #: 句柄解析出来的全部 chunk_id（按句柄出现顺序，已去重、已剔除越界）。
    #: 一条陈述可能由多份资料共同支撑，提示词允许并列写 ``[ID:1][ID:3]``。
    chunk_ids: list[str] = field(default_factory=list)
    #: 写出来的全部编号（0 基）；越界的也在里面，便于统计
    numbers: list[int] = field(default_factory=list)
    #: 逐字证据在正文里的位置 ``(start, end)``；None = 定位失败
    located: tuple[int, int] | None = None
    #: 原始行，便于人工抽检与降级时回显
    raw: str = ""

    @property
    def number(self) -> int | None:
        """第一个句柄号（展示与旧字段兼容用）。"""
        return self.numbers[0] if self.numbers else None

    @property
    def chunk_id(self) -> str | None:
        return self.chunk_ids[0] if self.chunk_ids else None

    @property
    def resolved(self) -> bool:
        return bool(self.chunk_ids)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence.strip())

    @property
    def verbatim(self) -> bool:
        return self.located is not None


@dataclass
class ClaimReport:
    """一次回答的 claim 侧打点。字段名直接进评测 JSON，别随手改。"""

    lines: int = 0
    parsed: int = 0
    with_evidence: int = 0
    with_handle: int = 0
    resolved_handle: int = 0
    verbatim: int = 0
    unresolved_handle: list[int] = field(default_factory=list)
    unparsed_lines: int = 0

    @property
    def verbatim_rate(self) -> float | None:
        """逐字率：分母是**给了引文的 claim**。

        分母不用 ``parsed``：没给引文的 claim 谈不上"逐字"，把它算进分母会
        让"少给引文"看起来像"引文更准"。
        """
        return round(self.verbatim / self.with_evidence, 4) if self.with_evidence else None


def _strip_quote(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    # 只剥最外层一对包装；内部引号属于正文，不能动
    for opener, closer in (("「", "」"), ("『", "』"), ("“", "”"), ('"', '"'), ("'", "'")):
        if stripped.startswith(opener) and stripped.endswith(closer) and len(stripped) >= 2:
            return stripped[1:-1].strip()
    return stripped


def _parse_handles(value: str) -> list[int]:
    """从句柄字段里取出全部编号（0 基）。

    一行写多个句柄是提示词允许的写法（同一事实出现在多份资料里），所以这里
    不能只取第一个：只取第一个会让"引文在第二块"的 claim 被误判成不逐字。
    兼容旧写法 ``[n]``（1 基）。
    """
    numbers: list[int] = []
    for match in _HANDLE_ID.finditer(value or ""):
        numbers.append(int(match.group(1)))
    if numbers:
        return numbers
    for match in _HANDLE_LEGACY.finditer(value or ""):
        number = int(match.group(1)) - 1
        if number >= 0:
            numbers.append(number)
    return numbers


def parse_claims(text: str, registry: EvidenceRegistry | None = None) -> list[Claim]:
    """把模型输出解析成 claim 列表；解析不出来的行只计数，不改原文。

    容忍：列表符号（``-``/``*``/``•``）、全半角冒号、全半角竖线、
    引文外层的中英文引号。不容忍：字段改名、把三个字段压成散文。
    """
    claims: list[Claim] = []
    for line in (text or "").splitlines():
        bullet = _BULLET.match(line)
        if bullet is None:
            continue
        body = bullet.group("body")
        statement = _STATEMENT.search(body)
        if statement is None:
            continue
        evidence = _EVIDENCE.search(body)
        handle_field = _HANDLE_FIELD.search(body)
        numbers: list[int] = []
        if handle_field is not None:
            numbers = _parse_handles(handle_field.group("v"))
        claim = Claim(
            statement=statement.group("v").strip(),
            evidence=_strip_quote(evidence.group("v")) if evidence else "",
            numbers=numbers,
            raw=line.strip(),
        )
        if numbers and registry is not None:
            seen: list[str] = []
            for number in numbers:
                chunk_id = registry.resolve(number)
                if chunk_id is not None and chunk_id not in seen:
                    seen.append(chunk_id)
            claim.chunk_ids = seen
        claims.append(claim)
    return claims


def verify_claims(
    claims: list[Claim],
    text_by_chunk: dict[str, str],
    registry: EvidenceRegistry | None = None,
    raw_text: str = "",
) -> ClaimReport:
    """逐条定位引文，返回打点。

    ``text_by_chunk`` 是"chunk_id → 模型当时看到的正文"（父子块下是父块）。
    引文只在**被引的那一块**里找：源码在"模型没写 chunk_id 时"会退化到扫整批
    （``claims.go:478-490``），我们不做这个退化——扫整批会让"引错了块"看起来
    像"引对了"，而 M3 要量的正是"引文在不在它指的那一块里"。
    """
    report = ClaimReport(lines=len(claims), parsed=len(claims))
    for claim in claims:
        if claim.has_evidence:
            report.with_evidence += 1
        if claim.numbers:
            report.with_handle += 1
        if claim.chunk_ids:
            report.resolved_handle += 1
        if registry is not None:
            report.unresolved_handle.extend(
                number for number in claim.numbers if registry.resolve(number) is None
            )

        if not claim.has_evidence:
            continue
        # 多个句柄时逐个试：同一段引文可能来自其中任何一块，任意一块定位成功即算逐字。
        span = None
        for chunk_id in claim.chunk_ids:
            span = locate_evidence(claim.evidence, text_by_chunk.get(chunk_id, ""))
            if span is not None:
                break
        if span is None:
            continue
        claim.located = span
        report.verbatim += 1

    # 不是列表行、也不是空行的部分：只计数，供"解析失败率"用
    if raw_text:
        bullets = sum(1 for line in raw_text.splitlines() if _BULLET.match(line))
        report.unparsed_lines = max(len(raw_text.strip().splitlines()) - bullets, 0)
    return report


def render_claim_line(claim: Claim) -> str:
    """把 claim 渲染回契约行（导出/人工抽检用）。"""
    if claim.numbers:
        handle_text = "".join(f"[ID:{number}]" for number in claim.numbers)
    else:
        handle_text = "[ID:?]"
    evidence = f"证据：「{claim.evidence}」" if claim.evidence else "证据："
    return f"- 陈述：{claim.statement}｜{evidence}｜句柄：{handle_text}"


def flatten_sentences(text: str) -> str:
    """把可见空白压平，便于把 claim 陈述当句子做比对。"""
    return _WS.sub(" ", text or "").strip()
