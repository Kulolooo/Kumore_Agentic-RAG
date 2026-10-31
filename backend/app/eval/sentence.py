"""句级校验：切句 → 编号映射 → 被引 chunk 内逐字定位 → 数字一致性。

M3 的验收要一个"每一句是不是真的被它引的那一块支撑"的判据。这里**不用大模型
打分**（路线图 M3 第 4 条：启发式优先），只做四件可复算的事：

1. **切句**：按中英文句末标点切（英文句号要认，但避开小数点与常见缩写），
   编号跟着前一句走。
2. **编号映射**：由调用方把句内编号解析成 chunk（``[ID:n]`` 与过渡期 ``[n]``
   都认），本模块只负责"这句话有没有编号、有几个"。
3. **逐字定位**：句子的正文与被引 chunk 正文做连续字符串比对，用
   ``SequenceMatcher`` 的匹配块算**并集覆盖率**与最长连续命中。
   并集的意思是：一句话的事实分布在两条被引块里也算数（多跳/拼接），
   每块各贡献一段，取并。
4. **数字一致性**：句子里出现的数字必须能在被引 chunk 里找到。中文数字
   （"八万五千"）与单位族由 ``app/eval/numbers.py`` 处理；"第 7 页""资料[3]"
   这类**引用型数字**先摘掉，它们不是事实。

两条判据（严格 / 数字锚定）见模块常量。它们是**判据**不是事实：报告里必须连同
人工抽检的一致率一起看；单独一个数字不能证明校验器可信。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.eval.numbers import ZERO_WIDTH, missing_tokens, numeric_tokens, strip_references, token_label

#: 句末标点。分号也算：中文技术文本里分号常用来切并列的独立事实。
_SENTENCE_END = "。！？!?；;"
_SPLIT = re.compile(rf"(?<=[{re.escape(_SENTENCE_END)}])|\n+")

#: 英文句末：标点 + 空白 + 大写/数字/引号开头。小数点不会命中（点后不跟空白），
#: 缩写（Dr./Fig.）由 _EN_ABBREV 排除。
_EN_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“”‘(（])")
_EN_ABBREV = (
    "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "corp", "vs", "etc",
    "e.g", "i.e", "no", "fig", "eq", "ref", "sec", "approx", "st", "jr", "sr",
)

#: 两种编号写法。``[ID:n]`` 是 M3 契约；``[n]`` 是过渡期兼容，也用于旧报告。
_MARKER_ID = re.compile(r"\[ID:(\d{1,3})\]")
_MARKER_LEGACY = re.compile(r"\[(\d{1,2})\]")

#: 编号写在句末标点之后时，会作为下一段的开头出现；这些要并回上一句。
_LEADING_MARKERS = re.compile(r"^(?:(?:\[ID:\d{1,3}\])|(?:\[\d{1,2}\]))+")

_WS = re.compile(r"\s+")
_ZW_TABLE = str.maketrans("", "", ZERO_WIDTH)

#: 逐字判据（严格路径）：句子至少一半的字符能在被引块里对上，且有一段 >=8 字的连续命中。
VERBATIM_MIN_COVERAGE = 0.5
VERBATIM_MIN_LONGEST = 8
#: 数字锚定路径：句子里有数字、数字全对、且有 >=6 字的连续命中时，0.4 的覆盖率即可。
#: 为什么需要它：表格里的"产品长度 | 2160mm"被竖线/单元格切断，最长连续只有 6 字，
#: 严格路径会把它判成不支持（人工看是有据的）。
ANCHORED_MIN_COVERAGE = 0.4
ANCHORED_MIN_LONGEST = 6

#: 句级支撑的判据 = 召回 ∧ 精确：
#:   * 召回：至少一条被引块里能找到该题的证据串；
#:   * 精确：**过半**的被引块里都能找到。
#:
#: 为什么必须带精确这一半：M3 起点实测，模型给几乎每句都挂 3–5 个编号，只要金标块
#: 在上下文里，按"任一被引块含证据"就能拿到 ~85% 的支撑率——而那个数字几乎等于
#: "该题金标块是否进了上下文"，说明它测的是检索，不是引用。
#: 过半（而不是"全部"）是给"一题多版本、并列两个说法"留的余地。
SUPPORT_MIN_PRECISION = 0.5


def is_supported(grounded: int, cited_total: int) -> bool:
    """句级支撑判据（召回 ∧ 精确）。两个计数都由调用方按金标证据串算。"""
    if cited_total <= 0:
        return False
    return grounded >= 1 and grounded >= SUPPORT_MIN_PRECISION * cited_total


@dataclass
class SentenceCheck:
    """一句的判定结果。字段名直接进评测 JSON。"""

    text: str
    #: 去掉编号后的正文
    body: str
    #: 本句写出的编号（原样数字，按出现顺序，含重复）
    markers: list[int]
    #: 与 ``markers`` 一一对应的写法：``id``（``[ID:n]``）或 ``legacy``（``[n]``）
    marker_forms: list[str] = field(default_factory=list)
    #: ``[ID:n]`` 形式的编号个数 / 过渡期 ``[n]`` 形式的个数
    marker_id: int = 0
    marker_legacy: int = 0
    coverage: float = 0.0
    longest: int = 0
    numbers_total: int = 0
    #: 对不上的数字，字面如 ``"85000(money)"``
    numbers_missing: list[str] = field(default_factory=list)
    #: 因为属于"第 X 页/资料[n]"而被摘掉、未参与比对的数字个数
    numbers_ignored: int = 0

    @property
    def cited(self) -> bool:
        return bool(self.markers)

    @property
    def numbers_ok(self) -> bool:
        return not self.numbers_missing

    @property
    def verbatim(self) -> bool:
        """逐字自证：数字全对，且正文够多地出现在被引块里。

        两条路径：严格的连续命中原判据；以及"有数字锚定"时的放宽版（见常量注释）。
        """
        if not self.cited or not self.numbers_ok:
            return False
        if self.longest >= VERBATIM_MIN_LONGEST and self.coverage >= VERBATIM_MIN_COVERAGE:
            return True
        return (
            self.numbers_total > 0
            and self.longest >= ANCHORED_MIN_LONGEST
            and self.coverage >= ANCHORED_MIN_COVERAGE
        )

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "body": self.body,
            "markers": list(self.markers),
            "marker_forms": list(self.marker_forms),
            "marker_id": self.marker_id,
            "marker_legacy": self.marker_legacy,
            "coverage": round(self.coverage, 3),
            "longest": self.longest,
            "numbers_total": self.numbers_total,
            "numbers_missing": list(self.numbers_missing),
            "numbers_ignored": self.numbers_ignored,
            "verbatim": self.verbatim,
        }


def _split_english(piece: str) -> list[str]:
    """在一段里再按英文句末切；缩写不切。"""
    out: list[str] = []
    start = 0
    for match in _EN_SPLIT.finditer(piece):
        head = piece[start : match.start()]
        # 前一个词是不是常见缩写（"Dr." 后面不该断）
        word = re.search(r"([A-Za-z.]+)\.$", head.strip())
        if word and word.group(1).lower() in _EN_ABBREV:
            continue
        if head.strip():
            out.append(head.strip())
            start = match.start()
    tail = piece[start:].strip()
    if tail:
        out.append(tail)
    return out or [piece]


def split_sentences(text: str) -> list[str]:
    """切成句子；编号跟着它修饰的那一句走，空行与空白碎片丢弃。"""
    pieces = [p.strip() for p in _SPLIT.split(text or "") if p and p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        for part in _split_english(piece):
            if _is_marker_only(part):
                if sentences:
                    sentences[-1] = f"{sentences[-1]}{part}"
                continue
            if not _has_content(part):
                continue
            # 模型被要求把编号放在标点之前；万一放在之后，它就成了下一段的开头，
            # 这里搬回上一句，否则"这句有没有引用"会被判错。
            leading = _LEADING_MARKERS.match(part)
            if leading is not None and sentences:
                sentences[-1] = f"{sentences[-1]}{leading.group(0)}"
                part = part[leading.end() :].strip()
                if not part or not _has_content(part):
                    continue
            sentences.append(part)
    return sentences


def _is_marker_only(piece: str) -> bool:
    return bool(
        (_MARKER_ID.fullmatch(piece) or _MARKER_LEGACY.fullmatch(piece))
    )


def _has_content(piece: str) -> bool:
    return any(ch.isalnum() for ch in piece)


def markers_in(text: str) -> list[tuple[int, str]]:
    """``[(原样编号, 写法)]``，按出现位置排序。

    两种写法可能同时出现，规则是**同句内 ID 优先**：

    * 句子里有 ``[ID:n]`` → 只认 ``[ID:n]``，裸 ``[n]`` 一律当正文（模型没写 `ID:`，
      或文档原文里本来就有 ``[9,10]`` 这种参考文献编号——实测 claim 答案里混进过
      15 个，全部来自被复制的正文，把它们当引用会凭空造出"引错了"的句子）；
    * 句子里一个 ``[ID:n]`` 都没有 → 按过渡期的 ``[n]``（1 基）解析，用于复算
      M3 之前的旧报告。

    这条规则让"同一份报告里两种写法并存"不会互相污染，也让旧报告仍然可复算。
    """
    found: list[tuple[int, int, str]] = []
    for match in _MARKER_ID.finditer(text or ""):
        found.append((match.start(), int(match.group(1)), "id"))
    if found:
        found.sort()
        return [(number, form) for _, number, form in found]
    for match in _MARKER_LEGACY.finditer(text or ""):
        found.append((match.start(), int(match.group(1)), "legacy"))
    found.sort()
    return [(number, form) for _, number, form in found]


def strip_markers(text: str) -> str:
    return _MARKER_LEGACY.sub("", _MARKER_ID.sub("", text or "")).strip()


def _flat(text: str) -> str:
    """压平空白并丢掉零宽字符（提示词用它隔开正文里的 ``[ID:n]`` 变体）。"""
    return _WS.sub("", (text or "").translate(_ZW_TABLE))


def match_profile(body: str, cited_texts: list[str]) -> tuple[float, int]:
    """``(并集覆盖率, 最长连续命中字符数)``。

    覆盖率 = 句子正文字符中，能在**任一**被引块里对上的比例（各块取并）。
    分母是**句子**：一句 20 字的话里有 16 字在原文里，就算 0.8，
    不因为 chunk 里还有 700 字别的内容而缩水。
    并集让"一句话的事实分布在两条被引块里"也能被承认（多跳/拼接）。
    """
    flat_body = _flat(body)
    if not flat_body:
        return 0.0, 0
    matched = bytearray(len(flat_body))
    longest = 0
    for cited in cited_texts:
        flat_cited = _flat(cited)
        if not flat_cited:
            continue
        matcher = SequenceMatcher(None, flat_body, flat_cited, autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.size <= 1:
                continue
            matched[block.a : block.a + block.size] = b"\x01" * block.size
            longest = max(longest, block.size)
    return sum(matched) / len(flat_body), longest


def check_sentence(sentence: str, cited_texts: list[str]) -> SentenceCheck:
    """判一句。``cited_texts`` 是该句编号解析出来的被引 chunk 正文。"""
    pairs = markers_in(sentence)
    id_count = sum(1 for _, form in pairs if form == "id")
    body = strip_markers(sentence)
    coverage, longest = match_profile(body, cited_texts)
    # 句子侧摘掉引用型数字（"第 7 页""资料[3]"），块侧保留全部
    wanted = numeric_tokens(body)
    available = {token for text in cited_texts for token in numeric_tokens(text, keep_references=True)}
    missing = [token_label(token) for token in missing_tokens(wanted, available)]
    return SentenceCheck(
        text=sentence,
        body=body,
        markers=[number for number, _ in pairs],
        marker_forms=[form for _, form in pairs],
        marker_id=id_count,
        marker_legacy=len(pairs) - id_count,
        coverage=coverage,
        longest=longest,
        numbers_total=len(wanted),
        numbers_missing=missing,
        numbers_ignored=max(
            0,
            len(numeric_tokens(body, keep_references=True))
            - len(numeric_tokens(strip_references(body))),
        ),
    )
