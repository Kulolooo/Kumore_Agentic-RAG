"""M5 分块侧指标：边界合理性 + chunk 数膨胀率。

**先定口径，再看数**。路线图 M5 只写了两个名字，这里把"分子/分母"写死；
以后换分块策略、换语料，都按这一份定义复算，不允许在报告里临时改口径。

一、边界合理性（boundary reasonableness）
==========================================

**问的是**：一个检索单元（chunk）是不是在"人读起来自然"的位置结束？
它衡量分块器有没有把一句话/一个数字切断。**不**衡量 chunk 内容是否有用。

判定按"最后一个非空白字符"（以及紧挨着的收尾引号/括号）分类：

============  ==========================================================  ==========
类别          判据                                                        算不算合理
============  ==========================================================  ==========
sentence      以句末标点 ``。！？!?；;`` 结束；允许后面跟收尾符号          算
              ``”’」』）)]】》``；ASCII 句点 ``.`` 也算（英文文档）       
paragraph     原始文本以换行结束（行尾空白不算）                          算
clause        以逗号/顿号/冒号 ``，,、：:`` 结束                            不算（可读，但切在分句中间）
mid_sentence  其余（以汉字/字母/数字/空白等非标点结尾）                    不算
empty         空 chunk（理论上不该出现）                                  不计入分母
============  ==========================================================  ==========

- **分子**：``sentence`` + ``paragraph`` 的检索单元数。
- **分母**：**全部非父块检索单元数**（``Chunk.is_parent == False``）。
  父块不参与检索，混进来会把长父块的"天然段落尾"算成优势，四策略不可比。
- 同时报 ``clause_end_rate`` / ``mid_sentence_rate``，让读者看到"不算合理"的
  里面有多少只是切在逗号（较轻）而不是切在词中间（较重）。
- ``cut_mid_token_rate`` 是更窄的代理：最后一个字符是 ASCII 字母/数字
  （如 ``CH340`` / ``28BYJ`` 被切断）的单元占比。中文没有词边界，这个代理
  只对英文/型号名成立——报告里必须把它当"下界"，不能当完整口径。

二、chunk 数膨胀率（chunk count inflation）
==========================================

**问的是**：同一批文档、同一次解析下，某个策略切出的检索单元数是 fixed 的几倍。

- **分子**：该策略的检索单元数（非父块）。
- **分母**：**同一语料、同一次解析的 fixed 策略检索单元数**。
- **口径**：``rate = strategy_units / fixed_units``；``fixed`` 自己恒为 1.0。
  分子分母都不含父块（理由同上）；库内总行数（含父块）另列，不参与比值。
- 为什么不用"字符总数 / 平均长度"推算：那样会把 overlap 重复计入，
  且不同策略的丢弃规则不同；直接数检索单元最不容易自欺。

这两个指标都**不是**"越高越好"：边界合理率高但 chunk 数膨胀 3 倍，检索成本
和上下文噪声都会涨；报告里必须两个一起看（M1 四策略矩阵就是这么排的）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: 句末标点。分号算句末：中文分号两侧通常是独立分句，切在这里可读。
SENTENCE_END = frozenset("。！？!?；;")
#: ASCII 句点：英文文档的句末。单独列出，避免和中文标点混在一起。
ASCII_PERIOD = "."
#: 分句标点：切在这里不算"合理边界"，但比切在词中间轻。
CLAUSE_END = frozenset("，,、：:")
#: 收尾符号：允许出现在句末标点之后（引号、括号、书名号等）。
CLOSERS = frozenset("”’」』）)]】》〉")

BOUNDARY_CLASSES = ("sentence", "paragraph", "clause", "mid_sentence", "empty")


def classify_boundary(text: str | None) -> str:
    """返回一个 chunk 的边界类别（见模块头）。

    只吃掉尾部空白与收尾符号；不吃正文。空串归 ``empty``。
    """
    raw = text or ""
    if not raw.strip():
        return "empty"
    # 段落尾：原始文本以换行结束（行尾空格/Tab 不算内容）。
    if raw.rstrip(" \t").endswith("\n"):
        return "paragraph"
    body = raw.rstrip()
    # 先剥掉收尾符号（可能有多层，如 」））。
    while body and body[-1] in CLOSERS:
        body = body[:-1].rstrip()
    if not body:
        return "mid_sentence"
    last = body[-1]
    if last in SENTENCE_END:
        return "sentence"
    if last == ASCII_PERIOD:
        # 英文句点：排除 "1.2" / "v1." 这类不太可能是句末的形态。
        # 前一个字符是数字时仍可能是编号结尾（"第 3."），当成合理边界；
        # 这里不做更复杂的猜测——代理指标要可机械复算。
        return "sentence"
    if last in CLAUSE_END:
        return "clause"
    return "mid_sentence"


def _is_ascii_alnum(char: str) -> bool:
    return bool(char) and char.isascii() and char.isalnum()


def _chunks_of(chunks: Iterable[Any]) -> list[Any]:
    """只保留检索单元（非父块）。兼容 Chunk / dict。"""
    out = []
    for chunk in chunks:
        is_parent = (
            chunk.get("is_parent", False)
            if isinstance(chunk, dict)
            else getattr(chunk, "is_parent", False)
        )
        if not is_parent:
            out.append(chunk)
    return out


def _text_of(chunk: Any) -> str:
    return str(chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "") or "")


def boundary_stats(chunks: Iterable[Any]) -> dict[str, Any]:
    """按模块头口径算边界分布。返回计数与比率（分母显式写出）。"""
    units = _chunks_of(chunks)
    counts = {name: 0 for name in BOUNDARY_CLASSES}
    mid_token = 0
    for chunk in units:
        text = _text_of(chunk)
        kind = classify_boundary(text)
        counts[kind] += 1
        body = (text or "").rstrip()
        while body and body[-1] in CLOSERS:
            body = body[:-1].rstrip()
        if body and _is_ascii_alnum(body[-1]):
            mid_token += 1

    total = len(units)
    denominator = total - counts["empty"]
    reasonable = counts["sentence"] + counts["paragraph"]

    def rate(value: int, denom: int) -> float | None:
        return round(value / denom, 4) if denom else None

    return {
        "denominator": denominator,
        "n_units": total,
        "counts": counts,
        "boundary_ok": reasonable,
        "boundary_ok_rate": rate(reasonable, denominator),
        "sentence_end_rate": rate(counts["sentence"], denominator),
        "paragraph_end_rate": rate(counts["paragraph"], denominator),
        "clause_end_rate": rate(counts["clause"], denominator),
        "mid_sentence_rate": rate(counts["mid_sentence"], denominator),
        "cut_mid_token": mid_token,
        "cut_mid_token_rate": rate(mid_token, denominator),
    }


@dataclass
class Inflation:
    """chunk 数膨胀率：strategy / fixed（同一语料、同一解析）。"""

    strategy_units: int
    baseline_units: int
    baseline_name: str = "fixed"
    #: 库内总行数（含父块），只作展示，不参与比值
    strategy_total: int | None = None
    baseline_total: int | None = None

    @property
    def rate(self) -> float | None:
        if not self.baseline_units:
            return None
        return round(self.strategy_units / self.baseline_units, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_units": self.strategy_units,
            "baseline_units": self.baseline_units,
            "baseline_name": self.baseline_name,
            "rate": self.rate,
            "strategy_total": self.strategy_total,
            "baseline_total": self.baseline_total,
        }


def inflation_rate(
    strategy_units: int,
    baseline_units: int,
    *,
    strategy_total: int | None = None,
    baseline_total: int | None = None,
    baseline_name: str = "fixed",
) -> Inflation:
    """构造膨胀率。分母为 0 时 ``rate`` 返回 None（报告写"—"而不是除零）。"""
    return Inflation(
        strategy_units=int(strategy_units),
        baseline_units=int(baseline_units),
        baseline_name=baseline_name,
        strategy_total=strategy_total,
        baseline_total=baseline_total,
    )


def chunk_side_report(chunks: Iterable[Any], *, baseline_units: int | None = None) -> dict[str, Any]:
    """一次性算边界 + 膨胀率（给 M1 矩阵/报告头用）。

    ``baseline_units`` 传 fixed 策略的检索单元数；不传则只报边界。
    """
    units = _chunks_of(chunks)
    stats = boundary_stats(units)
    if baseline_units is not None:
        stats["inflation"] = inflation_rate(len(units), baseline_units).to_dict()
    return stats


def render_boundary_table(rows: list[dict[str, Any]]) -> str:
    """把多组 ``boundary_stats`` 渲染成一张表（每行必须带分母）。"""
    lines = [
        "| 策略 | 检索单元（分母） | 合理边界 | 句末 | 段落尾 | 分句尾 | 句中 | 切断型号/英文 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        name = row.get("strategy", "—")
        denom = row.get("denominator", 0)
        counts = row.get("counts", {})
        lines.append(
            f"| {name} | {denom} | {row.get('boundary_ok_rate')} "
            f"| {row.get('sentence_end_rate')} | {row.get('paragraph_end_rate')} "
            f"| {row.get('clause_end_rate')} | {row.get('mid_sentence_rate')} "
            f"| {row.get('cut_mid_token_rate')}（{counts.get('mid_sentence', 0)} 块中 "
            f"{row.get('cut_mid_token', 0)} 块） |"
        )
    return "\n".join(lines)
