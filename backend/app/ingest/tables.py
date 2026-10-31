"""表格 → Markdown：保留行列结构，不吞任何一行表头。

M0 的表格是压平成 ``A | B`` 的纯文本：行列关系没了，表头和数据行长得一样，
检索到了也读不懂哪一列是什么。M1 改成 Markdown 表，并刻意遵守两条：

1. **表头不止取 1 行**。多级表头（合并单元格）在 docx 里往往写成 2–3 行。
   RAGFlow 的固定 1 行表头会把第二行表头当数据"吞"进第一行数据里
   （`docs/逆向/02-摄取链路.md` §10 坑 2）。这里的做法是：
   第一行当 Markdown 表头，**其余行一行不丢**地当数据行保留。
2. **合并单元格只保留左上角的值**，其余位置留空。python-docx 对横向/纵向合并的
   格子会返回同一个底层 ``<w:tc>``，靠对象标识去重，比"文本相同就删"安全——
   两个不同格子恰好写着同样的字（例如两行都是"1班"）不会被误删。

表格正文里出现的换行、竖线都做转义，否则会破坏 Markdown 表结构。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")


@dataclass
class TableData:
    rows: list[list[str]]
    caption: str | None = None


def _clean(text: str) -> str:
    """单元格内不留换行：Markdown 表格的一行必须是一行。"""
    return _WS.sub(" ", (text or "").replace("\u00a0", " ")).strip()


def _trim_rows(rows: list[list[str]]) -> list[list[str]]:
    """清掉空行，但**保留行尾空格子**。

    行尾空格子不是脏数据：整表标题行常写成 ``["附件3", "", "", ...]``，
    把空列裁掉后它就变成"单列表格"，标题会被当成表头（实测踩过）。
    """
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_clean(cell) for cell in row]
        if any(cells):
            cleaned.append(cells)
    return cleaned


def _extract_caption(rows: list[list[str]]) -> tuple[list[list[str]], str | None]:
    """首行只有一个非空值、且表有多列 → 当成整表标题。

    合并单元格让标题行变成 ``["表1 参数", "", ""]``；把它当表头会让真正的表头
    降级成数据行。这里把它提成 caption，正文从下一行开始。
    """
    if len(rows) < 2:
        return rows, None
    first = rows[0]
    values = [cell for cell in first if cell]
    if len(values) != 1:
        return rows, None
    if len(first) < 2:
        return rows, None
    return rows[1:], values[0]


def render_table(rows: list[list[str]], caption: str | None = None) -> str:
    """把二维单元格渲染成 Markdown。空表返回空串。"""
    rows = _trim_rows(rows)
    rows, auto_caption = _extract_caption(rows)
    caption = auto_caption or _clean(caption or "") or None
    if not rows:
        return f"表：{caption}" if caption else ""

    # 宽度按"最后一个非空格子"算：xlsx 的合并区域会把行撑到几十列空壳，
    # 不裁的话每张表都要渲染一排空的 | --- |。
    width = max(
        (max((i + 1 for i, value in enumerate(row) if value), default=0) for row in rows),
        default=0,
    )
    if width <= 1:
        lines = [f"表：{caption}"] if caption else []
        lines.extend(row[0] for row in rows if row[0])
        return "\n".join(lines)
    rows = [row[:width] + [""] * (width - len(row)) for row in rows]

    def cell(value: str) -> str:
        return value.replace("|", "\\|")

    lines: list[str] = []
    if caption:
        lines.append(f"表：{caption}")
    lines.append("| " + " | ".join(cell(value) for value in rows[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- docx
def from_docx_table(table) -> TableData:
    """按底层 XML 单元格对象去重合并区域（横向 + 纵向都有效）。

    注意不能只存 ``id(cell._tc)``：临时对象被回收后 id 会被复用，实测把不同格子的
    正文误删（"上限"单元格被当成合并区域的延续）。这里直接把元素对象放进 set，
    既做身份比较，也保证它不被回收。
    """
    rows: list[list[str]] = []
    seen_tc: set = set()
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            try:
                marker = cell._tc
            except AttributeError:  # 非 python-docx 的兼容实现
                marker = cell
            if marker in seen_tc:
                cells.append("")
                continue
            seen_tc.add(marker)
            cells.append(cell.text)
        rows.append(cells)
    return TableData(rows=rows)


# ---------------------------------------------------------------- pptx
def from_pptx_table(table) -> TableData:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            if getattr(cell, "is_spanned", False):
                cells.append("")
            else:
                cells.append(cell.text)
        rows.append(cells)
    return TableData(rows=rows)


# ---------------------------------------------------------------- xlsx
def from_xlsx_rows(rows: list[list[str]]) -> TableData:
    return TableData(rows=rows)
