"""PDF 版式后处理：页眉页脚、页码、双栏。

三件事都是**全文档统计**，不是逐页拍脑袋——这是 RAGFlow 逆向报告给的教训：
逐页独立判断会把"某页正好没有页眉"当成正常内容留下，而按全文档多数投票才稳。

1. **页眉/页脚**：同一行文字在 ≥ 一半页面出现（数字归一化后比较），或形如
   "第 3 页 / 3 / 20 / - 5 -"，且位于页面首尾，就整篇丢弃。
   只对 ≥3 页的文档生效：2 页文档里"两页都有"说明不了任何事。
2. **双栏**：逐页找一条贯穿页面的空白走廊；多数页面存在走廊才认定整篇双栏
   （RAGFlow `02-摄取链路.md` §2.2 的做法）。认定后所有页面都按"左栏读完再读右栏"
   排序；单页没检测到走廊的，用页宽中线兜底。
3. 单栏文档按从上到下排序（pdfminer 的内容流顺序对单栏通常没问题，但扫描/导出
   工具常把相邻两行顺序写反，按坐标排更稳）。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_PAGE_NO = re.compile(
    r"^[\s\-—–_·]*(?:第\s*)?\d{1,4}\s*(?:页)?\s*(?:[/／]\s*\d{1,4}\s*页?)?[\s\-—–_·]*$"
)
#: 常见英文页码："Page 3 of 10" / "3 / 10" / "- 3 -"。
_PAGE_NO_EN = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$", re.I)
_DIGITS = re.compile(r"\d+")


@dataclass
class PdfLine:
    """PDF 上的一段文本及其包围盒 (x0, y0, x1, y1)。"""

    text: str
    bbox: tuple[float, float, float, float]

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


def _norm(text: str) -> str:
    return _DIGITS.sub("#", " ".join(text.split()))


def _repeat_key(text: str) -> str:
    """重复行比较用的键。

    只有**页码类**文本才把数字归一化。正文里的 "Section 1.1" / "第 3 章" 也仅仅
    差数字，归一化后会被当成"每页重复的页眉"删掉——实测合成语料里整行小标题被删。
    页眉页脚的文字本身通常逐字相同，用原样比较就够；页码则由 ``_looks_like_page_number``
    单独识别。
    """
    stripped = " ".join(text.split())
    if _looks_like_page_number(stripped):
        return _norm(stripped)
    return stripped


def _looks_like_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 24:
        return False
    if _PAGE_NO_EN.match(stripped):
        # "3" / "Page 3 of 10"：前者可能只是正文里的一个数字，交给"多页重复"那条规则；
        # 带 page/of 的才单独判定。
        return bool(re.search(r"page|of", stripped, re.I))
    if not _PAGE_NO.match(stripped):
        return False
    # 只有带"页"或斜杠的中文写法才单独判定；裸数字/"- 5 -"交给"多页重复"那条规则，
    # 避免把封面上的年份、独立数字标题误删。
    return "页" in stripped or "/" in stripped or "／" in stripped


def drop_page_furniture(
    pages: list[list[PdfLine]], *, min_pages: int = 3, ratio: float = 0.5
) -> tuple[list[list[PdfLine]], int]:
    """丢掉页眉页脚与页码。返回 (新页面, 丢弃行数)。

    返回的列表与输入**一一对应**（空页保留），否则页码会整体错位。
    只检查每页**前两个 / 后两个**元素：页眉页脚不会出现在正文中间。
    """
    content_pages = [page for page in pages if page]
    n_pages = len(content_pages)
    if n_pages == 0:
        return pages, 0
    if n_pages < min_pages:
        return pages, 0

    first_counter: Counter[str] = Counter()
    last_counter: Counter[str] = Counter()
    for page in content_pages:
        for line in page[:2]:
            first_counter[_repeat_key(line.text)] += 1
        for line in page[-2:]:
            last_counter[_repeat_key(line.text)] += 1

    threshold = max(min_pages, int(n_pages * ratio + 0.999))
    # 只把**短行**当页眉页脚候选：长正文里也可能只差几个数字（"第 3 章""2024 年"），
    # 数字归一化会把它误判成重复行。页眉页脚几乎都短于 40 字。
    repeated = {
        text
        for counter in (first_counter, last_counter)
        for text, count in counter.items()
        if count >= threshold and text and len(text) <= 40
    }

    kept: list[list[PdfLine]] = []
    dropped = 0
    for page in pages:
        if not page:
            kept.append(page)
            continue
        head_cut = 0
        for line in page[:2]:
            if _repeat_key(line.text) in repeated or _looks_like_page_number(line.text):
                head_cut += 1
            else:
                break
        tail_cut = 0
        for line in reversed(page[-2:]):
            if _repeat_key(line.text) in repeated or _looks_like_page_number(line.text):
                tail_cut += 1
            else:
                break
        end = max(head_cut, len(page) - tail_cut)
        body = page[head_cut:end]
        dropped += len(page) - len(body)
        kept.append(body)
    return kept, dropped
    return kept, dropped


def _page_split(page: list[PdfLine]) -> float | None:
    """找一条纵向空白走廊；找到返回分割 x，否则 None。"""
    boxes = [line for line in page if line.text.strip()]
    if len(boxes) < 3:
        return None
    left_edge = min(box.x0 for box in boxes)
    right_edge = max(box.x1 for box in boxes)
    width = right_edge - left_edge
    if width <= 1:
        return None
    # 只认竖版页面。双栏排版的载体（论文/报告/书）是竖版；幻灯片是横版，
    # 里面的文本框左右分布会被误判成双栏——实测 29 页的 PPT 导出 PDF 整篇被排乱。
    height = max(box.y1 for box in boxes) - min(box.y0 for box in boxes)
    if height <= width:
        return None
    total_chars = sum(len(box.text) for box in boxes) or 1

    for ratio in [0.35 + 0.02 * i for i in range(16)]:  # 0.35 ~ 0.65
        split = left_edge + width * ratio
        left = [box for box in boxes if box.x1 <= split]
        right = [box for box in boxes if box.x0 >= split]
        if len(left) < 2 or len(right) < 2:
            continue
        covered = len(left) + len(right)
        if covered < len(boxes) * 0.8:
            continue
        gap = min(box.x0 for box in right) - max(box.x1 for box in left)
        if gap < width * 0.02:
            continue
        left_chars = sum(len(box.text) for box in left)
        right_chars = sum(len(box.text) for box in right)
        if min(left_chars, right_chars) < total_chars * 0.3:
            continue
        return split
    return None


def detect_two_column(pages: list[list[PdfLine]]) -> tuple[bool, int]:
    """全文档多数投票。返回 (是否双栏, 检测到走廊的页数)。"""
    pages = [page for page in pages if page]
    if len(pages) < 2:
        return False, 0
    two = sum(1 for page in pages if _page_split(page) is not None)
    return two >= max(2, int(len(pages) * 0.6 + 0.999)), two


def order_lines(page: list[PdfLine], two_column: bool) -> list[PdfLine]:
    """按阅读顺序返回一页的行。"""
    if not page:
        return []
    if not two_column:
        return sorted(page, key=lambda line: (-line.y1, line.x0))

    split = _page_split(page)
    if split is None:
        boxes = [line for line in page if line.text.strip()]
        if boxes:
            split = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    if split is None:
        return sorted(page, key=lambda line: (-line.y1, line.x0))

    left = [line for line in page if (line.x0 + line.x1) / 2 < split]
    right = [line for line in page if (line.x0 + line.x1) / 2 >= split]
    left.sort(key=lambda line: -line.y1)
    right.sort(key=lambda line: -line.y1)
    return left + right
