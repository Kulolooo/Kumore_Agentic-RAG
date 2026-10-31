"""文档解析：把各种格式统一成 ``list[Block]``。

## 后端

``native``（默认）：PDF 用 pdfminer.six，DOCX 用 python-docx，PPTX 用 python-pptx，
XLSX 用 openpyxl，MD/TXT 直接读。全部纯 CPU、离线可跑、无模型依赖。
``unstructured``：装好 spaCy 模型或能访问 GitHub 时可切换，方便对比两种解析的差异。

## M1 只做三件解析改进（不换解析器）

1. **去页眉页脚 / 页码**：全文档统计重复行，见 ``app/ingest/layout.py``；
2. **表格转 Markdown**：保留行列与多行表头，见 ``app/ingest/tables.py``；
3. **双栏判定**：按全文档多数投票，而不是逐页独立判断，见 ``layout.py``。

PDF 的表格结构**仍然不抽**：pdfminer 只给文字块，没有单元格边界；硬做要引入版面
模型或坐标启发式，而当前 26 份 PDF 里表格占比很低。这是已知限制，报告里如实写，
不拿"应该能行"充数。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.ingest.blocks import Block
from app.ingest.ocr import (
    OCRUnavailable,
    ocr_image,
    ocr_mode,
    ocr_pdf_page,
    should_ocr_pdf_page,
)
from app.ingest.layout import (
    PdfLine,
    detect_two_column,
    drop_page_furniture,
    order_lines,
)
from app.ingest.tables import from_docx_table, from_pptx_table, render_table

log = logging.getLogger(__name__)

#: 解析器版本号。M1 改了页眉页脚/表格/双栏，M0-2 审计又修了"只差数字的小标题被当页眉删"；
#: 写进 ``Doc.chunk_config`` 指纹，才能一眼看出"这份文档是旧解析器切的"。
PARSER_VERSION = "native-v4"

#: 直接按文本读的格式，不必经过任何第三方解析器。
PLAIN_TEXT_EXTS = frozenset({".md", ".txt"})

#: native 后端支持的格式。
NATIVE_EXTS = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})

#: 页眉页脚对检索几乎只有噪声作用（"XX大学"、"第 3 页"）。
SKIP_CATEGORIES = frozenset({"Header", "Footer"})

#: 纯文本读取时依次尝试的编码。中文 Windows 导出的 txt 常见 GBK。
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "utf-16", "latin-1")

#: Excel 单表最多读多少行。表格型文档常有几万行，全收会把索引撑爆且没有检索价值。
_XLSX_MAX_ROWS = 5000

#: Excel 单表最多读多少列。普通模式会把带格式的空单元格算进 max_column（可到 16384），
#: 全量迭代会从 90 秒入库变成几分钟（实测卡死过一次）。
_XLSX_MAX_COLS = 200

#: 零宽字符。PPT 导出的 PDF 里常见整行只有 \u200b，留着会变成空 chunk。
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def _clean_text(text: str) -> str:
    return _ZERO_WIDTH.sub("", text or "").strip()


@dataclass
class ParseResult:
    blocks: list[Block] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    n_dropped_headers: int = 0
    backend: str = "native"
    #: M1 解析质量计数：转成 Markdown 的表格数、丢掉的页眉页脚行数、是否判为双栏。
    n_tables: int = 0
    n_dropped_furniture: int = 0
    two_column: bool = False
    n_pages: int = 0
    #: 本次解析实际走 OCR 的页数（扫描页/图片页），用于入库报告与新鲜度排查。
    n_ocr_pages: int = 0
    #: 实际 OCR 的页码（1-based）；pipeline 据此渲染并保存页面图。
    ocr_pages: list[int] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


# ------------------------------------------------------------------ 纯文本
def _read_plain_text(path: Path) -> tuple[str, str]:
    """返回 (文本, 实际使用的编码)。"""
    raw = path.read_bytes()
    for enc in _TEXT_ENCODINGS:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8/replace"


def _split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if line.strip():
            buffer.append(line.rstrip())
        elif buffer:
            paragraphs.append("\n".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append("\n".join(buffer))
    return paragraphs


def _markdown_blocks(text: str) -> list[Block]:
    """按 ``#`` 标题切段，并把标题记成 section，便于引用时说明"出自哪一节"。"""
    blocks: list[Block] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(Block(text="\n".join(buffer), section=section))
            buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            hashes = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[hashes:].strip()
            if title:
                flush()
                section = title
                continue
            # 只有 # 没有标题（某些 README 的装饰线），按普通文本处理
        if not stripped:
            flush()
            continue
        buffer.append(line.rstrip())
    flush()
    return blocks


def _plain_text_blocks(path: Path) -> ParseResult:
    result = ParseResult()
    text, enc = _read_plain_text(path)
    if enc not in ("utf-8-sig", "utf-8"):
        result.warnings.append(f"按 {enc} 解码")
    blocks = _markdown_blocks(text) if path.suffix.lower() == ".md" else [
        Block(text=para) for para in _split_paragraphs(text)
    ]
    result.blocks = [b for b in blocks if b.text.strip()]
    if not result.blocks:
        # 不能只说"空文件"：Markdown 的 `# 标题` 会被收进 section 而不是 block，
        # 只有标题的页面（书里的章首页）到这里同样没有正文。误导性的报错会让
        # "2 份失败"看起来像文件损坏。
        result.warnings.append("没有可索引的正文（空文件或只有标题）")
    return result


# ------------------------------------------------------------------ PDF
def _read_pdf_pages(path: Path) -> list[list[PdfLine]]:
    """按页读取文字块与坐标。只收 LTTextContainer，图元不进文本流。"""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages: list[list[PdfLine]] = []
    for layout in extract_pages(str(path)):
        lines: list[PdfLine] = []
        for element in layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text()
                if text and text.strip():
                    lines.append(PdfLine(text=text, bbox=tuple(element.bbox)))
        pages.append(lines)
    return pages


def _pdf_blocks(path: Path, result: ParseResult) -> list[Block]:
    pages = _read_pdf_pages(path)
    result.n_pages = len(pages)

    # 先按文字层判断哪些页需要 OCR；文字层正常的页保持 pdfminer 的坐标/双栏逻辑，
    # 不为了 OCR 把整份 PDF 重走一遍图像模型。
    mode = ocr_mode()
    ocr_text: dict[int, str] = {}
    if mode != "off":
        for page_no, page in enumerate(pages, start=1):
            chars = sum(len(line.text) for line in page)
            if not should_ocr_pdf_page(chars, mode=mode):
                continue
            try:
                text = ocr_pdf_page(path, page_no - 1)
            except OCRUnavailable as exc:
                result.warnings.append(f"第 {page_no} 页 OCR 不可用：{exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - 单页 OCR 失败不应中断整份文档
                result.warnings.append(f"第 {page_no} 页 OCR 失败：{type(exc).__name__}: {exc}")
                continue
            if text.strip():
                ocr_text[page_no] = text
                result.n_ocr_pages += 1
                result.ocr_pages.append(page_no)

    pages, dropped = drop_page_furniture(pages)
    result.n_dropped_furniture = dropped
    two_column, two_pages = detect_two_column(pages)
    result.two_column = two_column

    blocks: list[Block] = []
    for page_no, page in enumerate(pages, start=1):
        if page_no in ocr_text:
            for para in _split_paragraphs(ocr_text[page_no]):
                text = para.strip()
                if text:
                    blocks.append(Block(text=text, page=page_no))
            continue
        for line in order_lines(page, two_column):
            for para in _split_paragraphs(line.text):
                text = para.strip()
                if text:
                    blocks.append(Block(text=text, page=page_no))
    if two_column:
        result.warnings.append(f"双栏版式（{two_pages}/{result.n_pages} 页检测到分栏）")
    if result.n_ocr_pages:
        result.warnings.append(f"OCR 补读 {result.n_ocr_pages}/{result.n_pages} 页")
    return blocks


def _image_blocks(path: Path, result: ParseResult) -> list[Block]:
    """图片/扫描页：直接走 OCR，把识别文字当正文。"""
    if ocr_mode() == "off":
        result.warnings.append("图片解析需要 OCR，但 KUMORE_OCR_MODE=off")
        return []
    text = ocr_image(path)
    if not text.strip():
        result.warnings.append("OCR 未识别出文字")
        return []
    result.n_pages = 1
    result.n_ocr_pages = 1
    result.ocr_pages = [1]
    return [Block(text=part.strip(), page=1) for part in _split_paragraphs(text) if part.strip()]


# ------------------------------------------------------------------ Office
def _is_heading(style_name: str | None) -> bool:
    style = (style_name or "").strip().lower()
    return style.startswith("heading") or style.startswith("标题") or style in {"title"}


def _iter_docx_content(doc):
    """按文档顺序产出段落与表格。python-docx>=1.1 才有 iter_inner_content。"""
    if hasattr(doc, "iter_inner_content"):
        yield from doc.iter_inner_content()
        return
    yield from doc.paragraphs  # 老版本退化为只取段落


def _docx_blocks(path: Path, result: ParseResult) -> list[Block]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks: list[Block] = []
    section: str | None = None

    for item in _iter_docx_content(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style_name = item.style.name if item.style is not None else None
            if _is_heading(style_name):
                section = text
            blocks.append(Block(text=text, section=section))
        elif isinstance(item, Table):
            table = from_docx_table(item)
            markdown = render_table(table.rows, table.caption)
            if markdown:
                blocks.append(Block(text=markdown, section=section, kind="table"))
                result.n_tables += 1
    return blocks


def _pptx_blocks(path: Path, result: ParseResult) -> list[Block]:
    from pptx import Presentation
    from pptx.enum.shapes import PP_PLACEHOLDER

    presentation = Presentation(str(path))
    blocks: list[Block] = []
    # 幻灯片页脚 / 页码 / 日期版式与 PDF 页眉页脚等价，直接跳过。
    furniture = {
        PP_PLACEHOLDER.FOOTER,
        PP_PLACEHOLDER.SLIDE_NUMBER,
        PP_PLACEHOLDER.DATE,
    }
    for slide_no, slide in enumerate(presentation.slides, start=1):
        title: str | None = None
        try:
            if slide.shapes.title is not None:
                title = (slide.shapes.title.text or "").strip() or None
        except (AttributeError, ValueError):
            title = None

        parts: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "is_placeholder", False):
                try:
                    if shape.placeholder_format.type in furniture:
                        continue
                except (AttributeError, ValueError):
                    pass
            if getattr(shape, "has_text_frame", False):
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        parts.append(text)
            if getattr(shape, "has_table", False):
                table = from_pptx_table(shape.table)
                markdown = render_table(table.rows, table.caption)
                if markdown:
                    parts.append(markdown)
                    result.n_tables += 1
        if parts:
            blocks.append(Block(text="\n".join(parts), page=slide_no, section=title))
    return blocks


def _xlsx_blocks(path: Path, result: ParseResult) -> list[Block]:
    import openpyxl

    # 不用 read_only：合并单元格信息只在普通模式下有，而合并区域决定了
    # 哪一格该留值、哪一格该留空。10 份 xlsx 的规模不需要只读优化。
    workbook = openpyxl.load_workbook(str(path), data_only=True)
    blocks: list[Block] = []
    try:
        for sheet in workbook.worksheets:
            max_col = min(sheet.max_column or 1, _XLSX_MAX_COLS)
            covered: set[tuple[int, int]] = set()
            for merged in sheet.merged_cells.ranges:
                for row in range(merged.min_row, min(merged.max_row, _XLSX_MAX_ROWS) + 1):
                    for col in range(merged.min_col, min(merged.max_col, max_col) + 1):
                        if (row, col) != (merged.min_row, merged.min_col):
                            covered.add((row, col))

            rows: list[list[str]] = []
            for cells in sheet.iter_rows(
                min_row=1, max_row=_XLSX_MAX_ROWS, max_col=max_col
            ):
                values = []
                for cell in cells:
                    if (cell.row, cell.column) in covered:
                        values.append("")
                        continue
                    value = cell.value
                    values.append("" if value is None else str(value).strip())
                if any(values):
                    rows.append(values)
            if sheet.max_row and sheet.max_row > _XLSX_MAX_ROWS:
                rows.append([f"…（超过 {_XLSX_MAX_ROWS} 行，已截断）"])
            rendered = render_table(rows, caption=sheet.title)
            if rendered:
                blocks.append(Block(text=rendered, section=sheet.title, kind="table"))
                result.n_tables += 1
    finally:
        workbook.close()
    return blocks


_NATIVE_PARSERS = {
    ".pdf": _pdf_blocks,
    ".docx": _docx_blocks,
    ".pptx": _pptx_blocks,
    ".xlsx": _xlsx_blocks,
    **{ext: _image_blocks for ext in IMAGE_EXTS},
}


def _native_parse(path: Path) -> ParseResult:
    result = ParseResult(backend="native")
    ext = path.suffix.lower()
    if ext in PLAIN_TEXT_EXTS:
        return _plain_text_blocks(path)
    if ext not in _NATIVE_PARSERS:
        return ParseResult(
            ok=False,
            backend="native",
            error=f"native 后端不支持 {ext}（旧版 .doc 需 LibreOffice 或 antiword）",
        )
    result.blocks = _NATIVE_PARSERS[ext](path, result)
    if not result.blocks:
        if ext == ".pdf":
            reason = "没有文字层，且 OCR 未补读到内容（检查 OCR 依赖/扫描清晰度）"
        elif ext in IMAGE_EXTS:
            reason = "图片 OCR 未识别出文字"
        else:
            reason = "未抽取到文本"
        result.warnings.append(reason)
    return result


# ------------------------------------------------------------------ unstructured 后端
def _page_of(el) -> int | None:
    meta = getattr(el, "metadata", None)
    if meta is None:
        return None
    page = getattr(meta, "page_number", None)
    if page is None:
        page = getattr(meta, "slide_number", None)
    return int(page) if isinstance(page, int) else None


def _unstructured_parse(path: Path, strategy: str) -> ParseResult:
    from unstructured.partition.auto import partition

    result = ParseResult(backend="unstructured")
    elements = partition(filename=str(path), strategy=strategy)

    section: str | None = None
    for el in elements:
        category = getattr(el, "category", None) or ""
        text = (getattr(el, "text", None) or "").strip()
        if not text:
            continue
        if category in SKIP_CATEGORIES:
            result.n_dropped_headers += 1
            continue
        if category == "Title":
            section = text
        result.blocks.append(Block(text=text, page=_page_of(el), section=section))

    if not result.blocks:
        result.warnings.append("未抽取到任何文本（可能是扫描件或纯图片文档）")
    if result.n_dropped_headers:
        result.warnings.append(f"丢弃页眉页脚 {result.n_dropped_headers} 段")
    return result


# ------------------------------------------------------------------ 入口
def parse_document(
    path: Path,
    *,
    backend: str = "native",
    strategy: str = "fast",
    max_bytes: int = 0,
) -> ParseResult:
    """解析单个文件，任何异常都变成 ``ok=False``，不中断整批入库。"""
    path = Path(path)
    if not path.is_file():
        return ParseResult(ok=False, error="文件不存在", backend=backend)

    if max_bytes and path.stat().st_size > max_bytes:
        mb = path.stat().st_size / 1024 / 1024
        return ParseResult(
            ok=False,
            backend=backend,
            error=f"文件过大（{mb:.0f} MB > {max_bytes / 1024 / 1024:.0f} MB 上限），已跳过",
        )

    try:
        if backend == "unstructured":
            if path.suffix.lower() in PLAIN_TEXT_EXTS:
                return _plain_text_blocks(path)
            result = _unstructured_parse(path, strategy)
        else:
            result = _native_parse(path)
        # 统一清洗：零宽字符、首尾空白；清完为空的块直接丢。
        cleaned: list[Block] = []
        for block in result.blocks:
            text = _clean_text(block.text)
            if text:
                block.text = text
                cleaned.append(block)
        result.blocks = cleaned
        return result
    except Exception as exc:  # noqa: BLE001 - 解析器内部异常类型不可控，统一降级
        log.warning("解析失败 %s: %s", path.name, exc)
        return ParseResult(
            ok=False, backend=backend, error=f"{type(exc).__name__}: {exc}"
        )
