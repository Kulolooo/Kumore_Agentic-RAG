"""本地 OCR：给扫描 PDF / 图片补文字层。

为什么放在 ingest 而不是检索层：OCR 产出的是 ``Block`` 文本，后面的分块、
BM25、claim、引用都不需要知道它来自图像。检索层不应该出现"图片 chunk"
这种第二套协议，否则引用、校验、拒答都要分叉。

引擎选 RapidOCR（ONNX Runtime，CPU）：
- 不依赖 tesseract 安装包，pip 可装；
- 中英文模型随包，离线可跑；
- 单页 200 DPI 在 CPU 上约 1–2 秒，适合入库而不是在线问答。

OCR 失败不阻断整批入库：页面保留已有文字层，或记录明确失败原因。
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path
from typing import Any

# 必须在 numpy/OpenBLAS 初始化前限制线程数；否则 56 逻辑核会申请大量线程，
# 在 24GB 机器上可能直接 "Memory allocation failed"。
_DEFAULT_THREADS = max(1, int(os.environ.get("KUMORE_OCR_THREADS", "4") or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))

import numpy as np

log = logging.getLogger(__name__)

# 200 DPI 在 24GB 机器上曾触发 ONNX bad allocation；150 对扫描件仍足够清晰，
# 又明显降低内存峰值。需要更高精度时显式设 KUMORE_OCR_DPI=200。
DEFAULT_DPI = 150
DEFAULT_MIN_CHARS = 20
DEFAULT_MIN_SCORE = 0.30

_ENGINE: Any | None = None
_ENGINE_LOCK = threading.Lock()
_PAGES_SINCE_RESET = 0


def reset_engine() -> None:
    """释放 ONNX session；长时间连续 OCR 后内存碎片会导致 bad allocation。"""
    global _ENGINE, _PAGES_SINCE_RESET
    with _ENGINE_LOCK:
        _ENGINE = None
        _PAGES_SINCE_RESET = 0
    gc.collect()


class OCRUnavailable(RuntimeError):
    """缺少 OCR 依赖/模型时的明确错误，不把 ImportError 冒到上层。"""


def ocr_mode() -> str:
    """auto（默认，只 OCR 文字层过少的页）/ always / off。"""
    mode = os.environ.get("KUMORE_OCR_MODE", "auto").strip().lower()
    return mode if mode in {"auto", "always", "off"} else "auto"


def ocr_dpi() -> int:
    try:
        return max(96, min(400, int(os.environ.get("KUMORE_OCR_DPI", DEFAULT_DPI))))
    except ValueError:
        return DEFAULT_DPI


def ocr_min_chars() -> int:
    try:
        return max(1, int(os.environ.get("KUMORE_OCR_MIN_CHARS", DEFAULT_MIN_CHARS)))
    except ValueError:
        return DEFAULT_MIN_CHARS


def _get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        threads = int(os.environ.get("KUMORE_OCR_THREADS", "4") or 4)
        # OpenBLAS 默认会按 56 逻辑核申请线程，在小内存机器上直接 bad allocation。
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, threads)))
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, threads)))
        os.environ.setdefault("MKL_NUM_THREADS", str(max(1, threads)))
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise OCRUnavailable(
                "未安装 rapidocr_onnxruntime/onnxruntime，无法 OCR。\n"
                "安装：pip install rapidocr_onnxruntime onnxruntime"
            ) from exc
        kwargs = {}
        if threads > 0:
            # RapidOCR 不同小版本参数名不同，失败就退回默认线程。
            kwargs["intra_op_num_threads"] = threads
        try:
            _ENGINE = RapidOCR(**kwargs)
        except TypeError:
            _ENGINE = RapidOCR()
        log.info("OCR 引擎就绪（RapidOCR，threads=%s）", threads)
        return _ENGINE


def _image_to_array(image: Any, *, max_side: int | None = None) -> np.ndarray:
    from PIL import Image

    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    # OCR 检测阶段的峰值内存随像素数增长；超长截图先缩到 1400px 长边。
    if max_side is None:
        try:
            max_side = max(600, int(os.environ.get("KUMORE_OCR_MAX_SIDE", "1400")))
        except ValueError:
            max_side = 1400
    if max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        pil = pil.resize(
            (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(pil.convert("RGB"))


def ocr_image(
    image: Any,
    *,
    min_score: float | None = None,
    max_side: int | None = None,
) -> str:
    """对 PIL Image / numpy 数组 / 图片路径做 OCR，返回按行拼接的文本。"""
    engine = _get_engine()
    threshold = DEFAULT_MIN_SCORE if min_score is None else float(min_score)
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as pil:
            payload = _image_to_array(pil, max_side=max_side)
    else:
        payload = _image_to_array(image, max_side=max_side)
    result, _elapse = engine(payload)
    lines: list[str] = []
    for item in result or []:
        if len(item) < 3:
            continue
        text = str(item[1] or "").strip()
        try:
            score = float(item[2])
        except (TypeError, ValueError):
            score = 0.0
        if text and score >= threshold:
            lines.append(text)
    return "\n".join(lines).strip()


def render_pdf_page(path: Path | str, page_index: int, *, dpi: int | None = None):
    """把 PDF 的第 ``page_index`` 页（0-based）渲染成 PIL Image。

    用 pypdfium2 而不是 pdf2image：不需要 Poppler 外部程序，pip 可装。
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise OCRUnavailable(
            "未安装 pypdfium2，无法把 PDF 页面渲染成图片。\n"
            "安装：pip install pypdfium2"
        ) from exc

    scale = (dpi or ocr_dpi()) / 72.0
    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf[page_index]
        try:
            bitmap = page.render(scale=scale)
            try:
                return bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        pdf.close()


def ocr_pdf_page(path: Path | str, page_index: int, *, dpi: int | None = None) -> str:
    """渲染并 OCR 单页；调用方决定哪些页需要 OCR。

    扫描页偶尔会在 ONNX 检测阶段 bad allocation；逐级降到 120/96 DPI 重试，
    宁可识别率略低，也不要把整份 PDF 标成解析失败。
    """
    global _PAGES_SINCE_RESET
    requested = dpi or ocr_dpi()
    last_exc: Exception | None = None
    # 分辨率 + 长边一起降；只在 ONNX 报 bad allocation 时才会走到后面的尝试。
    attempts = [
        (requested, 1400),
        (min(requested, 120), 1000),
        (96, 800),
        (96, 650),
    ]
    for attempt_dpi, attempt_side in dict.fromkeys(attempts):
        try:
            image = render_pdf_page(path, page_index, dpi=attempt_dpi)
            text = ocr_image(image, max_side=attempt_side)
            _PAGES_SINCE_RESET += 1
            try:
                reset_every = int(os.environ.get("KUMORE_OCR_RESET_EVERY", "4"))
            except ValueError:
                reset_every = 4
            if reset_every > 0 and _PAGES_SINCE_RESET >= reset_every:
                # 主动释放 session，避免第 N 页开始 ONNX bad allocation。
                reset_engine()
            return text
        except Exception as exc:  # noqa: BLE001 - 重试后仍失败由调用方记录
            last_exc = exc
            reset_engine()
            log.warning("OCR 第 %d 页 %d DPI 失败：%s", page_index + 1, attempt_dpi, exc)
    if last_exc is not None:
        raise last_exc
    return ""


def should_ocr_pdf_page(text_chars: int, *, mode: str | None = None) -> bool:
    mode = (mode or ocr_mode()).lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    return text_chars < ocr_min_chars()
