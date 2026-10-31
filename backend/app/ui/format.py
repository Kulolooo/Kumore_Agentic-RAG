"""展示层的纯函数：把后端结构压成界面上的短句。

单独成文件是为了能脱离 Streamlit 做单测——渲染层（``render.py``）只负责
把这些 dict 拼成 HTML，不再做判断。
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from app.generate.claim import locate_evidence

_HANDLE_ID = re.compile(r"\[ID:(\d{1,3})\]")
_WS = re.compile(r"\s+")
_BEFORE_PUNCT = re.compile(r"\s+([，。；：、！？,.;:!?])")
#: 展示层的隐私兜底：文件名里常见的学号/手机号/邮箱，以及"学号 姓名"连写。
#: 这不是内容脱敏（入库/发模型仍按原有边界），只是避免截图与演示把个人信息带出去。
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ID_WITH_NAME = re.compile(r"(?<!\d)\d{11}\s*[\u4e00-\u9fa5]{2,4}")
_LONG_ID = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{11})(?!\d)")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

TOOL_LABELS = {
    "search_kb": "混合检索",
    "grep_kb": "正则检索",
    "read_section": "读小节",
    "list_docs": "列文档",
    "sufficiency": "充分性判定",
    "claims": "逐字校验",
    "finalize": "合成答案",
    "single_pass": "单步生成",
}

TOOL_ICONS = {
    "search_kb": "⌕",
    "grep_kb": "≡",
    "read_section": "¶",
    "list_docs": "▤",
    "sufficiency": "✓",
    "claims": "≟",
    "finalize": "↳",
    "single_pass": "→",
}

MODE_LABELS = {"agent": "Agent", "single": "单步"}

REFUSAL_LABELS = {
    "no_hit": "没找到相关证据",
    "judged_insufficient": "有证据，但判定不足",
    "degraded_dependency": "依赖失败，降级交付",
}


def strip_handles(text: str) -> str:
    """去掉正文里的 ``[ID:n]``，但不碰 ``[n]``（claim 证据里的参考文献编号是正文）。"""
    cleaned = _HANDLE_ID.sub("", text or "")
    cleaned = _BEFORE_PUNCT.sub(r"\1", cleaned)
    return cleaned.strip()


def used_handles(text: str) -> list[int]:
    seen: list[int] = []
    for match in _HANDLE_ID.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def format_ms(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def format_tokens(tokens: int | None) -> str:
    if not tokens:
        return ""
    if tokens >= 1_000:
        return f"{tokens / 1000:.1f}k tok"
    return f"{tokens:,} tok"


def page_label(page: int | None, page_end: int | None = None) -> str:
    if not page:
        return ""
    if page_end and page_end != page:
        return f"第 {page}–{page_end} 页"
    return f"第 {page} 页"


def mask_private(text: str) -> str:
    """把路径里可机械识别的个人信息换掉。

    只处理"文件名里带学号/手机号/邮箱"这一种已观测形态；姓名单独出现时无法
    可靠识别（会误伤正常标题），属于已知限制，见 ``adversarial_m6.md``。
    """
    value = _EMAIL.sub("[邮箱]", text or "")
    value = _ID_WITH_NAME.sub("[已脱敏]", value)
    value = _PHONE.sub("[手机号]", value)
    value = _LONG_ID.sub("[编号]", value)
    return value


def file_name(rel_path: str, limit: int = 26) -> str:
    """路径太长时优先保留文件名；文件名也长就截断。"""
    cleaned = mask_private(rel_path)
    name = cleaned.replace("\\", "/").rsplit("/", 1)[-1] or "未知来源"
    return name if len(name) <= limit else name[: limit - 1] + "…"


def citation_label(citation: dict[str, Any]) -> str:
    """引用 chip 的短标签：``[ID:2] 任务书.docx · 第 3 页``。"""
    handle = citation.get("handle") or f"[ID:{citation.get('marker', 0)}]"
    parts = [handle, file_name(str(citation.get("rel_path") or ""))]
    page = page_label(citation.get("page"), citation.get("page_end"))
    if page:
        parts.append(page)
    return " · ".join(parts)


def citation_help(citation: dict[str, Any]) -> str:
    bits = [mask_private(str(citation.get("rel_path") or "未知来源"))]
    page = page_label(citation.get("page"), citation.get("page_end"))
    if page:
        bits.append(page)
    section = str(citation.get("section") or "").strip()
    if section:
        bits.append(section)
    return " · ".join(bits)


def short_arg(value: Any, limit: int = 64) -> str:
    if isinstance(value, str):
        text = _WS.sub(" ", value).strip()
    elif value is None or value == "":
        return ""
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def step_view(step: dict[str, Any]) -> dict[str, Any]:
    """把轨迹里的一步压成一行。字段都允许缺失（旧轨迹/单步模式都可读）。"""
    tool = str(step.get("tool") or "")
    args = step.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    arg = ""
    if tool in {"search_kb", "grep_kb", "list_docs", "read_section"}:
        arg = short_arg(
            args.get("query")
            or args.get("pattern")
            or args.get("section")
            or args.get("prefix")
            or ""
        )
        if tool in {"read_section"} and args.get("doc"):
            doc = short_arg(args.get("doc"), 32)
            arg = f"{doc} · {arg}" if arg else doc
        top_k = args.get("top_k")
        if top_k:
            arg = f"{arg} · top {top_k}" if arg else f"top {top_k}"
    elif tool == "sufficiency":
        answerable = args.get("answerable")
        arg = "可答" if answerable is True else "不足" if answerable is False else "解析失败"
        missing = str(args.get("missing") or "").strip()
        if missing and answerable is False:
            arg += f" · 缺：{short_arg(missing, 36)}"
    elif tool == "claims":
        count = args.get("claims")
        arg = f"claim {count} 条" if isinstance(count, int) else ""
    elif tool == "finalize":
        arg = "兜底合成" if not step.get("tokens") else ""
    elif tool == "single_pass":
        arg = str(step.get("detail") or "")

    meta_bits: list[str] = []
    n_hits = int(step.get("n_hits") or 0)
    if n_hits:
        meta_bits.append(f"+{n_hits} 块")
    duration = format_ms(step.get("ms"))
    if duration:
        meta_bits.append(duration)
    tokens = format_tokens(step.get("tokens"))
    if tokens:
        meta_bits.append(tokens)
    if step.get("ok") is False:
        meta_bits.append("异常")

    detail = str(step.get("detail") or "").strip()
    if tool == "single_pass":
        detail = ""
    return {
        "no": int(step.get("step") or 0),
        "tool": tool,
        "label": TOOL_LABELS.get(tool, tool or "步骤"),
        "icon": TOOL_ICONS.get(tool, "·"),
        "arg": arg,
        "meta": " · ".join(meta_bits),
        "detail": short_arg(detail, 120),
        "ok": step.get("ok") is not False,
        "decision": str(step.get("decision") or ""),
        "sources": list(step.get("sources") or []),
    }


def trace_summary(trace: dict[str, Any] | None) -> dict[str, Any]:
    trace = trace or {}
    steps = trace.get("steps") or []
    total_ms = sum(float(s.get("ms") or 0) for s in steps if isinstance(s, dict))
    return {
        "n_steps": len(steps),
        "duration": format_ms(total_ms) if total_ms else "",
        "tokens": format_tokens(trace.get("tokens_used")),
        "n_evidence": int(trace.get("n_evidence") or 0),
        "outcome": str(trace.get("outcome") or ""),
        "refusal_reason": trace.get("refusal_reason"),
        "degradations": list(trace.get("degradations") or []),
    }


def highlight_html(text: str, quote: str) -> str:
    """转义原文，再把逐字引文包成 ``<mark>``。

    **先转义再定位**会让偏移错位，所以先在原文上定位，再按区间转义拼接。
    找不到引文时只返回转义后的全文——不猜、不做模糊匹配。
    """
    text = text or ""
    quote = (quote or "").strip()
    span = locate_evidence(quote, text) if quote else None
    if span is None:
        return html.escape(text)
    start, end = span
    return (
        html.escape(text[:start])
        + "<mark>"
        + html.escape(text[start:end])
        + "</mark>"
        + html.escape(text[end:])
    )


def answer_mode(turn: dict[str, Any]) -> str:
    if turn.get("claims"):
        return "claim"
    if turn.get("claim_fallback"):
        return "plain"
    return "prose"


def refusal_label(reason: str | None) -> str:
    return REFUSAL_LABELS.get(str(reason or ""), "已拒答")


def mode_label(mode: str | None) -> str:
    return MODE_LABELS.get(str(mode or ""), str(mode or ""))


def human_count(n: int) -> str:
    return f"{int(n):,}"
