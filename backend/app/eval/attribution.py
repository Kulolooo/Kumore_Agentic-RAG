"""M5 归因：每一步的召回来源、每一次拒答的原因，单步与 agent 共用一张表。

**为什么要独立成模块**：M4 的 trace 已经有 ``Step.sources``，评测行已经有
``row.sources`` / ``row.refusal_reason``，但它们只被零散地写进 JSON；
报告里没有一张"来源 × 题数"和"拒答原因 × 可答/无答案"的表。M5 把这些口径
固定下来，单步（``runner``）与 agent（``agent_runner``）都调用同一组函数——
两边各写一份是 M3 已经踩过的坑（口径漂移比指标本身更难发现）。

口径（先定，再看数）
====================

一、召回来源
------------
- **块级**：分母 = 所有进入评测上下文的命中条目数（``row.sources`` 的条目，
  不是"题数"）。一题 top-5 有 5 条就贡献 5 个分母。agent 的 ``row.sources``
  是整轮注册表的全部块，所以它的块级分母比单步大，**不能直接和单步比绝对值**；
  比的是"同一路内部的构成比例"。
- **步级**（仅 agent）：分母 = 所有工具步的 ``sources`` 条目之和。
  它回答"这一步是词法/向量/grep/read 召回的"，与块级回答的不是同一个问题。
- 来源取值：``bm25`` / ``vector`` / ``bm25+vector`` / ``grep`` / ``read`` /
  ``version_sibling``。未知值原样保留（不丢、不映射），否则口径一变就静默消失。

二、拒答原因
------------
分母 = **该组全部拒答行**（不是全部题）。原因按以下优先级判定，优先级即
"责任归属"：

1. ``degraded_dependency``：生成/合成依赖失败，降级交付（最具体的失败）；
2. ``budget_exhausted``：研究阶段预算耗尽，走短合成后拒答；
3. ``no_hit``：没有任何证据；
4. ``judged_insufficient``：有证据但判定不足。

**为什么 budget 要排在 no_hit/judged 前面**：一次拒答可能同时有"证据不足"
和"预算耗尽"两个标签。只看生成侧的 ``refusal_reason`` 会把它记成
``judged_insufficient``，从而掩盖"其实是预算把研究截断了"（P68 的教训）。
``degraded_dependency`` 同理：依赖失败不是"资料不足"。

三、失败例
----------
只收**可机械判定**的行，每条都带 qid 与证据字段；不做"看起来不太好"的
主观挑选。类别见 ``FAILURE_KINDS``。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

#: 来源的显示名与一句话解释（报告表用）。
SOURCE_LABELS: dict[str, str] = {
    "bm25": "只有词法腿",
    "vector": "只有向量腿",
    "bm25+vector": "两路都召回",
    "grep": "全文 grep 补检",
    "read": "按文档读段",
    "version_sibling": "同族版本补充（不计入 top-k 指标）",
}

#: 拒答原因的显示名。不在表里的值原样显示，不丢。
REFUSAL_LABELS: dict[str, str] = {
    "no_hit": "没找到（零证据）",
    "judged_insufficient": "找到了但判定不充分",
    "budget_exhausted": "预算耗尽（研究被截断）",
    "degraded_dependency": "依赖失败（降级交付）",
}

#: 失败例类别（机械判据）。
FAILURE_KINDS: dict[str, str] = {
    "answerable_refused": "可答题被拒答",
    "unanswerable_answered": "无答案题被作答",
    "no_citation": "作答但零引用",
    "unsupported_citation": "有引用但支撑率 < 1",
    "invalid_marker": "出现越界引用编号",
    "error": "调用失败",
    "budget_on_answerable": "可答题预算耗尽",
}


def _value(row: Any, key: str, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def effective_refusal_reason(row: Any) -> str | None:
    """按模块头的优先级给出一次拒答的原因；未拒答返回 None。"""
    if not _value(row, "refused"):
        return None
    declared = _value(row, "refusal_reason")
    degradations = _value(row, "agent_degradations") or []
    if declared == "degraded_dependency" or any(
        str(item).startswith("degraded_dependency") for item in degradations
    ):
        return "degraded_dependency"
    if _value(row, "agent_budget_exhausted") or any(
        str(item).startswith("budget_exhausted") for item in degradations
    ):
        return "budget_exhausted"
    if declared in ("no_hit", "judged_insufficient"):
        return declared
    # 生成侧没给原因时按 M3 的口径推导：空上下文 = 没找到。
    if not (_value(row, "context") or _value(row, "citations")):
        return "no_hit"
    return "judged_insufficient"


def recall_attribution(rows: Iterable[Any]) -> dict[str, Any]:
    """块级召回来源。分母 = 全部命中条目数。"""
    rows = list(rows)
    hits: Counter[str] = Counter()
    questions: Counter[str] = Counter()
    total_hits = 0
    for row in rows:
        sources = list(_value(row, "sources") or [])
        total_hits += len(sources)
        for source in sources:
            hits[source] += 1
        for source in set(sources):
            questions[source] += 1
    return {
        "denominator": total_hits,
        "n_rows": len(rows),
        "hits": dict(hits),
        "questions": dict(questions),
        "shares": {
            key: round(value / total_hits, 4) if total_hits else None
            for key, value in hits.items()
        },
    }


def step_attribution(rows: Iterable[Any]) -> dict[str, Any]:
    """步级召回来源（agent）。分母 = 全部工具步的来源条目数。

    单步评测没有 ``agent_step_sources``，此时把每一行的 ``sources`` 当作
    "一步"来算——报告里必须标出这是单步的近似，不能和 agent 的步级混读。
    """
    rows = list(rows)
    hits: Counter[str] = Counter()
    n_steps = 0
    total = 0
    for row in rows:
        step_sources = _value(row, "agent_step_sources") or {}
        if step_sources:
            #: agent 行有真实工具调用数；缺失时按"一题至少一次检索"计，
            #: 避免把来源条目数误当成步数。
            n_steps += int(_value(row, "agent_tool_calls") or 1)
            for source, count in step_sources.items():
                hits[source] += int(count)
                total += int(count)
        else:
            # 单步：一行 = 一次检索；把 top-k 的来源各算一条"步级"命中。
            sources = list(_value(row, "sources") or [])
            if sources:
                n_steps += 1
                for source in sources:
                    hits[source] += 1
                    total += 1
    return {
        "denominator": total,
        "n_steps": n_steps,
        "hits": dict(hits),
        "shares": {
            key: round(value / total, 4) if total else None for key, value in hits.items()
        },
        "approximation": "单步：一行按一次检索近似为一步",
    }


def refusal_attribution(rows: Iterable[Any]) -> dict[str, Any]:
    """拒答原因。分母 = 该组全部拒答行；同时按可答/无答案拆开。"""
    rows = list(rows)
    reasons: Counter[str] = Counter()
    answerable: Counter[str] = Counter()
    unanswerable: Counter[str] = Counter()
    refused = 0
    for row in rows:
        reason = effective_refusal_reason(row)
        if reason is None:
            continue
        refused += 1
        reasons[reason] += 1
        if _value(row, "answerable"):
            answerable[reason] += 1
        else:
            unanswerable[reason] += 1
    return {
        "denominator": refused,
        "n_rows": len(rows),
        "reasons": dict(reasons),
        "answerable": dict(answerable),
        "unanswerable": dict(unanswerable),
        "shares": {
            key: round(value / refused, 4) if refused else None
            for key, value in reasons.items()
        },
    }


def failure_cases(rows: Iterable[Any], *, max_cases: int = 12) -> list[dict[str, Any]]:
    """按 ``FAILURE_KINDS`` 收机械可判定的失败例，最多 ``max_cases`` 条。"""
    cases: list[dict[str, Any]] = []
    for row in rows:
        qid = _value(row, "qid", "")
        answerable = bool(_value(row, "answerable"))
        refused = _value(row, "refused")
        if answerable and refused is True:
            cases.append({"kind": "answerable_refused", "qid": qid,
                          "detail": "该答未答；原因=" + str(effective_refusal_reason(row))})
        if not answerable and refused is False:
            cases.append({"kind": "unanswerable_answered", "qid": qid,
                          "detail": "该拒不拒"})
        if refused is False and not (_value(row, "citations") or []):
            cases.append({"kind": "no_citation", "qid": qid, "detail": "作答但零引用"})
        checked = _value(row, "checked_citations") or 0
        supported = _value(row, "supported_citations") or 0
        if checked and supported < checked:
            cases.append({
                "kind": "unsupported_citation",
                "qid": qid,
                "detail": f"支撑 {supported}/{checked}，差 {checked - supported} 条",
            })
        if _value(row, "invalid_markers"):
            cases.append({"kind": "invalid_marker", "qid": qid,
                          "detail": f"越界编号 {_value(row, 'invalid_markers')}"})
        if _value(row, "error"):
            cases.append({"kind": "error", "qid": qid, "detail": str(_value(row, "error"))[:120]})
        if answerable and _value(row, "agent_budget_exhausted"):
            cases.append({"kind": "budget_on_answerable", "qid": qid,
                          "detail": "可答题预算耗尽——这是 P68 要盯的那一类"})
        if len(cases) >= max_cases:
            break
    return cases[:max_cases]


def attribution(rows: Iterable[Any]) -> dict[str, Any]:
    """一次算齐归因三件套。报告与 JSON 都用这一份。"""
    rows = list(rows)
    return {
        "recall": recall_attribution(rows),
        "steps": step_attribution(rows),
        "refusal": refusal_attribution(rows),
        "failures": failure_cases(rows),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_attribution(rows: Iterable[Any], *, title: str = "归因") -> str:
    """渲染成一段 Markdown：来源表 + 步级表 + 拒答表 + 失败例。两边共用。"""
    data = attribution(rows)
    lines = [f"## {title}", ""]
    recall = data["recall"]
    lines += [
        f"### 召回来源（块级；分母 {recall['denominator']} 条命中，{recall['n_rows']} 题）",
        "",
        "| 来源 | 命中条数 | 占比 | 涉及题数 | 含义 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in sorted(recall["hits"]):
        lines.append(
            f"| {source} | {recall['hits'][source]} | {_fmt(recall['shares'].get(source))} "
            f"| {recall['questions'].get(source, 0)} | {SOURCE_LABELS.get(source, '')} |"
        )
    if not recall["hits"]:
        lines.append("| — | 0 | — | 0 | 无命中 |")

    steps = data["steps"]
    lines += [
        "",
        f"### 召回来源（步级；分母 {steps['denominator']} 条，{steps['n_steps']} 步）",
        "",
        "| 来源 | 条目数 | 占比 | 含义 |",
        "| --- | --- | --- | --- |",
    ]
    for source in sorted(steps["hits"]):
        lines.append(
            f"| {source} | {steps['hits'][source]} | {_fmt(steps['shares'].get(source))} "
            f"| {SOURCE_LABELS.get(source, '')} |"
        )
    if not steps["hits"]:
        lines.append("| — | 0 | — | 无工具步 |")

    refusal = data["refusal"]
    lines += [
        "",
        f"### 拒答归因（分母 {refusal['denominator']} 次拒答）",
        "",
        "| 原因 | 次数 | 占拒答 | 其中可答题 | 其中无答案题 | 含义 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for reason in sorted(refusal["reasons"]):
        lines.append(
            f"| {reason} | {refusal['reasons'][reason]} | {_fmt(refusal['shares'].get(reason))} "
            f"| {refusal['answerable'].get(reason, 0)} | {refusal['unanswerable'].get(reason, 0)} "
            f"| {REFUSAL_LABELS.get(reason, '')} |"
        )
    if not refusal["reasons"]:
        lines.append("| — | 0 | — | 0 | 0 | 无拒答 |")

    lines += ["", "### 失败例（机械判定）", ""]
    if not data["failures"]:
        lines.append("无。")
    else:
        lines += ["| 类别 | qid | 证据 |", "| --- | --- | --- |"]
        for case in data["failures"]:
            lines.append(
                f"| {case['kind']}（{FAILURE_KINDS.get(case['kind'], '')}） "
                f"| {case['qid']} | {case['detail']} |"
            )
    lines.append("")
    return "\n".join(lines)
