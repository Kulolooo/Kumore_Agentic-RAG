"""多次取样：把同一配置的 N 次评测合成"中位数 + 批间波动"。

为什么需要它：模型侧有随机性（P23/P24），同一配置两次跑的差异常有 1–2 题；
M3/M4 之前靠 `tools/m4_group_median.py` 这类一次性脚本取中位数，口径散落在脚本里。
收进 runner 之后，`python -m app eval --repeat N` 一条命令出表。

口径（先定，再看数）：

- 中位数只对**标量数值**做（率、毫秒、token 数）；``None`` 不参与，也不补 0。
- 波动用**极差**（最大值 − 最小值），不用标准差：样本少于 5 次时标准差没有解释力，
  极差至少能回答"最坏的一次差多少"。
- 结构化字段（``source_counts`` / ``refusal_reasons`` 等）保留每次的原值，不合成——
  把它们压成一个数会丢掉"哪一类变了"。
- ``aggregate_runs`` 额外给逐题翻转与**跨次合并的归因表**；多次取样报告不能只有
  汇总数字，否则"每次拒答为什么"仍然要手工翻 per-run 文件。
"""

from __future__ import annotations

from typing import Any

from app.eval.attribution import attribution, render_attribution

#: 报告头部优先列出的指标（顺序即阅读顺序）。
HEADLINE: tuple[tuple[str, str], ...] = (
    ("n_items", "题量"),
    ("hit_at_k", "Hit@5"),
    ("mrr", "MRR"),
    ("answer_rate_on_answerable", "可答回答率"),
    ("refusal_rate_on_unanswerable", "无答案拒答率"),
    ("citation_support_rate", "引用支撑率"),
    ("citation_support_strict_rate", "引用支撑率（严格）"),
    ("sentence_support_rate", "句级支撑（主口径）"),
    ("sentence_support_rate_self", "句级自证率"),
    ("claim_verbatim_rate", "claim 逐字率"),
    ("avg_latency_ms", "平均延迟 ms"),
    ("avg_completion_tokens", "平均输出 token"),
    ("errors", "调用失败"),
)

#: ``compare_aggregates`` 只比较这些"会因模型随机性波动"的指标。
#: 计数/分母类（题量、被引条数）不参与，否则极差天然随样本量变化。
HEADLINE_KEYS = {key for key, _ in HEADLINE}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def aggregate(
    summaries: list[dict[str, Any]], *, label: str, gold: str = ""
) -> dict[str, Any]:
    """把 N 份 ``EvalRun.summary()`` 合成一份"每次 + 中位数 + 极差"。"""
    if not summaries:
        raise ValueError("没有可合成的评测结果")
    keys = sorted({key for summary in summaries for key in summary if _is_number(summary[key])})
    per_run: dict[str, list[Any]] = {}
    median: dict[str, Any] = {}
    spread: dict[str, Any] = {}
    for key in keys:
        values = [summary[key] for summary in summaries if _is_number(summary.get(key))]
        per_run[key] = [summary.get(key) for summary in summaries]
        middle = _median([float(v) for v in values])
        median[key] = int(middle) if middle is not None and isinstance(values[0], int) else _round(middle)
        spread[key] = _round(max(values) - min(values)) if values else None
    return {
        "label": label,
        "gold": gold,
        "repeat": len(summaries),
        "per_run": summaries,
        "per_run_scalars": per_run,
        "median": median,
        "spread": spread,
    }


#: 逐题稳定性只跟踪这几个"会翻转"的指标；其余标量看组级中位数即可。
QUESTION_METRICS: tuple[tuple[str, str], ...] = (
    ("hit_at_k", "Hit@k"),
    ("refused", "拒答"),
    ("citation_support", "引用支撑"),
    ("sentence_support_rate", "句级支撑率"),
)


def aggregate_runs(
    runs: list[dict[str, Any]], *, label: str, gold: str = ""
) -> dict[str, Any]:
    """在 ``aggregate`` 之上补逐题稳定性。

    ``runs`` 是 ``EvalRun.to_dict()`` 的列表（含 ``summary`` 与 ``rows``）。
    逐题表回答"哪几道题在各次之间翻转"——只看组级中位数会把它藏起来。
    每题的率仍给中位数与极差；分母是该题被评到的次数。
    """
    if not runs:
        raise ValueError("没有可合成的评测结果")
    result = aggregate([run["summary"] for run in runs], label=label, gold=gold)
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for row in run.get("rows") or []:
            by_qid.setdefault(str(row.get("qid")), []).append(row)

    questions: list[dict[str, Any]] = []
    for qid, rows in sorted(by_qid.items()):
        entry: dict[str, Any] = {
            "qid": qid,
            "answerable": bool(rows[0].get("answerable")),
            "runs": len(rows),
            "values": {},
            "median": {},
            "spread": {},
            "flipped": [],
        }
        for key, title in QUESTION_METRICS:
            values = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, bool):
                    value = 1.0 if value else 0.0
                if _is_number(value):
                    values.append(float(value))
            if not values:
                continue
            middle = _median(values)
            entry["values"][key] = values
            entry["median"][key] = _round(middle)
            entry["spread"][key] = _round(max(values) - min(values))
            if len(set(values)) > 1:
                entry["flipped"].append(title)
        questions.append(entry)
    result["per_question"] = questions
    result["flipped_questions"] = [
        entry["qid"] for entry in questions if entry["flipped"]
    ]
    #: 多次取样报告也要能回答"召回来自哪条腿 / 拒答为什么"。
    #: 合并全部 run 的逐题行后算一次归因；单次报告与多次报告共用一套口径。
    all_rows = [row for run in runs for row in run.get("rows") or []]
    if all_rows:
        result["attribution"] = attribution(all_rows)
        result["attribution_markdown"] = render_attribution(
            all_rows, title="多次取样·归因"
        )
    return result


def compare_aggregates(
    baseline: dict[str, Any], candidate: dict[str, Any], *, min_metrics: int = 1
) -> dict[str, Any]:
    """同配置两组多次取样：判断中位数差是否落在已记录的批间极差内。

    **判据**：``|median_b − median_a| <= max(spread_a, spread_b)``。
    这不是显著性检验（少于 5 次样本做不了），只是一条可复算的"别把噪声当改动"闸门。
    ``spread`` 为 ``None`` 的指标跳过；分母/计数类指标不参与（它们的极差天然
    随题量，不是波动）。
    """
    rows: list[dict[str, Any]] = []
    skipped = 0
    for key, median_a in baseline.get("median", {}).items():
        median_b = candidate.get("median", {}).get(key)
        if median_a is None or median_b is None or key not in HEADLINE_KEYS:
            skipped += 1
            continue
        spread_a = baseline.get("spread", {}).get(key) or 0.0
        spread_b = candidate.get("spread", {}).get(key) or 0.0
        delta = round(float(median_b) - float(median_a), 4)
        tolerance = max(float(spread_a), float(spread_b))
        rows.append(
            {
                "key": key,
                "median_a": median_a,
                "median_b": median_b,
                "spread_a": spread_a,
                "spread_b": spread_b,
                "delta": delta,
                "tolerance": tolerance,
                "within": abs(delta) <= tolerance + 1e-9,
            }
        )
    return {
        "baseline": baseline.get("label", ""),
        "candidate": candidate.get("label", ""),
        "repeat_a": baseline.get("repeat"),
        "repeat_b": candidate.get("repeat"),
        "metrics": rows,
        "outside": [row["key"] for row in rows if not row["within"]],
        "skipped": skipped,
        "checked": len(rows),
        "passed": len(rows) >= min_metrics and all(row["within"] for row in rows),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render(result: dict[str, Any]) -> str:
    """把 ``aggregate()`` 的结果渲染成 Markdown。"""
    lines = [
        f"# 多次取样：{result['label']}",
        "",
        f"- 题集：{result['gold'] or '—'}",
        f"- 次数：{result['repeat']}（中位数 + 极差；``None`` 不参与合成）",
        "",
        "## 中位数与极差",
        "",
        "| 指标 | 中位数 | 极差 | 每次 |",
        "| --- | --- | --- | --- |",
    ]
    median = result["median"]
    spread = result["spread"]
    per_run = result["per_run_scalars"]
    shown = {key for key, _ in HEADLINE if key in median}
    for key, title in HEADLINE:
        if key not in median:
            continue
        runs = " / ".join(_fmt(v) for v in per_run.get(key, []))
        lines.append(f"| {title} | {_fmt(median[key])} | {_fmt(spread[key])} | {runs} |")
    extra = [key for key in median if key not in shown]
    if extra:
        lines += ["", "## 其余标量（同名键，未逐项翻译）", "", "| 指标 | 中位数 | 极差 |", "| --- | --- | --- |"]
        for key in extra:
            lines.append(f"| {key} | {_fmt(median[key])} | {_fmt(spread[key])} |")
    lines += [
        "",
        "> 极差 = 各次里的最大值 − 最小值。用它回答「最坏一次差多少」；"
        "样本少于 5 次时，标准差没有解释力。",
        "",
    ]
    flipped = result.get("flipped_questions") or []
    if flipped:
        lines += [
            f"## 逐题翻转（{len(flipped)} 题）",
            "",
            "> 这些题在多次运行之间值变过；组级中位数把它们藏住了。",
            "",
            "| qid | 翻转的指标 |",
            "| --- | --- |",
        ]
        by_qid = {entry["qid"]: entry for entry in result.get("per_question", [])}
        for qid in flipped:
            entry = by_qid.get(qid, {})
            lines.append(f"| {qid} | {'、'.join(entry.get('flipped', []))} |")
        lines.append("")
    if result.get("attribution_markdown"):
        lines += ["", result["attribution_markdown"]]
    return "\n".join(lines)


def render_stability(result: dict[str, Any]) -> str:
    """渲染 ``compare_aggregates`` 的结果：同配置两批的差 vs 批间极差。"""
    lines = [
        f"# 批间稳定性：{result['baseline']} vs {result['candidate']}",
        "",
        f"- 次数：{result['repeat_a']} vs {result['repeat_b']}",
        f"- 判据：|中位数差| ≤ max(两批极差)；检查 {result['checked']} 个指标，"
        f"跳过 {result['skipped']} 个（计数/非标量）",
        f"- 结论：{'**全部落在批间波动内**' if result['passed'] else '**有指标超出批间波动**'}",
        "",
        "| 指标 | 第一批中位数 | 第二批中位数 | 差 | 批间极差上界 | 在波动内 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["metrics"]:
        lines.append(
            f"| {row['key']} | {_fmt(row['median_a'])} | {_fmt(row['median_b'])} "
            f"| {_fmt(row['delta'])} | {_fmt(row['tolerance'])} "
            f"| {'✓' if row['within'] else '✗'} |"
        )
    if result["outside"]:
        lines += ["", f"超出波动的指标：{', '.join(result['outside'])}"]
    lines.append("")
    return "\n".join(lines)
