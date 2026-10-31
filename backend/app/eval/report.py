"""M5 报告模板与渲染器：固定七节，缺一节/缺一个分母就拒绝渲染。

为什么要有它：M1–M4 的报告每份形状都不一样，"改前/改后"有时是单次、有时是
中位数，"分母"散在正文里，失败例和未验证经常混在一起。模板把口径写死成七节：

    改动 / 改前 / 改后 / 归因 / 失败例 / 参数理由 / 未验证

**渲染器不是排版器，是校验器**：``render()`` 先跑 ``validate()``，任何一节为空、
任何指标没有分母、任何参数没写"为什么是这个值"，都直接抛 ``ValueError``。
这样下一次实验只能"照着填"，不能靠记忆补。

与 ``app/eval/aggregate.py`` / ``attribution.py`` 的分工：

- ``aggregate`` 负责"多次取样怎么合成"（中位数 + 极差 + 逐题翻转）；
- ``attribution`` 负责"来源/拒答/失败例怎么算"；
- 本模块负责"这些数字怎么写进一份可验收的报告"。

``build_metric_rows`` 从两份 ``EvalRun.summary()`` 自动取指标与分母，
避免手抄时把分母抄错；需要额外指标时再手工补 ``Metric``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: 固定七节。顺序即阅读顺序，也是 ``render`` 的标题顺序。
SECTIONS: tuple[str, ...] = (
    "改动",
    "改前",
    "改后",
    "归因",
    "失败例",
    "参数理由",
    "未验证",
)

#: 从 ``EvalRun.summary()`` 自动取数的指标：(summary 键, 显示名, 分母键)。
#: 分母键是 ``summary["denominators"]`` 里的键；``n_items`` 特殊处理为自身。
METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("answer_rate_on_answerable", "可答回答率", "answer_rate_on_answerable"),
    ("refusal_rate_on_unanswerable", "无答案拒答率", "refusal_rate_on_unanswerable"),
    ("hit_at_k", "Hit@k", "hit_at_k"),
    ("recall_at_k", "Recall@k", "recall_at_k"),
    ("mrr", "MRR", "mrr"),
    ("citation_support_rate", "引用精确率", "citation_support_rate"),
    ("citation_support_strict_rate", "引用精确率（严格）", "citation_support_strict_rate"),
    ("sentence_support_rate", "句级支撑率（主口径）", "sentence_support_rate"),
    ("sentence_support_rate_self", "句级自证率", "sentence_support_rate_self"),
    ("claim_verbatim_rate", "claim 逐字率", "claim_verbatim_rate"),
    ("span_intact_rate", "证据完整率", "span_intact_rate"),
    ("span_hit_at_k", "证据 chunk 命中率", "span_hit_at_k"),
    ("agent_rounds_mean", "平均轮次", "agent_rounds_mean"),
    ("agent_tokens_mean", "平均 tokens/题", "agent_tokens_mean"),
    ("agent_short_circuit_rate", "claim 短路率", "agent_short_circuit_rate"),
    ("avg_latency_ms", "平均生成延迟 ms", "avg_latency_ms"),
    ("avg_completion_tokens", "平均输出 token", "avg_completion_tokens"),
    ("errors", "真实调用失败", "errors"),
)


@dataclass
class Metric:
    """一行指标。``denominator`` 是文字（如 ``"39 题"`` / ``"431 条引用"``）。"""

    metric: str
    before: Any = None
    after: Any = None
    denominator: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "before": self.before,
            "after": self.after,
            "denominator": self.denominator,
            "note": self.note,
        }


@dataclass
class Failure:
    qid: str
    observed: str
    status: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "observed": self.observed,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass
class Parameter:
    name: str
    value: Any
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "why": self.why}


@dataclass
class ExperimentReport:
    title: str
    date: str
    config: str
    gold: str
    n_items: int | str
    conclusion: str
    change: str
    metrics: list[Metric] = field(default_factory=list)
    attribution: str = ""
    failures: list[Failure] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    #: 可选的批间波动说明（放在"改后"节末尾）
    stability: str = ""

    # ---------------------------------------------------------------- 校验
    def validate(self) -> list[str]:
        """返回所有不满足模板的问题；空列表才算可发布。"""
        errors: list[str] = []
        if not self.title.strip():
            errors.append("标题为空")
        if not self.date.strip():
            errors.append("日期为空")
        if not self.config.strip():
            errors.append("配置为空")
        if not str(self.gold).strip():
            errors.append("题集为空")
        if not self.conclusion.strip():
            errors.append("结论为空")
        if not self.change.strip():
            errors.append("「改动」为空")
        if not self.metrics:
            errors.append("「改前/改后」没有任何指标")
        if not self.attribution.strip():
            errors.append("「归因」为空")
        if not self.failures:
            errors.append("「失败例」为空（没有也要显式写一条'无'）")
        if not self.parameters:
            errors.append("「参数理由」为空（没有参数改动也要写默认值与理由）")
        if not self.unverified:
            errors.append("「未验证」为空（必须显式写，哪怕是'无'）")

        for metric in self.metrics:
            if not metric.metric.strip():
                errors.append("有一行指标没有名字")
            if not str(metric.denominator).strip() or str(metric.denominator).strip() in {"—", "?", "未写"}:
                errors.append(f"指标 {metric.metric!r} 没有分母")
            if metric.before is None and metric.after is None:
                errors.append(f"指标 {metric.metric!r} 改前/改后都为空")
        for failure in self.failures:
            if not failure.qid.strip():
                errors.append("失败例缺 qid")
            if not failure.observed.strip():
                errors.append(f"失败例 {failure.qid} 缺现象")
            if failure.status.strip() not in {"已修", "未修", "已知限制", "不修", "无"}:
                errors.append(
                    f"失败例 {failure.qid} 的状态 {failure.status!r} 不在 "
                    "已修/未修/已知限制/不修/无 里"
                )
        for parameter in self.parameters:
            if not parameter.why.strip():
                errors.append(f"参数 {parameter.name!r} 没有写为什么取这个值")
        return errors

    # ---------------------------------------------------------------- 渲染
    def render(self) -> str:
        errors = self.validate()
        if errors:
            raise ValueError("报告不满足模板：\n- " + "\n- ".join(errors))
        lines = [
            f"# {self.title}",
            "",
            f"- 日期：{self.date}",
            f"- 配置：{self.config}",
            f"- 题集：{self.gold}（{self.n_items} 题）",
            f"- 结论：{self.conclusion}",
            "",
            "## 1. 改动",
            "",
            self.change.strip(),
            "",
            "## 2. 改前",
            "",
            _metric_table(self.metrics, "before"),
            "",
            "## 3. 改后",
            "",
            _metric_table(self.metrics, "after"),
        ]
        if self.stability.strip():
            lines += ["", "### 批间波动", "", self.stability.strip()]
        lines += [
            "",
            "## 4. 归因",
            "",
            self.attribution.strip(),
            "",
            "## 5. 失败例",
            "",
            "| qid | 现象 | 状态 | 证据 |",
            "| --- | --- | --- | --- |",
        ]
        for failure in self.failures:
            lines.append(
                f"| {failure.qid} | {failure.observed} | {failure.status} | {failure.evidence} |"
            )
        lines += [
            "",
            "## 6. 参数理由",
            "",
            "| 参数 | 取值 | 为什么是这个值 |",
            "| --- | --- | --- |",
        ]
        for parameter in self.parameters:
            lines.append(f"| {parameter.name} | {parameter.value} | {parameter.why} |")
        lines += ["", "## 7. 未验证", ""]
        for item in self.unverified:
            lines.append(f"- {item}")
        lines.append("")
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _metric_table(metrics: Iterable[Metric], side: str) -> str:
    lines = ["| 指标 | 值 | 分母 | 说明 |", "| --- | --- | --- | --- |"]
    for metric in metrics:
        value = getattr(metric, side)
        lines.append(
            f"| {metric.metric} | {_fmt(value)} | {metric.denominator} | {metric.note} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- 从评测自动取数
def _denominator_text(summary: dict[str, Any], key: str) -> str:
    denominators = summary.get("denominators") or {}
    value = denominators.get(key)
    if value is None:
        return ""
    unit = "题"
    if key in {"citation_support_rate", "citation_support_strict_rate"}:
        unit = "条引用"
    elif key in {"sentence_support_rate", "sentence_support_rate_self", "claim_verbatim_rate"}:
        unit = "句/条"
    elif key == "span_intact_rate":
        unit = "条证据串"
    return f"{value} {unit}"


def build_metric_rows(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    keys: Iterable[str] | None = None,
    notes: dict[str, str] | None = None,
) -> list[Metric]:
    """从两份 ``EvalRun.summary()`` 生成指标行；分母自动从 ``denominators`` 取。

    ``before`` / ``after`` 也可以是 ``aggregate`` 的结果（取 ``median``），
    这样多次取样与单次报告共用同一个构造函数。
    """
    notes = notes or {}
    chosen = list(keys) if keys is not None else [key for key, _, _ in METRIC_SPECS]
    specs = {key: (title, denom_key) for key, title, denom_key in METRIC_SPECS}
    rows: list[Metric] = []
    for key in chosen:
        if key not in specs:
            raise KeyError(f"未知指标键 {key!r}，可用的见 METRIC_SPECS")
        title, denom_key = specs[key]
        before_value = _pick(before, key)
        after_value = _pick(after, key)
        denominator = _denominator_text(after, denom_key) or _denominator_text(before, denom_key)
        rows.append(
            Metric(
                metric=title,
                before=before_value,
                after=after_value,
                denominator=denominator,
                note=notes.get(key, ""),
            )
        )
    return rows


def _pick(payload: dict[str, Any], key: str) -> Any:
    if "median" in payload and isinstance(payload.get("median"), dict):
        return payload["median"].get(key)
    return payload.get(key)


# ---------------------------------------------------------------- JSON 读写
def from_spec(spec: dict[str, Any]) -> ExperimentReport:
    """把 JSON spec 转成报告对象。缺字段时让 dataclass 默认值兜住，校验再报。"""
    return ExperimentReport(
        title=str(spec.get("title") or ""),
        date=str(spec.get("date") or ""),
        config=str(spec.get("config") or ""),
        gold=str(spec.get("gold") or ""),
        n_items=spec.get("n_items", 0),
        conclusion=str(spec.get("conclusion") or ""),
        change=str(spec.get("change") or ""),
        metrics=[Metric(**row) for row in spec.get("metrics") or []],
        attribution=str(spec.get("attribution") or ""),
        failures=[Failure(**row) for row in spec.get("failures") or []],
        parameters=[Parameter(**row) for row in spec.get("parameters") or []],
        unverified=[str(item) for item in spec.get("unverified") or []],
        stability=str(spec.get("stability") or ""),
    )


def load_spec(path: str | Path) -> ExperimentReport:
    return from_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def render_spec(path: str | Path) -> str:
    return load_spec(path).render()
