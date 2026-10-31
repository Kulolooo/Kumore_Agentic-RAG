"""执行评测并生成报告。

M0 的纪律：**检索指标和回答指标分开报**，不合成一个总分。
理由很直接——如果 Recall@5 是 0，回答再流利也没有意义；反过来检索命中了但模型
没引用或拒答错，也是两个独立的问题。合成分数会把这两类错误互相抵消。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.content_identity import document_fingerprints
from app.eval.attribution import attribution, render_attribution
from app.eval.gold import GoldItem
from app.eval.sentence import check_sentence, is_supported, markers_in, split_sentences
from app.generate.answer import answer_from_hits
from app.generate.client import DeepSeekClient, LLMError
from app.retrieve.retriever import Retriever

_WS = re.compile(r"\s+")

#: 低于这个题量，任何"提升/退化"都不该被当成结论。
#: 45 题上一题的权重是 2.2 个点；10 题上一次翻转是 10 个点，噪声大于信号。
MIN_ITEMS_FOR_COMPARISON = 20

#: 证据串出现在多少份**内容不同**的文档里算"太泛"，超过就不计入严格口径。
#: 计数单位是文档内容指纹（`app/content_identity.py`），不是 chunk 行：
#: 同一段话被 chunk overlap 切成多个窗口、或在几个副本里各存一份时，chunk 计数会
#: 把一段证据算成 9~25 次，严格口径几乎每道题都被剔除（B-1）。
#: 参照 RAGFlow 的 `document.content_hash` / `ServeText` 的 sha256(text) 思路。
STRICT_SPAN_MAX = 5


def _flat(text: str) -> str:
    """去掉所有空白再比对。

    PDF/PPTX 抽出来的一行常被换行截断（如"长 4.1 米"可能变成"长 4.1\n米"），
    直接子串匹配会误判为"不支持"，所以两边都压平。
    """
    return _WS.sub("", text or "")


def supports_spans(chunk_text: str, spans: list[str]) -> bool:
    """被引用的 chunk 里是否真的出现了金标证据串。"""
    flat = _flat(chunk_text)
    if not flat:
        return False
    return any(f and f in flat for f in (_flat(s) for s in spans))


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def _percentile(values: list[float], q: float) -> float | None:
    """线性插值分位数。报告里的 P50/P95 用它算，避免只给平均值掩盖长尾。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 2)


def _auc(positive: list[float], negative: list[float]) -> float | None:
    """top-1 分数区分"可答/无答案"的 ROC-AUC（无阈值，尺度无关）。

    用法：AUC ≈ 0.5 表示这个分数对判断"能不能答"没有信息量，
    这正是 P16 的结论（BM25 分数不能当拒答判据）；0.5 以上才有可用的分离度。
    """
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for n in negative:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return round(wins / (len(positive) * len(negative)), 4)


class _SkipGeneration(Exception):
    """内部信号：--offline 模式下不调用模型。"""


@dataclass
class EvalRow:
    qid: str
    question: str
    category: str
    answerable: bool
    hit_at_k: bool | None = None
    recall_at_k: float | None = None
    first_gold_rank: int | None = None
    mrr: float | None = None
    #: 分块侧指标：金标证据串是否**完整落在同一个 chunk**里。
    #: ``span_total`` 是本题证据串数，``span_intact`` 是完整落在某一块里的条数。
    #: 它比 Hit@5 灵敏：换分块策略时，文档还在 top-5、证据却被切开，这个指标会掉。
    span_total: int = 0
    span_intact: int = 0
    #: 命中的 chunk（不是整份文档）里是否含一条完整证据串。
    span_hit_at_k: bool | None = None
    refused: bool | None = None
    #: 拒答原因：no_hit（没检索到）/ judged_insufficient（检索到了但判定不够）。
    #: RAGFlow 逆向点名的验收之一（`08-对标复刻清单.md:306`）：两者修法完全不同。
    refusal_reason: str | None = None
    refuse_correct: bool | None = None
    n_citations: int = 0
    #: 引用的 chunk 里确实含有金标证据串的条数（需要 gold_spans 非空）
    supported_citations: int = 0
    checked_citations: int = 0
    citation_support: float | None = None
    #: 严格口径：只用"在全库出现次数 <= STRICT_SPAN_MAX"的证据串来判支撑
    supported_strict: int = 0
    checked_strict: int = 0
    citation_support_strict: float | None = None
    ambiguous_spans: list[str] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    # ---- M3：句级与 claim 侧指标 ----
    #: 答案切出来的句子总数 / 其中带编号的句子数
    sentence_total: int = 0
    sentence_cited: int = 0
    #: 带编号的句子里，被引 chunk 真的含该题证据串的条数（金标代理，主口径）
    sentence_supported: int = 0
    #: 同一批句子里"任一被引块含证据"的条数。过度引用会让它虚高，只作敏感性对照。
    sentence_supported_any: int = 0
    #: 同上，但只用"全库低频"的严格证据串
    sentence_supported_strict: int = 0
    #: 带编号的句子里，逐字定位 + 数字一致性都通过的条数（无金标自证口径）
    sentence_self_supported: int = 0
    #: 该题的金标证据是否**进了上下文**。句级主指标只在这个子集上算：
    #: 证据没进上下文时，任何引用都不可能被支撑，那是检索的锅不是引用的锅。
    gold_span_in_context: bool | None = None
    #: 编号写法计数，用来判断模型有没有遵守 [ID:n] 契约
    marker_form_id: int = 0
    marker_form_legacy: int = 0
    #: claim 行解析与逐字校验的打点（claim 变体才有内容）
    claim_total: int = 0
    claim_with_evidence: int = 0
    claim_verbatim: int = 0
    claim_unresolved_handles: list[int] = field(default_factory=list)
    #: claim 解析失败、已降级成普通回答
    claim_fallback: bool = False
    claim_unparsed_lines: int = 0
    #: 逐句明细（含金标判定），供人工抽检抽样
    sentences: list[dict[str, Any]] = field(default_factory=list)
    # ---- M4：agent 轨迹侧打点（单步评测里全为 0/空）----
    #: 模型轮次数（agent）或 1（单步）
    agent_rounds: int = 0
    #: 工具调用次数与各自次数
    agent_tool_calls: int = 0
    agent_tools: list[str] = field(default_factory=list)
    #: claim 短路是否命中（agent 专属）
    agent_short_circuit: bool | None = None
    #: 本次 ask 的依赖失败标记（非空 = 走了降级路径）
    agent_degradations: list[str] = field(default_factory=list)
    #: 整轮消耗的 token（按 turn 结算）
    agent_tokens: int = 0
    #: 真正进了合成上下文的块数（M4 改动 1 之后可能远小于发号块数）
    agent_synthesis_blocks: int = 0
    #: 发布清单块数（只含被引，M4 改动 2）
    agent_published_blocks: int = 0
    #: 步级召回来源：``{source: 条目数}``，来自 ``Trace.tool_steps[*].sources``。
    #: 块级 ``sources`` 回答"最终上下文里有哪些腿"，这里回答"每一步是哪条腿"；
    #: 两者分母不同，报告里分表列（见 ``app/eval/attribution.py`` 模块头）。
    agent_step_sources: dict[str, int] = field(default_factory=dict)
    #: 本次是否因预算耗尽被截断（P68）。与 ``agent_degradations`` 冗余，
    #: 单独成字段是为了让汇总/归因不必解析字符串。
    agent_budget_exhausted: bool = False
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    #: 本题检索（含过滤/重排）耗时，毫秒。M2 的验收项之一，和生成延迟分开记。
    retrieval_ms: float | None = None
    #: 检索第一名得分。无答案题上这个分数衡量"检索有多像找到了答案"。
    top1_score: float | None = None
    #: top-k 每条命中的召回来源（bm25 / vector / bm25+vector / version_sibling）。
    sources: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class EvalRun:
    label: str
    top_k: int
    cutoff: float
    model: str | None
    #: 这次跑的是哪套配置。不记下来的话，"两份报告谁比谁好"根本无从解释——
    #: M1 的 4 组对照就靠这几行区分。
    mode: str = ""
    fusion: str = ""
    chunk_strategy: str = ""
    n_chunks: int = 0
    avg_chunk_len: float | None = None
    #: 提示词变体（strict / tight）。拒答实验必须把这一项记下来，否则两份报告没法比。
    prompt_variant: str = "strict"
    #: 题集文件名。现在有两套金标（45 题 / 5 题两跳），不写清楚的话
    #: "39/39" 和 "5/5" 会被当成同一把尺子（M5-0 的题集口径）。
    gold_name: str = ""
    #: 重排配置。报告头必须写清楚，否则 E/F 组与 C/D 组的差异会被归因错。
    rerank: bool = False
    rerank_model: str = ""
    rerank_candidates: int = 0
    #: 有多少题的检索发生了"重排失败→回退原序"。这是 fail-open 的可用性指标。
    rerank_failures: int = 0
    #: 元数据过滤条件的描述（含默认的 status=active）。
    doc_filter: str = ""
    #: 分块侧指标（M5）：边界分布。口径见 ``app/eval/chunk_metrics.py`` 模块头；
    #: 膨胀率需要同语料 fixed 基线，只在 M1 矩阵里算，不塞进单次评测。
    chunk_side: dict[str, Any] = field(default_factory=dict)
    rows: list[EvalRow] = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0

    # ---- 汇总 ----
    def summary(self) -> dict[str, Any]:
        answerable = [r for r in self.rows if r.answerable]
        unanswerable = [r for r in self.rows if not r.answerable]
        scored = [r for r in self.rows if r.hit_at_k is not None]
        answered = [r for r in self.rows if r.refused is not None]
        not_refused = [r for r in answered if not r.refused]
        # 离线模式（只评检索）没有生成结果：分母必须是"真的答过"的行，
        # 否则把所有行都算成"未回答"，报告会出现 0.0 这种假结论。
        answerable_answered = [r for r in answerable if r.refused is not None]
        unanswerable_answered = [r for r in unanswerable if r.refused is not None]
        refusal_reasons: dict[str, int] = {}
        for row in self.rows:
            if row.refusal_reason:
                refusal_reasons[row.refusal_reason] = refusal_reasons.get(row.refusal_reason, 0) + 1

        def rate(hits: int, total: int) -> float | None:
            return round(hits / total, 4) if total else None

        retrieval_times = [r.retrieval_ms for r in self.rows if r.retrieval_ms is not None]
        source_counts: dict[str, int] = {}
        for row in self.rows:
            for src in row.sources:
                source_counts[src] = source_counts.get(src, 0) + 1
        # 无答案误召回：先把可答题的 top-1 分数取 10% 分位当阈值
        # （保证 90% 的可答题在上方），再看无答案题有多少条越过了它。
        # 同时报 AUC——它不需要选阈值，避免"阈值是我挑的"这种自欺。
        a_top1 = [
            r.top1_score for r in self.rows if r.answerable and r.top1_score is not None
        ]
        u_top1 = [
            r.top1_score for r in self.rows if not r.answerable and r.top1_score is not None
        ]
        threshold = _percentile(a_top1, 0.10)
        false_recall = rate(
            sum(1 for s in u_top1 if threshold is not None and s >= threshold),
            len(u_top1),
        )
        answerable_kept = rate(
            sum(1 for s in a_top1 if threshold is not None and s >= threshold), len(a_top1)
        )

        summary = {
            "label": self.label,
            "gold": self.gold_name,
            "n_items": len(self.rows),
            "small_sample": len(self.rows) < MIN_ITEMS_FOR_COMPARISON,
            "comparison_min_items": MIN_ITEMS_FOR_COMPARISON,
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "top_k": self.top_k,
            "cutoff": self.cutoff,
            "model": self.model,
            "mode": self.mode,
            "fusion": self.fusion,
            "chunk_strategy": self.chunk_strategy,
            "n_chunks": self.n_chunks,
            "avg_chunk_len": self.avg_chunk_len,
            # ---- M5：分块侧边界（分母 = 检索单元数）----
            "chunk_boundary_ok_rate": (self.chunk_side or {}).get("boundary_ok_rate"),
            "chunk_boundary_denominator": (self.chunk_side or {}).get("denominator"),
            "chunk_mid_sentence_rate": (self.chunk_side or {}).get("mid_sentence_rate"),
            "chunk_cut_mid_token_rate": (self.chunk_side or {}).get("cut_mid_token_rate"),
            "prompt_variant": self.prompt_variant,
            "rerank": self.rerank,
            "rerank_model": self.rerank_model,
            "rerank_candidates": self.rerank_candidates,
            "rerank_failures": self.rerank_failures,
            "doc_filter": self.doc_filter,
            # ---- 检索延迟与召回来源（M2 验收项）----
            "retrieval_p50_ms": _percentile(retrieval_times, 0.50),
            "retrieval_p95_ms": _percentile(retrieval_times, 0.95),
            "source_counts": source_counts,
            # ---- 无答案误召回（M2 验收项）----
            "no_answer_false_recall": false_recall,
            "no_answer_false_recall_threshold": threshold,
            "answerable_kept_at_threshold": answerable_kept,
            "answerable_top1_median": _percentile(a_top1, 0.50),
            "unanswerable_top1_median": _percentile(u_top1, 0.50),
            "score_auc": _auc(a_top1, u_top1),
            "hit_at_k": rate(sum(1 for r in scored if r.hit_at_k), len(scored)),
            "recall_at_k": (
                round(sum(r.recall_at_k or 0 for r in scored) / len(scored), 4)
                if scored else None
            ),
            "mrr": (
                round(sum(r.mrr or 0 for r in scored) / len(scored), 4) if scored else None
            ),
            # 分块侧：证据串完整落在单个 chunk 的比例。分母是所有"有证据串"的题，
            # 不是所有题——没有证据串的题（无答案题）不该拉低这个指标。
            "span_intact_rate": rate(
                sum(r.span_intact for r in self.rows), sum(r.span_total for r in self.rows)
            ),
            "span_hit_at_k": rate(
                sum(1 for r in self.rows if r.span_hit_at_k),
                sum(1 for r in self.rows if r.span_hit_at_k is not None),
            ),
            # 可答题：不该拒答，拒了算"过度保守"
            "answer_rate_on_answerable": rate(
                sum(1 for r in answerable_answered if r.refused is False),
                len(answerable_answered),
            ),
            # 无答案题：该拒就必须拒，这是 M0 最核心的能力之一
            "refusal_rate_on_unanswerable": rate(
                sum(1 for r in unanswerable_answered if r.refused is True),
                len(unanswerable_answered),
            ),
            "refusal_reasons": refusal_reasons,
            "refuse_accuracy": rate(
                sum(1 for r in answered if r.refuse_correct), len(answered)
            ),
            "no_citation_rate": rate(
                sum(1 for r in not_refused if r.n_citations == 0), len(not_refused)
            ),
            "invalid_marker_rows": sum(1 for r in answered if r.invalid_markers),
            # 引用准确率的客观代理：被引 chunk 是否真的含有金标证据串。
            # 它只能证明"引文里出现该证据"，不能证明"该证据支撑整句"——后者仍需人工抽检。
            "citation_support_rate": rate(
                sum(r.supported_citations for r in self.rows),
                sum(r.checked_citations for r in self.rows),
            ),
            # 严格口径：剔除在全库高频出现的证据串，避免"哪个 chunk 都算命中"
            "citation_support_strict_rate": rate(
                sum(r.supported_strict for r in self.rows),
                sum(r.checked_strict for r in self.rows),
            ),
            "strict_span_max": STRICT_SPAN_MAX,
            "ambiguous_spans": sum(len(r.ambiguous_spans) for r in self.rows),
            # ---- M3：句级指标 ----
            # 口径提醒：分母是"带编号的句子"，不是"所有句子"。
            # 两句都不是同一种东西——"每句都标注"和"标注得对不对"要分开看。
            "sentence_total": sum(r.sentence_total for r in self.rows),
            "sentence_cited": sum(r.sentence_cited for r in self.rows),
            "sentence_citation_rate": rate(
                sum(r.sentence_cited for r in self.rows),
                sum(r.sentence_total for r in self.rows),
            ),
            # 主口径：金标代理。只收"本题有金标证据串"的句子做分母。
            "sentence_gold_checked": sum(
                r.sentence_cited for r in self.rows if r.span_total > 0
            ),
            "sentence_support_rate": rate(
                sum(r.sentence_supported for r in self.rows),
                sum(r.sentence_cited for r in self.rows if r.span_total > 0),
            ),
            # 敏感性对照：只要"任一被引块"含证据就算。过度引用会把它抬到接近
            # "金标块是否进上下文"的水平，所以它不能当主口径。
            "sentence_support_rate_any": rate(
                sum(r.sentence_supported_any for r in self.rows),
                sum(r.sentence_cited for r in self.rows if r.span_total > 0),
            ),
            "sentence_support_rate_strict": rate(
                sum(r.sentence_supported_strict for r in self.rows),
                sum(r.sentence_cited for r in self.rows if r.checked_strict > 0),
            ),
            # 条件化口径：只在"金标证据进了上下文"的题上算。
            # 证据没进上下文时任何引用都不可能被支撑，混进来就是拿检索的锅罚引用。
            "gold_context_rows": sum(
                1 for r in self.rows if r.gold_span_in_context is True
            ),
            "sentence_support_rate_gold_context": rate(
                sum(
                    r.sentence_supported
                    for r in self.rows
                    if r.gold_span_in_context is True
                ),
                sum(
                    r.sentence_cited
                    for r in self.rows
                    if r.gold_span_in_context is True and r.span_total > 0
                ),
            ),
            "sentence_cited_gold_context": sum(
                r.sentence_cited
                for r in self.rows
                if r.gold_span_in_context is True and r.span_total > 0
            ),
            # 无金标自证口径：逐字定位 + 数字一致性，换语料也能算
            "sentence_support_rate_self": rate(
                sum(r.sentence_self_supported for r in self.rows),
                sum(r.sentence_cited for r in self.rows),
            ),
            "marker_form_id": sum(r.marker_form_id for r in self.rows),
            "marker_form_legacy": sum(r.marker_form_legacy for r in self.rows),
            # ---- M3：claim 指标 ----
            "claim_total": sum(r.claim_total for r in self.rows),
            "claim_with_evidence": sum(r.claim_with_evidence for r in self.rows),
            "claim_verbatim": sum(r.claim_verbatim for r in self.rows),
            # 逐字率的分母是"给了引文的 claim"：没给引文的谈不上逐字，
            # 算进分母会让"少给引文"看起来像"引文更准"。
            "claim_verbatim_rate": rate(
                sum(r.claim_verbatim for r in self.rows),
                sum(r.claim_with_evidence for r in self.rows),
            ),
            "claim_fallback_rows": sum(1 for r in self.rows if r.claim_fallback),
            "claim_unparsed_lines": sum(r.claim_unparsed_lines for r in self.rows),
            "claim_unresolved_handles": sum(
                len(r.claim_unresolved_handles) for r in self.rows
            ),
            # ---- M4：agent 轨迹侧（单步评测里全是默认值）----
            "agent_rounds_mean": (
                round(sum(r.agent_rounds for r in self.rows) / len(self.rows), 2)
                if self.rows else None
            ),
            "agent_tool_calls": sum(r.agent_tool_calls for r in self.rows),
            "agent_tool_usage": _count_tools(self.rows),
            "agent_short_circuit_rows": sum(
                1 for r in self.rows if r.agent_short_circuit is True
            ),
            "agent_short_circuit_rate": rate(
                sum(1 for r in self.rows if r.agent_short_circuit is True),
                sum(1 for r in self.rows if r.agent_short_circuit is not None),
            ),
            "agent_degraded_rows": sum(1 for r in self.rows if r.agent_degradations),
            "agent_tokens_total": sum(r.agent_tokens for r in self.rows),
            "agent_tokens_mean": (
                round(sum(r.agent_tokens for r in self.rows) / len(self.rows), 1)
                if self.rows else None
            ),
            # ---- M5：步级归因与预算耗尽（P68）----
            "step_source_counts": _count_step_sources(self.rows),
            "step_source_denominator": sum(
                sum((r.agent_step_sources or {}).values()) for r in self.rows
            ),
            "budget_exhausted_rows": sum(
                1 for r in self.rows if r.agent_budget_exhausted
            ),
            "budget_exhausted_answerable_rows": sum(
                1 for r in self.rows if r.agent_budget_exhausted and r.answerable
            ),
            "budget_exhausted_unanswerable_rows": sum(
                1 for r in self.rows if r.agent_budget_exhausted and not r.answerable
            ),
            "budget_exhausted_refused_rows": sum(
                1 for r in self.rows if r.agent_budget_exhausted and r.refused is True
            ),
            "budget_exhausted_answered_rows": sum(
                1 for r in self.rows if r.agent_budget_exhausted and r.refused is False
            ),
            "agent_synthesis_blocks_mean": (
                round(
                    sum(r.agent_synthesis_blocks for r in self.rows) / len(self.rows), 1
                )
                if self.rows else None
            ),
            "agent_published_blocks_mean": (
                round(
                    sum(r.agent_published_blocks for r in self.rows) / len(self.rows), 1
                )
                if self.rows else None
            ),
            "agent_evidence_blocks_mean": (
                round(
                    sum(len(r.context) for r in self.rows) / len(self.rows), 1
                )
                if self.rows else None
            ),
            "errors": sum(1 for r in self.rows if r.error),
            "avg_latency_ms": (
                round(
                    sum(r.latency_ms or 0 for r in answered if r.latency_ms) / len(answered), 1
                )
                if answered else None
            ),
            "avg_completion_tokens": (
                round(
                    sum(r.completion_tokens or 0 for r in answered) / len(answered), 1
                )
                if answered else None
            ),
            "duration_s": self.duration_s,
            "started_at": self.started_at,
        }
        #: ---- M5：每个指标的分母（报告模板要求"没有分母不算指标"）----
        #: 分母是整数（条目/题/句），不是比率；``None`` 表示该指标本次不可算。
        #: 报告渲染器会拒绝"有值但没有分母"的指标行。
        summary["denominators"] = {
            "n_items": len(self.rows),
            "hit_at_k": len(scored),
            "recall_at_k": len(scored),
            "mrr": len(scored),
            "answer_rate_on_answerable": len(answerable_answered),
            "refusal_rate_on_unanswerable": len(unanswerable_answered),
            "refuse_accuracy": len(answered),
            "no_citation_rate": len(not_refused),
            "no_answer_false_recall": len(u_top1),
            "answerable_kept_at_threshold": len(a_top1),
            "citation_support_rate": sum(r.checked_citations for r in self.rows),
            "citation_support_strict_rate": sum(r.checked_strict for r in self.rows),
            "span_intact_rate": sum(r.span_total for r in self.rows),
            "span_hit_at_k": sum(1 for r in self.rows if r.span_hit_at_k is not None),
            "sentence_citation_rate": sum(r.sentence_total for r in self.rows),
            "sentence_support_rate": sum(
                r.sentence_cited for r in self.rows if r.span_total > 0
            ),
            "sentence_support_rate_any": sum(
                r.sentence_cited for r in self.rows if r.span_total > 0
            ),
            "sentence_support_rate_strict": sum(
                r.sentence_cited for r in self.rows if r.checked_strict > 0
            ),
            "sentence_support_rate_gold_context": sum(
                r.sentence_cited
                for r in self.rows
                if r.gold_span_in_context is True and r.span_total > 0
            ),
            "sentence_support_rate_self": sum(r.sentence_cited for r in self.rows),
            "claim_verbatim_rate": sum(r.claim_with_evidence for r in self.rows),
            "agent_rounds_mean": len(self.rows),
            "agent_tokens_mean": len(self.rows),
            "agent_synthesis_blocks_mean": len(self.rows),
            "agent_published_blocks_mean": len(self.rows),
            "agent_evidence_blocks_mean": len(self.rows),
            "agent_short_circuit_rate": sum(
                1 for r in self.rows if r.agent_short_circuit is not None
            ),
            "errors": len(self.rows),
            "avg_latency_ms": len(answered),
            "avg_completion_tokens": len(answered),
        }
        return summary
    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "rows": [r.to_dict() for r in self.rows],
            #: M5：归因与 summary 同级，任何一份存盘报告都能直接回答
            #: "召回来自哪条腿 / 拒答原因是什么 / 有哪些机械失败例"。
            "attribution": attribution(self.rows),
        }

    def save(self, runs_dir: Path) -> tuple[Path, Path]:
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        json_path = runs_dir / f"eval_{self.label}_{stamp}.json"
        md_path = runs_dir / f"eval_{self.label}_{stamp}.md"
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path

    def to_markdown(self) -> str:
        s = self.summary()
        lines: list[str] = []
        lines.append(f"# 评测报告：{self.label}")
        lines.append("")
        lines.append(f"- 时间：{self.started_at}")
        lines.append(f"- 题集：{self.gold_name or '—'}（{len(self.rows)} 题）")
        lines.append(f"- top_k：{self.top_k}　分数相对截断：{self.cutoff}　模型：{self.model}")
        lines.append(
            f"- 检索：{self.mode or '—'}"
            + (f"（{self.fusion} 融合）" if self.fusion else "")
            + (f"　重排：{self.rerank_model}（前 {self.rerank_candidates} 条）" if self.rerank else "")
            + f"　过滤：{self.doc_filter or '—'}"
            + f"　分块：{self.chunk_strategy or '—'}　chunk 数：{self.n_chunks}"
            + (f"　平均长度：{self.avg_chunk_len} 字" if self.avg_chunk_len else "")
            + f"　提示词：{self.prompt_variant}"
        )
        lines.append(f"- 题量：{s['n_items']}（可答 {s['n_answerable']} / 无答案 {s['n_unanswerable']}）")
        lines.append(f"- 耗时：{s['duration_s']} s")
        if s["small_sample"]:
            lines.append("")
            lines.append(
                f"> ⚠ **小样本（{s['n_items']} 题 < {s['comparison_min_items']}）**："
                "一次翻转就是 "
                f"{round(100 / max(s['n_items'], 1), 1)} 个点，"
                "这份报告只能当冒烟测试，**不能**和 20 题以上的对照实验并排下结论。"
            )
        lines.append("")
        lines.append("## 汇总")
        lines.append("")
        lines.append("| 指标 | 数值 | 说明 |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| Hit@{self.top_k} | {_num(s['hit_at_k'])} | 金标文档是否出现在前 {self.top_k} 条 |")
        lines.append(f"| Recall@{self.top_k} | {_num(s['recall_at_k'])} | 多金标题按比例计 |")
        lines.append(f"| MRR | {_num(s['mrr'])} | 首个金标首次出现的排名倒数 |")
        lines.append(
            f"| 检索 P50 / P95 | {_num(s['retrieval_p50_ms'])} / {_num(s['retrieval_p95_ms'])} ms "
            f"| 单题检索耗时（含融合/过滤/重排），不调用模型 |"
        )
        lines.append(
            f"| 无答案误召回率 | {_num(s['no_answer_false_recall'])} | "
            f"无答案题的 top-1 分数 ≥ {_num(s['no_answer_false_recall_threshold'])}"
            f"（可答题 P10 阈值）的比例；越低越好 |"
        )
        lines.append(
            f"| 分数可分性 AUC | {_num(s['score_auc'])} | "
            f"top-1 分数区分「可答 / 无答案」的能力，0.5 = 无信息 |"
        )
        lines.append(
            f"| 证据完整率 | {_num(s['span_intact_rate'])} | "
            f"金标证据串完整落在同一个 chunk 的比例（分块侧指标） |"
        )
        lines.append(
            f"| 证据 chunk 命中率 | {_num(s['span_hit_at_k'])} | "
            f"命中片段（不是整份文档）里含完整证据串 |"
        )
        lines.append(
            f"| 分块边界合理率 | {_num(s.get('chunk_boundary_ok_rate'))} | "
            f"以句末/段落尾结束的检索单元占比（分母 {s.get('chunk_boundary_denominator')} 块；"
            f"口径见 chunk_metrics 模块头） |"
        )
        lines.append(
            f"| 分块句中切断率 | {_num(s.get('chunk_mid_sentence_rate'))} | "
            f"不以任何标点结束（含切断型号/英文 {_num(s.get('chunk_cut_mid_token_rate'))}） |"
        )
        lines.append(
            f"| 可答题回答率 | {_num(s['answer_rate_on_answerable'])} | "
            f"该答的答了（未被拒答）；离线模式为 — |"
        )
        lines.append(
            f"| 无答案题拒答率 | {_num(s['refusal_rate_on_unanswerable'])} | 该拒的拒了 |"
        )
        lines.append(f"| 拒答判断准确率 | {_num(s['refuse_accuracy'])} | 两类错误合并看 |")
        reasons = s.get("refusal_reasons") or {}
        reason_text = (
            f"no_hit {reasons.get('no_hit', 0)} / judged_insufficient "
            f"{reasons.get('judged_insufficient', 0)}"
            if reasons
            else "—"
        )
        lines.append(f"| 拒答归因 | {reason_text} | 没找到 vs 判定不充分 |")
        lines.append(f"| 无引用率 | {_num(s['no_citation_rate'])} | 未拒答却没有 [n] 的比例 |")
        lines.append(
            f"| 引用支撑率 | {_num(s['citation_support_rate'])} | 被引片段里是否真含金标证据串 |"
        )
        lines.append(
            f"| 引用支撑率（严格） | {_num(s['citation_support_strict_rate'])} | "
            f"剔除全库出现 >{s['strict_span_max']} 次的泛证据串后重算 |"
        )
        lines.append(f"| 越界引用题数 | {s['invalid_marker_rows']} | 引用了上下文里不存在的编号 |")
        lines.append(
            f"| 句级引用率 | {_num(s['sentence_citation_rate'])} | "
            f"带编号的句子占全部句子的比例（分母 {s['sentence_total']} 句） |"
        )
        lines.append(
            f"| 句级支撑率 | {_num(s['sentence_support_rate'])} | "
            f"带编号的句子中，被引块里含该题金标证据串、**且过半被引块都含**的比例；"
            f"分母 {s['sentence_gold_checked']} 句（只收有金标证据的题） |"
        )
        lines.append(
            f"| 句级支撑率（任一被引块） | {_num(s['sentence_support_rate_any'])} | "
            f"敏感性对照：只要有一条被引块含证据就算。过度引用会把它抬高，不作结论 |"
        )
        lines.append(
            f"| 句级支撑率（严格） | {_num(s['sentence_support_rate_strict'])} | "
            f"剔除全库高频证据串后重算 |"
        )
        lines.append(
            f"| 句级支撑率（金标进上下文） | {_num(s['sentence_support_rate_gold_context'])} | "
            f"条件化口径：只在证据已进上下文的 {s['gold_context_rows']} 道上算，"
            f"分母 {s['sentence_cited_gold_context']} 句 |"
        )
        lines.append(
            f"| 句级自证率 | {_num(s['sentence_support_rate_self'])} | "
            f"逐字定位 + 数字一致性，不依赖金标（换语料也能算） |"
        )
        lines.append(
            f"| 编号写法 | [ID:n] {s['marker_form_id']} / [n] {s['marker_form_legacy']} | "
            f"后者是过渡期兼容写法，非 0 说明没完全遵守契约 |"
        )
        if s["claim_total"] or s["claim_fallback_rows"]:
            lines.append(
                f"| claim 逐字率 | {_num(s['claim_verbatim_rate'])} | "
                f"{s['claim_verbatim']}/{s['claim_with_evidence']} 条 claim 的引文能在被引块里"
                f"逐字定位（共 {s['claim_total']} 行） |"
            )
            lines.append(
                f"| claim 降级题数 | {s['claim_fallback_rows']} | "
                f"一行 claim 都没解析出来，按普通回答交付 |"
            )
        lines.append(f"| 调用失败 | {s['errors']} | 接口/网络错误 |")
        lines.append(f"| 平均延迟 | {_num(s['avg_latency_ms'])} ms | 仅统计调用成功的题 |")
        lines.append(f"| 平均输出 token | {_num(s['avg_completion_tokens'])} | — |")
        if s.get("budget_exhausted_rows"):
            lines.append(
                f"| 预算耗尽行 | {s['budget_exhausted_rows']}（可答 "
                f"{s['budget_exhausted_answerable_rows']} / 无答案 "
                f"{s['budget_exhausted_unanswerable_rows']}） | "
                f"其中拒答 {s['budget_exhausted_refused_rows']}、照常作答 "
                f"{s['budget_exhausted_answered_rows']}（P68） |"
            )
        lines.append("")
        counts = s.get("source_counts") or {}
        if counts:
            lines.append("## 召回来源（top-k 命中计数）")
            lines.append("")
            lines.append("| 来源 | 条数 | 说明 |")
            lines.append("| --- | --- | --- |")
            for key in sorted(counts):
                meaning = {
                    "bm25": "只有词法腿召回",
                    "vector": "只有向量腿召回",
                    "bm25+vector": "两路都召回",
                    "version_sibling": "同族版本补充（不计入 top-k 指标）",
                }.get(key, "")
                lines.append(f"| {key} | {counts[key]} | {meaning} |")
            lines.append("")
        if s.get("sentence_gold_checked") == 0 and s.get("sentence_cited"):
            lines.append(
                "> ⚠ 本批没有可用的金标证据串：**句级支撑率不可用**（它是评测专用指标）。"
                "换没有金标的语料时看「句级自证率」——逐字定位 + 数字一致性，不依赖金标。"
            )
            lines.append("")
        lines.append("## 逐题")
        lines.append("")
        lines.append("| # | 类别 | 问题 | 命中 | 金标名次 | 证据 | 拒答 | 引用数 | 引用支撑 | 句级 | 回答摘要 | 备注 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        for row in self.rows:
            hit = "—" if row.hit_at_k is None else ("✓" if row.hit_at_k else "✗")
            rank = "—" if row.first_gold_rank is None else str(row.first_gold_rank)
            refused = "—" if row.refused is None else ("是" if row.refused else "否")
            support = (
                "—"
                if row.citation_support is None
                else f"{row.supported_citations}/{row.checked_citations}"
            )
            span = (
                "—" if row.span_total == 0 else f"{row.span_intact}/{row.span_total}"
            )
            sentence = (
                "—"
                if not row.sentence_total
                else f"{row.sentence_supported}/{row.sentence_cited}（自证 {row.sentence_self_supported}）"
            )
            question = _clip(row.question, 26)
            answer = _clip(row.answer.replace("\n", " "), 34) if row.answer else (row.error or "")
            note = "；".join(row.warnings)
            if row.refusal_reason:
                note = f"拒答:{row.refusal_reason}" + (f"；{note}" if note else "")
            if row.error:
                note = f"ERROR: {_clip(row.error, 40)}"
            lines.append(
                f"| {row.qid} | {row.category} | {question} | {hit} | {rank} | {span} "
                f"| {refused} | {row.n_citations} | {support} | {sentence} | {answer} | {_clip(note, 40)} |"
            )
        lines.append("")
        lines.append("## 指标分母")
        lines.append("")
        lines.append("> 每个率的分母写在这里；分母为 0 的指标在汇总表里显示 `—`。")
        lines.append("")
        lines.append("| 指标键 | 分母 |")
        lines.append("| --- | --- |")
        for key in sorted(s.get("denominators") or {}):
            lines.append(f"| {key} | {s['denominators'][key]} |")
        lines.append("")
        lines.append(render_attribution(self.rows))
        return "\n".join(lines)


def _num(value: Any) -> Any:
    """报告里 None 不该显示成 "None"——离线模式大量字段是 None。"""
    return "—" if value is None else value


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _count_tools(rows) -> dict[str, int]:
    """工具调用次数直方图（agent 评测的汇总项）。"""
    counts: dict[str, int] = {}
    for row in rows:
        for name in getattr(row, "agent_tools", []) or []:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _count_step_sources(rows) -> dict[str, int]:
    """步级来源直方图（agent）。单步行没有该字段，返回空 dict。"""
    counts: dict[str, int] = {}
    for row in rows:
        for source, count in (getattr(row, "agent_step_sources", None) or {}).items():
            counts[source] = counts.get(source, 0) + int(count)
    return counts


def _sentence_source(answer) -> str:
    """句级分析用哪段文本。

    claim 回答只分析「陈述」部分：证据字段是逐字原文，混进来会让覆盖率自己证
    自己（引文必然 100% 命中），指标当场失效。陈述后面补上句柄，这样它和
    普通回答里"句子 + [ID:n]"的结构一致，同一套切句/映射逻辑可以复用。
    """
    claims = getattr(answer, "claims", None)
    if not claims:
        return answer.answer
    lines: list[str] = []
    for claim in claims:
        suffix = f"[ID:{claim.number}]" if claim.number is not None else ""
        lines.append(f"{claim.statement}{suffix}。")
    return "\n".join(lines)


def _check_sentences(
    row: EvalRow,
    answer,
    answer_hits: list,
    *,
    shown_text: dict[str, str],
    chunk_text: dict[str, str],
    gold_spans: list[str],
    strict_spans: list[str],
) -> None:
    """逐句判支撑并写回 ``row``。拒答不算句子（没有句子可判）。"""
    if answer.refused:
        return
    row.gold_span_in_context = (
        any(supports_spans(h.context_chunk.text, gold_spans) for h in answer_hits)
        if gold_spans
        else None
    )
    sentences = split_sentences(_sentence_source(answer))
    row.sentence_total = len(sentences)

    def resolve(form: str, number: int) -> str | None:
        index = number if form == "id" else number - 1
        if 0 <= index < len(answer_hits):
            return answer_hits[index].chunk.chunk_id
        return None

    for sentence in sentences:
        pairs = markers_in(sentence)
        if not pairs:
            continue
        row.sentence_cited += 1
        cited_ids: list[str] = []
        for number, form in pairs:
            chunk_id = resolve(form, number)
            if chunk_id is not None and chunk_id not in cited_ids:
                cited_ids.append(chunk_id)
        cited_texts = [shown_text.get(cid, chunk_text.get(cid, "")) for cid in cited_ids]
        check = check_sentence(sentence, cited_texts)
        row.marker_form_id += check.marker_id
        row.marker_form_legacy += check.marker_legacy
        gold_ok = (
            any(supports_spans(text, gold_spans) for text in cited_texts)
            if gold_spans
            else None
        )
        strict_ok = (
            any(supports_spans(text, strict_spans) for text in cited_texts)
            if strict_spans
            else None
        )
        grounded = (
            sum(1 for text in cited_texts if supports_spans(text, gold_spans))
            if gold_spans
            else 0
        )
        grounded_strict = (
            sum(1 for text in cited_texts if supports_spans(text, strict_spans))
            if strict_spans
            else 0
        )
        if gold_ok:
            row.sentence_supported_any += 1
        if is_supported(grounded, len(cited_texts)):
            row.sentence_supported += 1
        if is_supported(grounded_strict, len(cited_texts)):
            row.sentence_supported_strict += 1
        if check.verbatim:
            row.sentence_self_supported += 1
        row.sentences.append(
            {
                **check.to_dict(),
                "cited_chunk_ids": cited_ids,
                "gold": gold_ok,
                "grounded": grounded,
                "cited_total": len(cited_texts),
                "supported": is_supported(grounded, len(cited_texts)),
                "strict": strict_ok,
            }
        )


def fill_answer_metrics(
    row: EvalRow,
    item: GoldItem,
    answer,
    answer_hits: list,
    *,
    shown_text: dict[str, str],
    chunk_text: dict[str, str],
    span_freq: dict[str, int],
    strict_spans: list[str],
) -> None:
    """把一次回答的所有指标写进 ``row``。

    单步（``run_eval``）与 agent（``run_agent_eval``）**共用这一个函数**。
    M3 的教训是"两处口径漂移"比指标本身更难发现，所以这里不允许各写一份。

    ``answer_hits`` 的语义是"**编号下标 → 命中块**"的映射（``[ID:n]`` 的 n 就是下标），
    agent 必须传入"按证据注册表顺序排列的命中块"，否则句柄会解错、指标全错。
    """
    row.answer = answer.answer
    row.refused = answer.refused
    if answer.refused:
        # 生成侧给了归因就用它（M4 的降级拒答只有生成侧知道原因）；
        # 否则按 M3 的口径推导：空上下文 = 检索没找到；有上下文却拒答 = 判定不充分。
        # M5 修正：预算耗尽/依赖失败比"判定不充分"更具体，必须排在前面，
        # 否则 P68 里"预算把研究截断"会被伪装成"资料不足"（见 attribution 模块头）。
        reason = getattr(answer, "refusal_reason", None)
        degradations = getattr(row, "agent_degradations", None) or []
        if any(str(item).startswith("degraded_dependency") for item in degradations):
            reason = "degraded_dependency"
        elif any(str(item).startswith("budget_exhausted") for item in degradations):
            reason = "budget_exhausted"
        row.refusal_reason = reason or (
            "judged_insufficient" if answer.context_chunk_ids else "no_hit"
        )
    row.refuse_correct = (not answer.refused) if item.answerable else answer.refused
    row.n_citations = len(answer.citations)
    row.invalid_markers = list(answer.invalid_markers)
    row.latency_ms = answer.latency_ms
    row.prompt_tokens = answer.prompt_tokens
    row.completion_tokens = answer.completion_tokens
    row.citations = [
        {
            "marker": c.marker,
            "handle": c.handle,
            "chunk_id": c.chunk_id,
            "rel_path": c.rel_path,
            "page": c.page,
            "section": c.section,
        }
        for c in answer.citations
    ]
    row.warnings.extend(answer.warnings)

    # 引用支撑：被引用的每个 chunk 里是否真能找到金标证据串
    if item.gold_spans and answer.citations:
        row.checked_citations = len(answer.citations)
        row.supported_citations = sum(
            1
            for c in answer.citations
            if supports_spans(
                shown_text.get(c.chunk_id, chunk_text.get(c.chunk_id, "")),
                item.gold_spans,
            )
        )
        row.citation_support = round(
            row.supported_citations / row.checked_citations, 4
        )

        # 严格口径：丢掉"到处都是"的证据串，否则指标等于没测
        row.ambiguous_spans = [
            s for s in item.gold_spans if span_freq.get(s, 0) > STRICT_SPAN_MAX
        ]
        if strict_spans:
            row.checked_strict = len(answer.citations)
            row.supported_strict = sum(
                1
                for c in answer.citations
                if supports_spans(
                    shown_text.get(c.chunk_id, chunk_text.get(c.chunk_id, "")),
                    strict_spans,
                )
            )
            row.citation_support_strict = round(
                row.supported_strict / row.checked_strict, 4
            )

    # ---- M3：句级校验（纯离线，不调用模型）----
    _check_sentences(row, answer, answer_hits, shown_text=shown_text,
                     chunk_text=chunk_text, gold_spans=item.gold_spans,
                     strict_spans=strict_spans)

    # ---- M3：claim 侧打点（claim 变体才有内容）----
    claim_report = getattr(answer, "claim_report", None)
    if claim_report is not None:
        row.claim_total = claim_report.lines
        row.claim_with_evidence = claim_report.with_evidence
        row.claim_verbatim = claim_report.verbatim
        row.claim_unresolved_handles = list(claim_report.unresolved_handle)
        row.claim_unparsed_lines = claim_report.unparsed_lines
    row.claim_fallback = bool(getattr(answer, "claim_fallback", False))


def run_eval(
    items: list[GoldItem],
    retriever: Retriever,
    client: DeepSeekClient,
    *,
    top_k: int = 5,
    cutoff: float = 0.0,
    label: str = "m0",
    with_answers: bool = True,
    verbose: bool = True,
    include_paths: bool = False,
    prompt_variant: str = "strict",
    doc_filter=None,
    gold_name: str = "",
) -> EvalRun:
    run = EvalRun(
        label=label,
        top_k=top_k,
        cutoff=cutoff,
        model=client.model,
        gold_name=gold_name,
        mode=retriever.mode,
        fusion=retriever.fusion if retriever.mode == "hybrid" else "",
        chunk_strategy=getattr(retriever, "chunk_strategy", ""),
        n_chunks=sum(1 for c in retriever.chunks if not c.is_parent),
        prompt_variant=prompt_variant,
        rerank=retriever.reranker is not None,
        rerank_model=getattr(retriever.reranker, "name", "") if retriever.reranker else "",
        rerank_candidates=getattr(retriever, "rerank_candidates", 0),
        doc_filter=doc_filter.describe() if doc_filter is not None else "active",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if run.n_chunks:
        run.avg_chunk_len = round(
            sum(len(c.text) for c in retriever.chunks if not c.is_parent) / run.n_chunks, 1
        )
    #: M5：每个评测报告都带一份分块边界分布，避免"指标只存在于 M1 报告里"。
    from app.eval.chunk_metrics import boundary_stats

    run.chunk_side = boundary_stats(retriever.chunks)
    started = time.perf_counter()

    # rel_path → doc_id，金标里写的是路径
    path_to_doc = {_norm(d.rel_path): d.doc_id for d in retriever.docs_by_id.values()}
    chunk_text = {c.chunk_id: c.text for c in retriever.chunks}
    # 检索单元（不含父块）：分块侧指标按它算，否则父子块的 1800 字父块会把
    # "证据完整率"抬得虚高，四组策略不可比。
    units_by_doc: dict[str, list[object]] = {}
    for chunk in retriever.chunks:
        if not chunk.is_parent:
            units_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    # 证据串的"泛化程度"：出现在几份**内容不同**的文档里（按文档内容指纹计）。
    # 不用 chunk 行数：副本 + 分块重叠会把同一段话重复计数，严格口径会被误剔除。
    all_spans = sorted({s for item in items for s in item.gold_spans})
    doc_fingerprint = document_fingerprints(retriever.chunks)
    chunk_doc = {c.chunk_id: c.doc_id for c in retriever.chunks}
    span_freq = {
        s: len(
            {
                doc_fingerprint.get(chunk_doc.get(cid, ""), cid)
                for cid, text in chunk_text.items()
                if _flat(s) in _flat(text)
            }
        )
        for s in all_spans
    }

    for item in items:
        gold_ids = {path_to_doc[p] for p in (_norm(x) for x in item.gold_rel_paths) if p in path_to_doc}
        missing_gold = [p for p in item.gold_rel_paths if _norm(p) not in path_to_doc]

        search_started = time.perf_counter()
        result = retriever.search(
            item.question, top_k=top_k, relative_cutoff=cutoff, doc_filter=doc_filter
        )
        retrieval_ms = round((time.perf_counter() - search_started) * 1000, 2)
        retrieved_doc_ids = []
        for hit in result.hits:
            retrieved_doc_ids.append(hit.chunk.doc_id)

        row = EvalRow(
            qid=item.qid,
            question=item.question,
            category=item.category,
            answerable=item.answerable,
            retrieval_ms=retrieval_ms,
            top1_score=round(result.hits[0].score, 6) if result.hits else None,
            sources=[hit.source for hit in result.hits],
        )
        if result.rerank_status == "failed":
            run.rerank_failures += 1

        # 分块侧指标：每条证据串是否完整落在金标文档的**某一个检索单元**里。
        # 只在有 gold_spans 的题上算，无答案题不计入分母。
        if item.gold_spans and gold_ids:
            gold_units = [c for doc_id in gold_ids for c in units_by_doc.get(doc_id, [])]
            intact = sum(
                1
                for span in item.gold_spans
                if any(_flat(span) and _flat(span) in _flat(c.text) for c in gold_units)
            )
            row.span_total = len(item.gold_spans)
            row.span_intact = intact
            # 命中的片段本身是否含完整证据串（比"文档命中"更严）
            row.span_hit_at_k = any(
                supports_spans(hit.chunk.text, item.gold_spans) for hit in result.hits
            )

        if gold_ids:
            row.hit_at_k = any(doc_id in gold_ids for doc_id in retrieved_doc_ids)
            found = {doc_id for doc_id in retrieved_doc_ids if doc_id in gold_ids}
            row.recall_at_k = round(len(found) / len(gold_ids), 4)
            for position, doc_id in enumerate(retrieved_doc_ids, start=1):
                if doc_id in gold_ids:
                    row.first_gold_rank = position
                    row.mrr = round(1.0 / position, 4)
                    break
            if row.mrr is None:
                row.mrr = 0.0
        if missing_gold:
            row.warnings.append(f"金标路径未入库：{missing_gold}")

        # 严格证据串：剔除"全库高频"的那些（口径见 STRICT_SPAN_MAX）。
        # 放在循环体前算一次，引用支撑与句级支撑共用同一份，避免两处口径漂移。
        strict_spans_row = [
            s for s in item.gold_spans if span_freq.get(s, 0) <= STRICT_SPAN_MAX
        ]

        row.context = [
            {
                "rank": h.rank,
                "chunk_id": h.chunk.chunk_id,
                "doc_id": h.chunk.doc_id,
                "rel_path": (
                    retriever.docs_by_id[h.chunk.doc_id].rel_path
                    if h.chunk.doc_id in retriever.docs_by_id
                    else ""
                ),
                "page": h.context_chunk.page,
                "score": round(h.score, 4),
                "source": h.source,
                "rank_after_rerank": h.rank,
                "version_extra": h.rank > len(result.hits),
                "parent_chunk_id": h.parent.chunk_id if h.parent else None,
                "preview": h.context_chunk.preview,
                # 全文也存一份：句级校验/人工抽检要回看"模型当时读到的正文"，
                # 只留 80 字 preview 的话，事后无法复核（storage/ 不进仓库）。
                "text": h.context_chunk.text,
            }
            for h in result.answer_hits
        ]
        # 模型实际读到的正文：父子块下是父块，其余是命中块。
        # 引用支撑必须按"读到的文本"判，按子块判会把父块提供的证据算成不支持。
        shown_text = {h.chunk.chunk_id: h.context_chunk.text for h in result.answer_hits}

        try:
            if not with_answers:
                raise _SkipGeneration()
            answer = answer_from_hits(
                item.question, result.answer_hits, retriever.docs_by_id, client,
                versions=retriever.versions, include_paths=include_paths,
                prompt_variant=prompt_variant,
            )
        except _SkipGeneration:
            if verbose:
                print(f"[{item.qid}] {item.category} 命中={row.hit_at_k}（跳过生成）")
            run.rows.append(row)
            continue
        except LLMError as exc:
            row.error = str(exc)
            row.warnings.append("调用失败，回答指标缺失")
            if verbose:
                print(f"[{item.qid}] ERROR {exc}")
            run.rows.append(row)
            continue

        fill_answer_metrics(
            row,
            item,
            answer,
            result.answer_hits,
            shown_text=shown_text,
            chunk_text=chunk_text,
            span_freq=span_freq,
            strict_spans=strict_spans_row,
        )

        if verbose:
            flag = "拒答" if answer.refused else f"引用{row.n_citations}"
            print(f"[{item.qid}] {item.category} 命中={row.hit_at_k} {flag}")

        run.rows.append(row)

    run.duration_s = round(time.perf_counter() - started, 2)
    return run
