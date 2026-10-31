"""agent 路的评测：把 ``AgentRunner`` 的输出接进同一套指标。

为什么不另写一套指标：M3 的教训是"两处口径漂移"比指标本身更难发现。所以这里只做
两件事——把 agent 的**证据集合**摆成 ``answer_hits`` 那种"编号 = 下标"的形状，
然后调用 ``runner.fill_answer_metrics``（单步与 agent 共用同一个函数）。

与单步评测的三个口径差异（都写进报告，不藏着）：

1. **Hit@k 的分母变了**：单步的 top-5 是"一次检索的 5 条"，agent 的是"整轮进入注册表
   的全部块"。所以 agent 的 Hit@k 不能与单步直接相减——它测的是"循环有没有把证据找齐"，
   不是"一次检索的排序好不好"。单步的 Hit@5 仍是 0.9231 那个数。
2. **检索耗时**是各次工具调用的总和（含 grep/read），不是单次 BM25。
3. **top1_score** 取注册表里第一条的分数（= 第一次检索的第一名），无答案题上仍可比。
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.loop import AgentConfig, AgentRunner
from app.agent.trace import Trace
from app.eval.runner import (
    EvalRow,
    EvalRun,
    STRICT_SPAN_MAX,
    _flat,
    _norm,
    fill_answer_metrics,
    supports_spans,
)
from app.eval.gold import GoldItem
from app.generate.client import DeepSeekClient
from app.retrieve.retriever import Retriever
from app.content_identity import document_fingerprints


def run_agent_eval(
    items: list[GoldItem],
    retriever: Retriever,
    client: DeepSeekClient,
    *,
    config: AgentConfig | None = None,
    label: str = "m4",
    verbose: bool = True,
    doc_filter=None,
    trace_dir=None,
    keep_traces: bool = False,
    gold_name: str = "",
) -> tuple[EvalRun, list[Trace]]:
    """跑一遍 agent 评测。返回 ``(EvalRun, 每题的 Trace)``。"""
    config = config or AgentConfig()
    runner = AgentRunner(retriever, client, config)
    run = EvalRun(
        label=label,
        top_k=config.top_k,
        cutoff=0.0,
        model=client.model,
        gold_name=gold_name,
        mode=retriever.mode,
        fusion=retriever.fusion if retriever.mode == "hybrid" else "",
        chunk_strategy=getattr(retriever, "chunk_strategy", ""),
        n_chunks=sum(1 for c in retriever.chunks if not c.is_parent),
        prompt_variant=config.prompt_variant,
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
    #: M5：分块边界分布与单步路共用同一口径（``chunk_metrics.boundary_stats``）。
    from app.eval.chunk_metrics import boundary_stats

    run.chunk_side = boundary_stats(retriever.chunks)
    started = time.perf_counter()

    path_to_doc = {_norm(d.rel_path): d.doc_id for d in retriever.docs_by_id.values()}
    chunk_text = {c.chunk_id: c.text for c in retriever.chunks}
    units_by_doc: dict[str, list[object]] = {}
    for chunk in retriever.chunks:
        if not chunk.is_parent:
            units_by_doc.setdefault(chunk.doc_id, []).append(chunk)

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

    traces: list[Trace] = []
    for item in items:
        gold_ids = {
            path_to_doc[p] for p in (_norm(x) for x in item.gold_rel_paths) if p in path_to_doc
        }
        result = runner.run(item.question)
        trace = result.trace
        traces.append(trace)
        if trace_dir is not None and keep_traces:
            trace.save(trace_dir / f"{item.qid}.jsonl")

        hits = result.hits
        retrieved_doc_ids = [hit.chunk.doc_id for hit in hits]
        # 证据顺序就是编号顺序：_build_citations 与句级解析都按"编号 = 下标"工作
        row = EvalRow(
            qid=item.qid,
            question=item.question,
            category=item.category,
            answerable=item.answerable,
            retrieval_ms=round(
                sum(step.ms or 0.0 for step in trace.tool_steps), 2
            ),
            top1_score=round(hits[0].score, 6) if hits else None,
            sources=[hit.source for hit in hits],
        )
        row.agent_rounds = max((step.step for step in trace.steps), default=0)
        row.agent_tool_calls = len(trace.tool_steps)
        row.agent_tools = [step.tool for step in trace.tool_steps]
        row.agent_short_circuit = result.short_circuit
        row.agent_degradations = list(trace.degradations)
        #: M5：步级来源（每一步新增块来自哪条腿）与预算耗尽标记。
        step_sources: dict[str, int] = {}
        for step in trace.tool_steps:
            for source in step.sources or []:
                step_sources[source] = step_sources.get(source, 0) + 1
        row.agent_step_sources = step_sources
        row.agent_budget_exhausted = any(
            str(item).startswith("budget_exhausted") for item in trace.degradations
        )
        row.agent_tokens = trace.tokens_used
        row.agent_synthesis_blocks = len(result.synthesis)
        row.agent_published_blocks = len(result.published)

        # ---- 检索侧指标：分母是"整轮进入注册表的块"（口径差异见模块头）----
        if item.gold_spans and gold_ids:
            gold_units = [c for doc_id in gold_ids for c in units_by_doc.get(doc_id, [])]
            row.span_total = len(item.gold_spans)
            row.span_intact = sum(
                1
                for span in item.gold_spans
                if any(_flat(span) and _flat(span) in _flat(c.text) for c in gold_units)
            )
            row.span_hit_at_k = any(
                supports_spans(hit.chunk.text, item.gold_spans) for hit in hits
            )
        if gold_ids:
            row.hit_at_k = any(doc_id in gold_ids for doc_id in retrieved_doc_ids)
            found = {doc_id for doc_id in retrieved_doc_ids if doc_id in gold_ids}
            row.recall_at_k = round(len(found) / len(gold_ids), 4)
            row.first_gold_rank = next(
                (rank for rank, doc_id in enumerate(retrieved_doc_ids, start=1) if doc_id in gold_ids),
                None,
            )
            row.mrr = round(1.0 / row.first_gold_rank, 4) if row.first_gold_rank else 0.0

        row.context = [
            {
                "rank": index,
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "rel_path": (
                    retriever.docs_by_id[hit.chunk.doc_id].rel_path
                    if hit.chunk.doc_id in retriever.docs_by_id
                    else ""
                ),
                "page": hit.context_chunk.page,
                "score": round(hit.score, 4),
                "source": hit.source,
                "text": hit.context_chunk.text,
                "step": next(
                    (
                        entry["step"]
                        for entry in result.evidence
                        if entry["chunk_id"] == hit.chunk.chunk_id
                    ),
                    0,
                ),
            }
            for index, hit in enumerate(hits)
        ]
        shown_text = {hit.chunk.chunk_id: hit.context_chunk.text for hit in hits}

        strict_spans_row = [
            s for s in item.gold_spans if span_freq.get(s, 0) <= STRICT_SPAN_MAX
        ]
        #: **口径要点**：``answer_hits`` 必须是**发布清单**（首引序）——答案里的
        #: ``[ID:n]`` 在 M4 改动 2 之后指的就是它；传注册表全量会让句级解析错位。
        fill_answer_metrics(
            row,
            item,
            result.answer,
            result.published,
            shown_text=shown_text,
            chunk_text=chunk_text,
            span_freq=span_freq,
            strict_spans=strict_spans_row,
        )
        #: ``gold_span_in_context`` 问的是"模型最终看到了什么"，所以按**合成上下文**判，
        #: 而不是按发布清单（发布清单只含被引块，会低估"金标进过上下文"）。
        if item.gold_spans:
            row.gold_span_in_context = any(
                supports_spans(hit.context_chunk.text, item.gold_spans)
                for hit in result.synthesis
            )
        #: ``row.error`` 只放**真的调用失败**（模型/网络），不放"预算耗尽"这类降级：
        #: 否则 markdown 里的"调用失败"会把降级行也算进去，读报告的人会以为接口挂了。
        #: 降级本身在 ``row.agent_degradations`` 里完整保留。
        hard_failures = [
            item
            for item in trace.degradations
            if item.startswith(("finalize_failed", "llm_error"))
        ]
        if hard_failures:
            row.error = "; ".join(hard_failures)

        if verbose:
            flag = "拒答" if result.answer.refused else f"引用{row.n_citations}"
            print(
                f"[{item.qid}] {item.category} 步数={row.agent_rounds} "
                f"工具={row.agent_tool_calls} 证据={len(hits)} "
                f"合成={len(result.synthesis)} 发布={len(result.published)} "
                f"短路={result.short_circuit} {flag}"
            )
        run.rows.append(row)

    run.duration_s = round(time.perf_counter() - started, 2)
    return run, traces
