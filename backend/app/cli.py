"""命令行入口：``python -m app <子命令>``。

M0 用 CLI 而不是 Web，是为了让"能跑起来"这件事不依赖前端。
Web 界面是 P5 的事，而且一旦有了界面，失败就更难被看见。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from app.config import PROJECT_ROOT, Settings
from app.kb import (
    create_kb,
    index_stats,
    list_kbs,
    rebuild,
    resolve_kb,
    resolve_settings,
    set_active,
    set_retrieval_mode,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx 每个请求都打一条 INFO，会把真正有用的进度日志淹掉
    logging.getLogger("httpx").setLevel(logging.DEBUG if verbose else logging.WARNING)


def _with_overrides(args: argparse.Namespace) -> Settings:
    # 所有子命令默认解析"当前知识库"；--kb 显式指定。这样 eval 与 UI 不会读不同的索引。
    base, _ = resolve_settings(getattr(args, "kb", None))
    overrides: dict[str, object] = {}
    if getattr(args, "corpus", None):
        overrides["corpus_root"] = Path(args.corpus)
    if getattr(args, "chunk_size", None):
        overrides["chunk_size"] = args.chunk_size
    if getattr(args, "chunk_strategy", None):
        overrides["chunk_strategy"] = args.chunk_strategy
    if getattr(args, "chunk_tokens", None):
        overrides["chunk_token_size"] = args.chunk_tokens
    if getattr(args, "overlap", None) is not None:
        overrides["chunk_overlap"] = args.overlap
    if getattr(args, "top_k", None):
        overrides["top_k"] = args.top_k
    if getattr(args, "cutoff", None) is not None:
        overrides["retrieve_cutoff"] = args.cutoff
    if getattr(args, "strategy", None):
        overrides["parse_strategy"] = args.strategy
    if getattr(args, "parser", None):
        overrides["parser_backend"] = args.parser
    if getattr(args, "mode", None):
        overrides["retrieval_mode"] = args.mode
    if getattr(args, "fusion", None):
        overrides["fusion"] = args.fusion
    if getattr(args, "embed_model", None):
        overrides["embedding_model"] = args.embed_model
    if getattr(args, "embed_device", None):
        overrides["embedding_device"] = args.embed_device
    if getattr(args, "embed_batch", None):
        overrides["embedding_batch_size"] = args.embed_batch
    if getattr(args, "prompt_variant", None):
        overrides["prompt_variant"] = args.prompt_variant
    # 重排开关用 BooleanOptionalAction：默认 None（沿用 .env），显式 --rerank/--no-rerank 才覆盖。
    if getattr(args, "rerank", None) is not None:
        overrides["rerank"] = args.rerank
    if getattr(args, "rerank_candidates", None):
        overrides["rerank_candidates"] = args.rerank_candidates
    if getattr(args, "rerank_model", None):
        overrides["rerank_model"] = args.rerank_model
    result = dataclasses.replace(base, **overrides) if overrides else base
    result.validate()
    return result


def _doc_filter(args: argparse.Namespace):
    """把 CLI 的过滤参数变成 DocFilter；一条都没给时返回 None（零成本默认路径）。"""
    from app.retrieve.filters import from_args

    return from_args(
        ext=getattr(args, "ext", None),
        path_prefix=getattr(args, "path_prefix", None),
        since=getattr(args, "since", None),
        until=getattr(args, "until", None),
        include_superseded=getattr(args, "include_superseded", False),
    )


# ---------------------------------------------------------------- ingest
def cmd_ingest(args: argparse.Namespace) -> int:
    raw = Settings.from_env()
    kb = resolve_kb(raw.storage_dir, getattr(args, "kb", None), raw)
    cfg = _with_overrides(args)
    if not kb.corpus_root.is_dir():
        print(f"语料目录不存在：{kb.corpus_root}", file=sys.stderr)
        print("请在知识库里上传文档，或把 kb.json 的 corpus_root 指向已有目录。", file=sys.stderr)
        return 2

    # 清掉代际路径，保留 CLI 的分块/解析覆盖；rebuild 自己决定 staging 和 current。
    base = dataclasses.replace(cfg, storage_dir=raw.storage_dir, corpus_root=kb.corpus_root)
    try:
        stats, generation = rebuild(
            kb.id,
            base=base,
            limit=args.limit,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"重建失败，旧索引未改动：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    if args.limit:
        print(
            f"\n--limit {args.limit}：试跑代际 {generation} 未激活，生产索引未改动。",
            file=sys.stderr,
        )
    else:
        print(f"\n知识库 {kb.name}（{kb.id}）已切到代际 {generation.name}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- embed
def cmd_embed(args: argparse.Namespace) -> int:
    """文本塔 bge-m3 + 视觉塔 bge-visualized，两个独立索引。"""
    from app.embed.build import build_vectors
    from app.embed.local import EmbedderUnavailable

    cfg = _with_overrides(args)
    tower = getattr(args, "tower", "both")
    try:
        if tower == "visual":
            stats = build_vectors(cfg, tower="visual")
        else:
            stats = build_vectors(
                cfg,
                limit=args.limit,
                text_batch=args.embed_batch or 32,
                tower="text",
            )
            if tower == "both" and not args.limit:
                # 文本塔和视觉塔必须分进程：两套权重同时驻留会 OOM/access violation。
                import subprocess

                cmd = [
                    sys.executable, "-m", "app", "embed",
                    "--tower", "visual",
                ]
                if getattr(args, "kb", None):
                    cmd.extend(["--kb", args.kb])
                code = subprocess.call(cmd)
                if code != 0:
                    print("视觉塔构建失败；文本索引已保留。", file=sys.stderr)
                    return code
    except EmbedderUnavailable as exc:
        print(f"embedding 不可用：{exc}", file=sys.stderr)
        return 3
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.limit:
        print("试跑仅构建文本向量；视觉索引未改动。", file=sys.stderr)
    return 0


def _load_retriever(cfg: Settings):
    """构造检索器；索引对不上就带着可执行提示退出（exit 3）。"""
    from app.ingest.verify import IndexInconsistent
    from app.retrieve.retriever import Retriever

    try:
        return Retriever.from_storage(cfg)
    except IndexInconsistent as exc:
        print(f"索引不一致，已停止：\n{exc}", file=sys.stderr)
        raise SystemExit(3) from exc


# ---------------------------------------------------------------- ask
def _print_answer(answer, show_context: bool) -> None:
    print()
    print("回答：", answer.answer)
    if answer.citations:
        print()
        print("引用：")
        for c in answer.citations:
            page = f" 第 {c.page} 页" if c.page is not None else ""
            section = f" · {c.section}" if c.section else ""
            handle = c.handle or f"[{c.marker}]"
            print(f"  {handle} {c.rel_path}{page}{section}")
            print(f"       {c.text}")
    extra = [*answer.warnings]
    if answer.invalid_markers:
        extra.append(f"越界引用编号：{answer.invalid_markers}")
    # claim 形态：把"证据能不能逐字核"直接摆给用户看。
    # 没有金标集的语料上，句级主指标不可用，这条汇总就是唯一可读的质量信号。
    report = getattr(answer, "claim_report", None)
    if report is not None:
        print()
        print(
            f"陈述校验：{report.lines} 条｜证据逐字可核 {report.verbatim}"
            f"｜未定位 {report.lines - report.verbatim}"
            f"｜缺引文 {report.lines - report.with_evidence}"
            + (f"｜逐字率 {report.verbatim_rate}" if report.verbatim_rate is not None else "")
        )
    elif getattr(answer, "claim_fallback", False):
        print()
        print("陈述校验：模型未按 claim 格式输出，已按普通回答交付")
    if extra:
        print()
        for note in extra:
            print(f"警告：{note}")
    if show_context:
        print()
        print(f"检索上下文（{len(answer.context_chunk_ids)} 条）：")
        for chunk_id in answer.context_chunk_ids:
            print(f"  - {chunk_id}")
    if answer.latency_ms is not None:
        print()
        print(
            f"[{answer.model}] {answer.latency_ms:.0f} ms　"
            f"入 {answer.prompt_tokens} / 出 {answer.completion_tokens} tokens"
        )


def cmd_ask(args: argparse.Namespace) -> int:
    from app.generate.answer import answer_question
    from app.generate.client import DeepSeekClient, LLMError

    cfg = _with_overrides(args)
    retriever = _load_retriever(cfg)
    if not retriever.chunks:
        print("索引为空，请先运行：python -m app ingest", file=sys.stderr)
        return 2

    question = " ".join(args.question).strip()
    if not question:
        print("问题不能为空", file=sys.stderr)
        return 2

    if not cfg.deepseek_api_key:
        print("未配置 DEEPSEEK_API_KEY，只能看检索结果：", file=sys.stderr)
        print("  python -m app search \"<问题>\"", file=sys.stderr)
        return 2

    client = DeepSeekClient(
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        model=cfg.deepseek_model,
        timeout_s=cfg.request_timeout_s,
        cache_path=(cfg.storage_dir / "llm_cache.sqlite") if cfg.llm_cache else None,
    )
    try:
        answer = answer_question(
            question,
            retriever,
            client,
            top_k=cfg.top_k,
            cutoff=cfg.retrieve_cutoff,
            include_superseded=args.include_superseded,
            include_paths=cfg.send_source_paths,
            prompt_variant=cfg.prompt_variant,
            doc_filter=_doc_filter(args),
        )
    except LLMError as exc:
        print(f"模型调用失败：{exc}", file=sys.stderr)
        return 3

    if args.json:
        report = getattr(answer, "claim_report", None)
        print(
            json.dumps(
                {
                    "question": answer.question,
                    "answer": answer.answer,
                    "refused": answer.refused,
                    "citations": [dataclasses.asdict(c) for c in answer.citations],
                    "invalid_markers": answer.invalid_markers,
                    "warnings": answer.warnings,
                    "claims": [
                        {
                            "statement": claim.statement,
                            "evidence": claim.evidence,
                            "handles": list(claim.numbers),
                            "verbatim": claim.verbatim,
                        }
                        for claim in getattr(answer, "claims", [])
                    ],
                    "claim_check": (
                        None
                        if report is None
                        else {
                            "lines": report.lines,
                            "with_evidence": report.with_evidence,
                            "verbatim": report.verbatim,
                            "verbatim_rate": report.verbatim_rate,
                            "fallback": getattr(answer, "claim_fallback", False),
                        }
                    ),
                    "latency_ms": answer.latency_ms,
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "model": answer.model,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_answer(answer, args.show_context)
    return 0


# ---------------------------------------------------------------- search
def cmd_search(args: argparse.Namespace) -> int:
    cfg = _with_overrides(args)
    retriever = _load_retriever(cfg)
    if not retriever.chunks:
        print("索引为空，请先运行：python -m app ingest", file=sys.stderr)
        return 2

    query = " ".join(args.query).strip()
    result = retriever.search(
        query,
        top_k=cfg.top_k,
        include_superseded=args.include_superseded,
        relative_cutoff=cfg.retrieve_cutoff,
        doc_filter=_doc_filter(args),
    )
    print(
        f"查询：{query}　模式：{result.mode}"
        + (f"（{result.fusion} 融合）" if result.fusion else "")
    )
    print(
        f"命中 {len(result.hits)} 条（过滤：{result.doc_filter}；"
        f"同族限流跳过 {result.dropped_family} 条，"
        f"内容重复折叠 {result.dropped_duplicate} 条）"
    )
    if result.rerank_status == "ok":
        print(
            f"重排：{getattr(retriever.reranker, 'name', '?')}　"
            f"{result.rerank_candidates} 条　{result.rerank_ms} ms"
        )
    elif result.rerank_status == "failed":
        print(f"重排失败（已回退检索原序）：{result.rerank_error}", file=sys.stderr)
    for hit in result.answer_hits:
        doc = retriever.docs_by_id.get(hit.chunk.doc_id)
        page = f" 第 {hit.chunk.page} 页" if hit.chunk.page is not None else ""
        tag = "  [版本补充]" if hit.rank > len(result.hits) else ""
        info = retriever.versions.get(hit.chunk.doc_id)
        version = f" 版本：{info.label}" if info and info.is_multi_version else ""
        source = f" 来源：{hit.source}" if hit.source else ""
        print()
        print(
            f"[{hit.rank}] score={hit.score:.4f}  {doc.rel_path if doc else '?'}"
            f"{page}{version}{source}{tag}"
        )
        print(f"    {hit.chunk.preview}")
    return 0


# ---------------------------------------------------------------- agent
def cmd_agent(args: argparse.Namespace) -> int:
    """M4：走 agent 循环回答问题，并把轨迹落成 JSONL。

    退出码约定（验收项）：**依赖失败也返回 0**——降级答案已经交出去了，
    用非零退出码让脚本把它当"程序崩了"是误报。真正的失败标记在轨迹的
    ``degradations`` 字段里。
    """
    from app.agent.loop import AgentConfig, AgentRunner
    from app.generate.client import DeepSeekClient

    cfg = _with_overrides(args)
    retriever = _load_retriever(cfg)
    if not retriever.chunks:
        print("索引为空，请先运行：python -m app ingest", file=sys.stderr)
        return 2

    question = " ".join(args.question).strip()
    if not question:
        print("问题不能为空", file=sys.stderr)
        return 2
    if not cfg.deepseek_api_key:
        print("未配置 DEEPSEEK_API_KEY：agent 需要模型才能规划检索", file=sys.stderr)
        return 2

    client = DeepSeekClient(
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        model=cfg.deepseek_model,
        timeout_s=cfg.request_timeout_s,
        cache_path=(cfg.storage_dir / "llm_cache.sqlite") if cfg.llm_cache else None,
    )
    agent_config = AgentConfig(
        max_steps=args.max_steps,
        max_tokens_budget=args.max_tokens,
        top_k=cfg.top_k,
        prompt_variant=cfg.prompt_variant,
        include_paths=cfg.send_source_paths,
        include_superseded=args.include_superseded,
        verified_only=args.verified_only,
    )
    runner = AgentRunner(retriever, client, agent_config)
    result = runner.run(question)

    trace_path = None
    if args.trace:
        trace_path = Path(args.trace)
        result.trace.save(trace_path)

    if args.json:
        print(
            json.dumps(
                {
                    "question": question,
                    "answer": result.answer.answer,
                    "refused": result.answer.refused,
                    "refusal_reason": result.answer.refusal_reason,
                    "short_circuit": result.short_circuit,
                    "citations": [dataclasses.asdict(c) for c in result.answer.citations],
                    "warnings": result.answer.warnings,
                    "trace": result.trace.to_dict(),
                    "evidence": result.evidence,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("=" * 24 + " 研究过程 " + "=" * 24)
        print(result.trace.to_markdown())
        print()
        print("=" * 24 + " 最终答案 " + "=" * 24)
        print(result.answer.answer)
        if result.answer.citations:
            print()
            for citation in result.answer.citations:
                page = f" 第 {citation.page} 页" if citation.page else ""
                print(f"  {citation.handle} {citation.rel_path}{page}")
    if trace_path is not None:
        print(f"\n轨迹已写入：{trace_path}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- eval
def cmd_eval(args: argparse.Namespace) -> int:
    from app.eval.gold import load_gold
    from app.eval.runner import run_eval
    from app.generate.client import DeepSeekClient

    cfg = _with_overrides(args)
    retriever = _load_retriever(cfg)
    if not retriever.chunks:
        print("索引为空，请先运行：python -m app ingest", file=sys.stderr)
        return 2

    gold_path = cfg.resolve_gold(args.gold)
    if not gold_path.is_file():
        # 2026-10-30：旧金标集（10/45/5 题）已删除——它们是对旧课程语料写的，
        # 换到通用语料后不可用，见手册 P88。这里给可执行的下一步，而不是抛栈。
        print(
            f"没有金标集：{gold_path} 不存在。\n"
            "用 --gold 指定一份 .jsonl，或把金标放到 data/gold/ 下"
            "（不指定时取题量最多的一份）。字段说明见手册 P88。",
            file=sys.stderr,
        )
        return 2
    items = load_gold(gold_path)
    print(f"金标集：{gold_path.name}（{len(items)} 题）", file=sys.stderr)
    if len(items) < 20:
        print(
            f"⚠ 只有 {len(items)} 题：这份报告只能当冒烟测试，不能和 20 题以上的"
            "对照实验并排下结论（M2-0 的 C-1）。",
            file=sys.stderr,
        )

    if not args.offline and not cfg.deepseek_api_key:
        print("未配置 DEEPSEEK_API_KEY：本次只评检索，跳过生成（等同 --offline）", file=sys.stderr)
        args.offline = True

    client = DeepSeekClient(
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        model=cfg.deepseek_model,
        timeout_s=cfg.request_timeout_s,
        cache_path=(cfg.storage_dir / "llm_cache.sqlite") if cfg.llm_cache else None,
    )
    def run_once(run_label: str):
        """跑一遍评测；--repeat 时同一个函数被调用 N 次，口径完全一致。"""
        if args.agent:
            # M4：agent 路。指标与单步共用 fill_answer_metrics，但 Hit@k 的分母不同
            # （整轮进入注册表的块），报告里必须分开读——见 agent_runner 模块头。
            from app.agent.loop import AgentConfig
            from app.eval.agent_runner import run_agent_eval

            trace_root = cfg.runs_dir / f"{run_label}_traces"
            agent_run, traces = run_agent_eval(
                items,
                retriever,
                client,
                config=AgentConfig(
                    max_steps=args.max_steps,
                    max_tokens_budget=args.max_tokens,
                    top_k=cfg.top_k,
                    prompt_variant=cfg.prompt_variant,
                    include_paths=cfg.send_source_paths,
                    verified_only=args.verified_only,
                ),
                label=run_label,
                verbose=not args.quiet,
                doc_filter=_doc_filter(args),
                trace_dir=trace_root,
                keep_traces=not args.no_traces,
                gold_name=gold_path.name,
            )
            if traces and not args.no_traces:
                print(f"轨迹：{len(traces)} 份写入 {trace_root}", file=sys.stderr)
            return agent_run
        return run_eval(
            items,
            retriever,
            client,
            top_k=cfg.top_k,
            cutoff=cfg.retrieve_cutoff,
            label=run_label,
            with_answers=not args.offline,
            verbose=not args.quiet,
            include_paths=cfg.send_source_paths,
            prompt_variant=cfg.prompt_variant,
            doc_filter=_doc_filter(args),
            gold_name=gold_path.name,
        )

    repeat = max(1, int(getattr(args, "repeat", 1) or 1))
    if repeat == 1:
        run = run_once(args.label)
        json_path, md_path = run.save(cfg.runs_dir)
        if args.json:
            print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
        else:
            print()
            print(run.to_markdown())
        print(f"\n报告已写入：\n  {json_path}\n  {md_path}")
        return 0

    # ---- 多次取样：每次单独存盘，再合成"中位数 + 极差" ----
    import time

    from app.eval.aggregate import aggregate_runs, render as render_aggregate

    runs = []
    for index in range(repeat):
        run = run_once(f"{args.label}_r{index + 1}")
        run.save(cfg.runs_dir)
        runs.append(run)
        summary = run.summary()
        answer_rate = summary.get("answer_rate_on_answerable")
        print(
            f"[{index + 1}/{repeat}] Hit@5={summary.get('hit_at_k')}"
            f"　可答={answer_rate if answer_rate is not None else '—'}（离线模式不评生成）"
            if args.offline
            else f"[{index + 1}/{repeat}] Hit@5={summary.get('hit_at_k')}　可答={answer_rate}",
            file=sys.stderr,
        )

    result = aggregate_runs(
        [
            {"summary": run.summary(), "rows": [row.to_dict() for row in run.rows]}
            for run in runs
        ],
        label=args.label,
        gold=gold_path.name,
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = cfg.runs_dir / f"eval_{args.label}_repeat_{stamp}.json"
    md_path = cfg.runs_dir / f"eval_{args.label}_repeat_{stamp}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render_aggregate(result)
    md_path.write_text(text, encoding="utf-8")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print()
        print(text)
    print(f"\n多次取样报告已写入：\n  {json_path}\n  {md_path}")
    return 0


# ---------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> int:
    """按固定七节模板渲染一份实验报告（先校验，不合格直接报错）。"""
    from app.eval.report import load_spec, render_spec

    spec = load_spec(args.spec)
    text = render_spec(args.spec)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"报告已写入：{out}")
    else:
        print(text)
    return 0


# ---------------------------------------------------------------- kb
def cmd_kb(args: argparse.Namespace) -> int:
    """知识库管理：list / create / use / show / mode。"""
    raw = Settings.from_env()
    action = args.action
    if action == "list":
        resolve_kb(raw.storage_dir, None, raw)  # 首次调用时迁移旧单库/建默认库
        rows = list_kbs(raw.storage_dir)
        print(
            json.dumps(
                [kb.to_dict() for kb in rows],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if action == "create":
        if not args.name:
            print("用法：python -m app kb create <名称> [--corpus 目录]", file=sys.stderr)
            return 2
        try:
            kb = create_kb(
                args.name,
                storage_dir=raw.storage_dir,
                base=raw,
                corpus_root=args.corpus,
                make_active=True,
            )
        except Exception as exc:
            print(f"创建失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        print(json.dumps(kb.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if action == "use":
        if not args.name:
            print("用法：python -m app kb use <id>", file=sys.stderr)
            return 2
        kb = set_active(raw.storage_dir, args.name)
        print(f"当前知识库：{kb.name}（{kb.id}）")
        return 0
    if action == "mode":
        if not args.name or not args.value:
            print("用法：python -m app kb mode <id> <bm25|vector|hybrid|default>", file=sys.stderr)
            return 2
        mode = None if args.value == "default" else args.value
        kb = set_retrieval_mode(raw.storage_dir, args.name, mode)
        print(f"{kb.name}（{kb.id}）检索模式：{kb.retrieval_mode or '全局默认'}")
        return 0

    # show
    kb = resolve_kb(raw.storage_dir, args.name, raw)
    print(
        json.dumps(
            {**kb.to_dict(), "index": index_stats(kb, raw)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------- ui
def cmd_ui(args: argparse.Namespace) -> int:
    """M6：启动 Streamlit 界面。

    ``streamlit run`` 自己会起一个子进程常驻，所以这里用 ``subprocess.call``
    把端口/headless 原样传过去；Ctrl+C 的退出码由 streamlit 决定。
    """
    import subprocess

    script = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.headless else "false",
    ]
    print("界面启动中：http://127.0.0.1:%d" % args.port)
    try:
        return int(subprocess.call(cmd))
    except KeyboardInterrupt:
        return 130


# ---------------------------------------------------------------- web (M6.5)
def cmd_web(args: argparse.Namespace) -> int:
    """启动 FastAPI（React 前端的后端）。"""
    import uvicorn

    uvicorn.run(
        "app.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


# ---------------------------------------------------------------- stats
def cmd_stats(args: argparse.Namespace) -> int:
    from app.ingest.store import load_index, read_meta
    from app.ingest.verify import check
    from app.retrieve.retriever import Retriever

    cfg = _with_overrides(args)
    docs, chunks = load_index(cfg)

    vector_index = None
    if cfg.retrieval_mode != "bm25" and cfg.vector_index_path.is_file():
        from app.embed.store import VectorIndex

        try:
            vector_index = VectorIndex.load(cfg.vector_index_path)
        except Exception as exc:  # 读不出来也要把统计打完，问题放进一致性检查里报
            print(f"向量索引读取失败：{exc}", file=sys.stderr)

    retriever = Retriever(
        chunks,
        {d.doc_id: d for d in docs},
        mode=cfg.retrieval_mode,
        vector_index=vector_index,
    )
    problems = check(cfg, docs, chunks, vector_index)
    meta = read_meta(cfg)

    by_status: dict[str, int] = {}
    for doc in docs:
        by_status[doc.status] = by_status.get(doc.status, 0) + 1
    indexed = [d for d in docs if d.status == "active" and d.parse_ok and d.n_chunks > 0]
    skipped_limit = [
        d for d in docs if str(d.parse_error or "").startswith("未解析：--limit")
    ]
    failed = [d for d in docs if not d.parse_ok and d not in skipped_limit]

    print(f"语料目录：{cfg.corpus_root}")
    print(f"分块策略：{cfg.chunk_strategy}")
    print(f"索引统计：{json.dumps(retriever.stats, ensure_ascii=False)}")
    print(
        f"文档口径：候选 {len(docs)}｜active {by_status.get('active', 0)}｜"
        f"已索引 {len(indexed)}｜未尝试解析 {len(skipped_limit)}"
    )
    print(f"文档状态：{by_status}")
    print(f"解析失败：{len(failed)}")
    for doc in failed[:10]:
        print(f"  - {doc.rel_path}：{doc.parse_error}")
    if skipped_limit:
        print(f"未尝试解析（--limit 试跑）：{len(skipped_limit)} 份，不计入失败")

    if problems:
        print("一致性：✗")
        for problem in problems:
            print(f"  - {problem}")
        return 3

    n_vec = len(vector_index.chunk_ids) if vector_index is not None else 0
    n_searchable = sum(1 for c in chunks if not c.is_parent)
    print(
        f"一致性：✓ docs/chunks/vectors = {len(indexed)}/{n_searchable}/{n_vec}"
        + (f"（另有父块 {len(chunks) - n_searchable}）" if len(chunks) != n_searchable else "")
    )

    if not getattr(args, "no_scan", False):
        from app.ingest.verify import corpus_drift

        drift = corpus_drift(cfg, docs)
        total = len(drift["added"]) + len(drift["changed"]) + len(drift["removed"])
        if total:
            print(
                f"语料新鲜度：✗ 索引落后（新增 {len(drift['added'])} / "
                f"修改 {len(drift['changed'])} / 删除 {len(drift['removed'])}）"
            )
            for rel in (*drift["added"], *drift["changed"], *drift["removed"])[:5]:
                print(f"  - {rel}")
            print("  → python -m app ingest（会重建索引；向量索引随后需 --rebuild 检查）")
        else:
            print("语料新鲜度：✓ 与索引一致")

    for key in ("built_at", "counts", "config"):
        if key in meta:
            print(f"{key}: {meta[key]}")
    return 0


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="库默 Kumore：本地可运行、可评测的 agentic RAG（M0 基线）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--kb", help="知识库 id（默认 storage/kbs/active.json）")
        p.add_argument("--corpus", help="语料根目录（覆盖 KUMORE_CORPUS_ROOT）")
        p.add_argument("--top-k", type=int, help="检索返回条数")
        p.add_argument("--cutoff", type=float, help="检索分数相对截断（0~1）")
        p.add_argument("--chunk-size", type=int, help="分块窗口字符数（fixed 策略）")
        p.add_argument("--overlap", type=int, help="分块重叠字符数（fixed 策略）")
        p.add_argument(
            "--chunk-strategy",
            choices=["fixed", "token", "section", "parent_child"],
            help="分块策略（默认 fixed，见 docs/路线图.md M1）",
        )
        p.add_argument("--chunk-tokens", type=int, help="token 策略的单块 budget")
        p.add_argument("--strategy", help="解析策略（fast / hi_res）")
        p.add_argument("--parser", help="解析后端（native / unstructured）")
        p.add_argument("--mode", choices=["bm25", "vector", "hybrid"], help="检索模式")
        p.add_argument("--fusion", choices=["rrf", "weighted"], help="混合检索的融合方式")
        p.add_argument(
            "--prompt-variant",
            choices=["strict", "tight", "claim"],
            help="拒答规则：strict（默认）/ tight（只在本就找不到相关句时拒答，实验用）/ "
            "claim（输出 {陈述, 逐字证据, 句柄} 行，M3 实验）",
        )
        p.add_argument(
            "--rerank",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="启用本地重排（默认关闭；--no-rerank 可覆盖 .env 里的开关）",
        )
        p.add_argument("--rerank-candidates", type=int, help="只重排融合后的前 N 条")
        p.add_argument("--rerank-model", help="重排模型名（默认 BAAI/bge-reranker-base）")
        # 元数据过滤：条件在检索打分前生效，不是取回 top-k 再筛。
        p.add_argument("--ext", help="只在这些扩展名里检索（逗号分隔，如 .docx,.pdf）")
        p.add_argument("--path-prefix", help="只在这些相对路径前缀里检索（逗号分隔）")
        p.add_argument("--since", help="只检索 mtime >= 该日期的文档（YYYY-MM-DD）")
        p.add_argument("--until", help="只检索 mtime <= 该日期的文档（YYYY-MM-DD）")

    p_ingest = sub.add_parser("ingest", help="扫描语料并重建索引")
    add_common(p_ingest)
    p_ingest.add_argument("--limit", type=int, help="只解析前 N 份文档（试跑用）")
    p_ingest.add_argument("--quiet", action="store_true", help="不打印进度日志")
    p_ingest.set_defaults(func=cmd_ingest)

    p_embed = sub.add_parser("embed", help="用本地模型构建向量索引（不出网）")
    add_common(p_embed)
    p_embed.add_argument("--rebuild", action="store_true", help="已有索引时强制重建")
    p_embed.add_argument("--limit", type=int, help="只编码前 N 个 chunk（试跑用）")
    p_embed.add_argument("--embed-model", dest="embed_model", help="embedding 模型名")
    p_embed.add_argument("--embed-device", dest="embed_device", choices=["cpu", "cuda"], help="推理设备")
    p_embed.add_argument("--embed-batch", dest="embed_batch", type=int, help="批大小")
    p_embed.add_argument(
        "--tower", choices=["text", "visual", "both"], default="both",
        help="只建文本/视觉塔，或两者分进程建（默认）",
    )
    p_embed.add_argument("--quiet", action="store_true", help="不显示进度条")
    p_embed.set_defaults(func=cmd_embed)

    p_ask = sub.add_parser("ask", help="提问并生成带引用的回答")
    add_common(p_ask)
    p_ask.add_argument("question", nargs="+", help="问题")
    p_ask.add_argument("--show-context", action="store_true", help="打印上下文 chunk_id")
    p_ask.add_argument("--include-superseded", action="store_true", help="包含旧版本文档")
    p_ask.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_ask.set_defaults(func=cmd_ask)

    p_agent = sub.add_parser("agent", help="M4：agentic 检索（多步 + 工具 + 轨迹）")
    add_common(p_agent)
    p_agent.add_argument("question", nargs="+", help="问题")
    p_agent.add_argument("--include-superseded", action="store_true", help="包含旧版本文档")
    p_agent.add_argument("--json", action="store_true", help="以 JSON 输出（含完整轨迹）")
    p_agent.add_argument("--trace", help="把轨迹写成 JSONL 的路径")
    p_agent.add_argument("--max-steps", type=int, default=10, help="最多几轮模型调用（默认 10）")
    p_agent.add_argument(
        "--max-tokens", type=int, default=120_000, help="按 turn 结算的 token 预算（默认 120k）"
    )
    p_agent.add_argument(
        "--verified-only",
        action="store_true",
        help="P69 实验：短路后合成只复述已核实 claim，不新增 claim/句柄",
    )
    p_agent.set_defaults(func=cmd_agent)

    p_search = sub.add_parser("search", help="只检索，不生成（无需 API key）")
    add_common(p_search)
    p_search.add_argument("query", nargs="+", help="查询词")
    p_search.add_argument("--include-superseded", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="跑金标集并生成报告")
    add_common(p_eval)
    p_eval.add_argument("--gold", help="金标集路径")
    p_eval.add_argument("--label", default="m0", help="报告标签")
    p_eval.add_argument("--offline", action="store_true", help="只评检索，不调用模型")
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("--quiet", action="store_true")
    p_eval.add_argument("--agent", action="store_true", help="M4：走 agent 循环而不是单步")
    p_eval.add_argument("--max-steps", type=int, default=10, help="agent 最多几轮模型调用")
    p_eval.add_argument(
        "--max-tokens", type=int, default=120_000, help="agent 每题的 token 预算"
    )
    p_eval.add_argument(
        "--verified-only",
        action="store_true",
        help="P69 实验：短路后合成只复述已核实 claim，不新增 claim/句柄",
    )
    p_eval.add_argument("--no-traces", action="store_true", help="不落每题轨迹 JSONL")
    p_eval.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="同一配置跑 N 次并合成中位数 + 极差（模型侧有随机性，见 P24）",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_stats = sub.add_parser("stats", help="查看索引状态与解析失败清单")
    add_common(p_stats)
    p_stats.add_argument("--no-scan", action="store_true", help="跳过语料新鲜度扫描（省 1~2 秒）")
    p_stats.set_defaults(func=cmd_stats)

    p_kb = sub.add_parser("kb", help="知识库管理：list / create / use / show / mode")
    p_kb.add_argument("action", choices=["list", "create", "use", "show", "mode"])
    p_kb.add_argument("name", nargs="?", help="create 的名称 / use/show/mode 的 id")
    p_kb.add_argument("value", nargs="?", help="mode 的 bm25/vector/hybrid/default")
    p_kb.add_argument("--corpus", help="create 时绑定已有语料目录；省略则建受管目录")
    p_kb.set_defaults(func=cmd_kb)

    p_ui = sub.add_parser("ui", help="M6：启动 Streamlit 界面（黑绿 / Codex 骨架）")
    p_ui.add_argument("--port", type=int, default=8501, help="端口（默认 8501）")
    p_ui.add_argument("--headless", action="store_true", help="不自动打开浏览器")
    p_ui.set_defaults(func=cmd_ui)

    p_web = sub.add_parser("web", help="M6.5：启动 FastAPI + React 前端")
    p_web.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    p_web.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    p_web.add_argument("--reload", action="store_true", help="开发时热重载")
    p_web.set_defaults(func=cmd_web)

    p_report = sub.add_parser(
        "report", help="M5：按固定七节模板渲染实验报告（缺分母/缺节直接报错）"
    )
    p_report.add_argument("--spec", required=True, help="报告 JSON spec")
    p_report.add_argument("--out", help="输出 Markdown 路径（默认 stdout）")
    p_report.set_defaults(func=cmd_report)

    return parser


def _force_utf8_console() -> None:
    """控制台默认 GBK 时，打印 ✓ / ✗ 或中文会抛 UnicodeEncodeError（P04 复发过一次）。

    只影响显示，不改文件写入路径。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
