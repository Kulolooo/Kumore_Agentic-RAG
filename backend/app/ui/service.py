"""UI 与问答后端之间的唯一接缝。

这里**不复制**任何检索/生成逻辑：agent 走 ``AgentRunner``，单步走
``answer_question``，与 CLI/评测同一套函数。UI 只做三件事：

- 把 ``Answer`` / ``AgentResult`` 展平成可 JSON 落盘的 turn；
- 把引用、claim、轨迹原样带出来，交给渲染层；
- 给 ``AgentRunner`` 挂 ``on_step``，让研究过程边跑边显示。

所有键名都是展示层约定，不参与评测口径。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.agent.loop import AgentConfig, AgentRunner
from app.agent.trace import Step, Trace
from app.config import Settings
from app.generate.answer import answer_question
from app.generate.client import DeepSeekClient
from app.retrieve.retriever import Retriever
from app.ui.format import mask_private

#: 单步模式在轨迹里用的工具名（不是真实工具，只用于展示一行"检索 → 生成"）。
SINGLE_STEP_TOOL = "single_pass"


@dataclass
class UIConfig:
    """一次提问的展示层选项；默认值对齐路线图 §6。"""

    mode: str = "agent"  # agent / single
    top_k: int = 5
    prompt_variant: str = "claim"
    max_steps: int = 10
    max_tokens: int = 120_000
    #: 合成输出上限。M6.5 起 UI 默认 1024：claim 答案实测 completion 约 30–260 tok，
    #: 2048 的余量用不到；上限只防"模型话痨"，不改变正常答案。
    finalize_max_tokens: int = 1024
    #: 合成是否走流式；React/API 显式开启，CLI/评测默认不走。
    stream: bool = True
    include_superseded: bool = False
    max_unparsed_rounds: int = 2
    synthesis_per_step: int = 0
    verified_only: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"agent", "single"}:
            raise ValueError(f"mode 只能是 agent / single，当前 {self.mode!r}")
        if self.prompt_variant not in {"strict", "tight", "claim"}:
            raise ValueError(f"未知 prompt_variant {self.prompt_variant!r}")


def build_retriever(settings: Settings) -> Retriever:
    """从落盘索引构造检索器。加载失败直接抛，由页面显示"先入库"。"""
    return Retriever.from_storage(settings)


def build_client(settings: Settings) -> DeepSeekClient:
    if not settings.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    return DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        timeout_s=settings.request_timeout_s,
        cache_path=(settings.storage_dir / "llm_cache.sqlite") if settings.llm_cache else None,
    )


def run_turn(
    question: str,
    retriever: Retriever,
    client: DeepSeekClient,
    config: UIConfig,
    *,
    on_step: Callable[[Step], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    """跑一轮问答，返回可落盘的 turn。**不抛业务异常**由调用方兜；这里只做投影。"""
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    started = time.monotonic()
    turn_id = f"t{int(time.time() * 1000)}-{uuid.uuid4().hex[:4]}"

    if config.mode == "agent":
        runner = AgentRunner(
            retriever,
            client,
            AgentConfig(
                max_steps=config.max_steps,
                max_tokens_budget=config.max_tokens,
                finalize_max_tokens=config.finalize_max_tokens,
                stream=config.stream,
                top_k=config.top_k,
                prompt_variant=config.prompt_variant,
                include_superseded=config.include_superseded,
                max_unparsed_rounds=config.max_unparsed_rounds,
                synthesis_per_step=config.synthesis_per_step,
                verified_only=config.verified_only,
            ),
            on_step=on_step,
            on_token=on_token,
            cancel_event=cancel_event,
        )
        result = runner.run(question)
        turn = project_agent_turn(question, result, config, turn_id)
    else:
        answer = answer_question(
            question,
            retriever,
            client,
            top_k=config.top_k,
            include_superseded=config.include_superseded,
            prompt_variant=config.prompt_variant,
        )
        turn = project_single_turn(question, answer, config, turn_id)

    turn["elapsed_s"] = round(time.monotonic() - started, 2)
    return turn


# ---------------------------------------------------------------- 投影
def project_agent_turn(
    question: str,
    result: Any,
    config: UIConfig,
    turn_id: str,
) -> dict[str, Any]:
    answer = result.answer
    trace = result.trace
    turn = _base_turn(question, answer, config, turn_id)
    turn["trace"] = trace.to_dict()
    turn["meta"].update(
        {
            "tokens": trace.tokens_used,
            "n_steps": trace.n_steps,
            "n_evidence": trace.n_evidence,
            "short_circuit": bool(getattr(result, "short_circuit", False)),
            "outcome": trace.outcome,
            "degradations": list(trace.degradations),
        }
    )
    return turn


def project_single_turn(
    question: str,
    answer: Any,
    config: UIConfig,
    turn_id: str,
) -> dict[str, Any]:
    trace = Trace(question=question)
    trace.add(
        Step(
            step=1,
            tool=SINGLE_STEP_TOOL,
            ok=True,
            n_hits=len(answer.context_chunk_ids),
            ms=answer.latency_ms,
            tokens=(answer.prompt_tokens or 0) + (answer.completion_tokens or 0) or None,
            decision="refuse" if answer.refused else "finalize",
            detail=f"检索 top-{config.top_k} → 生成" + ("（拒答）" if answer.refused else ""),
        )
    )
    trace.outcome = "refuse" if answer.refused else "answer"
    trace.refusal_reason = answer.refusal_reason
    trace.n_evidence = len(answer.context_chunk_ids)

    turn = _base_turn(question, answer, config, turn_id)
    turn["trace"] = trace.to_dict()
    turn["meta"].update(
        {
            "tokens": (answer.prompt_tokens or 0) + (answer.completion_tokens or 0),
            "n_steps": 1,
            "n_evidence": len(answer.context_chunk_ids),
            "short_circuit": True,
            "outcome": trace.outcome,
            "degradations": [],
        }
    )
    return turn


def _base_turn(question: str, answer: Any, config: UIConfig, turn_id: str) -> dict[str, Any]:
    return {
        "id": turn_id,
        "question": question,
        "mode": config.mode,
        "answer": answer.answer,
        "refused": bool(answer.refused),
        "refusal_reason": answer.refusal_reason,
        "citations": [_citation_view(c) for c in answer.citations],
        "claims": [_claim_view(c) for c in getattr(answer, "claims", [])],
        "claim_fallback": bool(getattr(answer, "claim_fallback", False)),
        "warnings": list(answer.warnings or []),
        "invalid_markers": list(answer.invalid_markers or []),
        "meta": {
            "model": answer.model,
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "latency_ms": answer.latency_ms,
            "top_k": config.top_k,
            "prompt_variant": config.prompt_variant,
            "mode": config.mode,
        },
    }


def _citation_view(citation: Any) -> dict[str, Any]:
    return {
        "marker": int(citation.marker),
        "chunk_id": citation.chunk_id,
        "doc_id": citation.doc_id,
        # 学号/手机号/邮箱在展示层先换掉，避免会话文件与截图把个人信息带出去；
        # 完整正文仍可用 chunk_id 回查本地索引。
        "rel_path": mask_private(citation.rel_path),
        "page": citation.page,
        "page_end": citation.page_end,
        "section": citation.section,
        "handle": citation.handle,
        # 只留 80 字预览：完整正文回查当前索引，避免会话文件备份整份语料。
        "text": citation.text,
    }


def _claim_view(claim: Any) -> dict[str, Any]:
    located = getattr(claim, "located", None)
    return {
        "statement": claim.statement,
        "evidence": claim.evidence,
        "numbers": [int(n) for n in claim.numbers],
        "located": list(located) if located else None,
        "verbatim": bool(getattr(claim, "verbatim", False)),
    }


# ---------------------------------------------------------------- 小工具
def citation_map(turn: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """编号 → 引用。同一编号只留第一条（发布清单保证编号唯一）。"""
    out: dict[int, dict[str, Any]] = {}
    for citation in turn.get("citations") or []:
        out.setdefault(int(citation.get("marker") or 0), citation)
    return out


def chunk_lookup(retriever: Retriever) -> dict[str, Any]:
    """``chunk_id → Chunk``（含父块）。UI 查原文用，不依赖私有字段。"""
    return {chunk.chunk_id: chunk for chunk in retriever.chunks}


def corpus_badge(retriever: Retriever) -> str:
    """侧栏状态：只报数量与检索模式，不报语料路径。"""
    n_docs = len(retriever.docs_by_id)
    n_chunks = len([c for c in retriever.chunks if not c.is_parent])
    return f"{n_docs} 文档 · {n_chunks:,} 片段 · {retriever.mode}"


def ensure_session_dir(storage_dir: Path) -> Path:
    path = Path(storage_dir) / "ui_sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path
