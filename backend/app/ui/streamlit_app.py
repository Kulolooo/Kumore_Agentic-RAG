"""M6 入口：``streamlit run app/ui/streamlit_app.py``（或 ``python -m app ui``）。

骨架口径（Codex 的三段式）：
- 左：会话栏（新对话 / 历史 / 状态 / 设置）
- 中：对话流（用户消息、可折叠研究过程、最终答案、引用）
- 下：固定输入框（Agent / 单步切换）
右侧证据抽屉是 Codex diff 面板的 RAG 对应物：点 ``[ID:n]`` 打开原文。
"""

from __future__ import annotations

import html
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="库默 · Kumore",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.config import settings as app_settings  # noqa: E402
from app.kb import active_kb_id, resolve_settings  # noqa: E402
from app.ui import render  # noqa: E402
from app.ui import session_store  # noqa: E402
from app.ui.service import (  # noqa: E402
    UIConfig,
    build_client,
    build_retriever,
    chunk_lookup,
    corpus_badge,
    ensure_session_dir,
    run_turn,
)
from app.ui.theme import inject_css  # noqa: E402


@st.cache_resource(show_spinner=False)
def _runtime(kb_id: str | None):
    """索引只加载一次；key 带当前知识库 id，切库后缓存自然失效。"""
    settings, kb = resolve_settings(kb_id)
    retriever = build_retriever(settings)
    return retriever, settings, kb


def main() -> None:
    inject_css()

    kb_id = active_kb_id(app_settings.storage_dir)
    try:
        retriever, runtime_settings, kb = _runtime(kb_id)
    except Exception as exc:  # 索引为空 / 一致性校验失败 / 依赖缺失
        _render_setup_error(exc)
        return

    has_index = bool(retriever.chunks)
    api_ready = bool(runtime_settings.deepseek_api_key)
    session_dir = ensure_session_dir(app_settings.storage_dir)
    _init_session(session_dir)

    sessions = session_store.list_sessions(session_dir)
    active = st.session_state["ui_session"]
    action = render.render_sidebar(
        sessions,
        str(active.get("id") or ""),
        (
            f"{kb.name} · {corpus_badge(retriever)}"
            if has_index
            else f"{kb.name} · 先运行 python -m app ingest"
        ),
        has_index,
    )
    if action["new"]:
        _start_new_session()
    if action["selected"] and action["selected"] != active.get("id"):
        loaded = session_store.load_session(action["selected"], session_dir)
        if loaded:
            st.session_state["ui_session"] = loaded
            st.session_state["evidence"] = None
            st.session_state["scroll_pending"] = True
            st.rerun()

    active = st.session_state["ui_session"]
    prompt_variant = str(st.session_state.get("ui_prompt_variant") or "claim")
    mode = str(st.session_state.get("ui_mode") or "agent")
    render.render_topbar(
        str(active.get("title") or "新对话"),
        mode,
        prompt_variant,
        retriever.mode,
        has_index,
    )

    # autoscroll：新 turn 追加后滚到最新一条（Codex 的行为）；否则长会话里
    # 用户提交完还停在顶部，看不到"研究中"。
    thread = st.container(autoscroll=True)
    with thread:
        turns = list(active.get("turns") or [])
        if not turns:
            render.render_hero(has_index)
        else:
            chunks = chunk_lookup(retriever)
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                render.render_user(str(turn.get("question") or ""))
                render.render_turn(turn, chunks, expanded=False)

    disabled = (not has_index) or (not api_ready)
    hint = (
        "先在终端运行 python -m app ingest"
        if not has_index
        else "未配置 DEEPSEEK_API_KEY，无法生成回答"
        if not api_ready
        else "问点什么…"
    )
    prompt = render.render_composer(disabled, hint)

    if prompt:
        _handle_prompt(
            str(prompt),
            thread=thread,
            retriever=retriever,
            runtime_settings=runtime_settings,
            session_dir=session_dir,
            mode=mode,
            prompt_variant=prompt_variant,
        )

    render.render_evidence_drawer(chunk_lookup(retriever))

    # 只在"刚提交/刚切会话"这一轮把视口带到最新一条；点引用导致的 rerun 不跳，
    # 否则用户正在读旧 turn 时会被甩到底部。
    if st.session_state.pop("scroll_pending", False):
        render.scroll_to_bottom()


def _init_session(session_dir: Path) -> None:
    if st.session_state.get("ui_session"):
        return
    existing = session_store.list_sessions(session_dir, limit=1)
    loaded = (
        session_store.load_session(str(existing[0]["id"]), session_dir)
        if existing
        else None
    )
    st.session_state["ui_session"] = loaded or session_store.new_session()
    if loaded:
        st.session_state["scroll_pending"] = True


def _start_new_session() -> None:
    st.session_state["ui_session"] = session_store.new_session()
    st.session_state["evidence"] = None
    st.rerun()


def _handle_prompt(
    question: str,
    *,
    thread: Any,
    retriever: Any,
    runtime_settings: Any,
    session_dir: Path,
    mode: str,
    prompt_variant: str,
) -> None:
    question = question.strip()
    if not question:
        return

    with thread:
        render.render_user(question)
        live = st.empty()
        live.markdown(
            '<div class="k-head"><span class="dot"></span><span class="who">库默</span>'
            '<span class="sep">/</span><span class="meta">连接中…</span></div>',
            unsafe_allow_html=True,
        )
        seen: list[dict[str, Any]] = []

        def on_step(step: Any) -> None:
            seen.append(step.to_dict())
            live.markdown(render.live_steps_html(seen), unsafe_allow_html=True)

    config = UIConfig(
        mode=mode,
        top_k=int(st.session_state.get("ui_top_k") or 5),
        prompt_variant=prompt_variant,
        include_superseded=bool(st.session_state.get("ui_include_superseded")),
    )
    started = time.monotonic()
    try:
        client = build_client(runtime_settings)
        turn = run_turn(question, retriever, client, config, on_step=on_step)
    except Exception as exc:
        turn = _error_turn(question, mode, exc, time.monotonic() - started)

    session = st.session_state["ui_session"]
    saved = session_store.append_turn_atomic(str(session.get("id") or ""), turn, session_dir)
    if saved is not None:
        session = saved
    else:
        session_store.append_turn(session, turn)
        session_store.save_session(session, session_dir)
    st.session_state["ui_session"] = session
    st.session_state["evidence"] = None
    st.session_state["scroll_pending"] = True
    st.rerun()


def _error_turn(question: str, mode: str, exc: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "id": f"t{int(time.time() * 1000)}-{uuid.uuid4().hex[:4]}",
        "question": question,
        "mode": mode,
        "answer": f"运行失败：{type(exc).__name__}: {exc}",
        "refused": True,
        "refusal_reason": None,
        "citations": [],
        "claims": [],
        "claim_fallback": False,
        "warnings": ["本次没有生成答案；检查 API key、网络或索引后重试。"],
        "invalid_markers": [],
        "trace": {},
        "meta": {"mode": mode},
        "elapsed_s": round(elapsed, 2),
    }


def _render_setup_error(exc: Exception) -> None:
    st.markdown(
        """
        <div class="k-hero">
          <div class="k-kicker">KUMORE / SETUP</div>
          <h1>索引没有<em>就绪</em>。</h1>
          <p>先运行 <code>python -m app ingest</code>；如果刚换过语料，
          再跑一次 <code>python -m app stats</code> 核对一致性。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="k-warn">{html.escape(f"{type(exc).__name__}: {exc}")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
