"""把 turn / trace / claim 拼成 Codex 风格的界面。

约束：所有来自模型或语料的字符串，进 HTML 前一律 ``html.escape``；
正文答案走 ``st.markdown`` 默认转义（不允许 HTML），避免文档里的
``<script>`` 被当成前端代码执行。组件只读 session_state，写状态集中在
``select_evidence`` / ``render_evidence_drawer`` 两处。
"""

from __future__ import annotations

import html
import textwrap
from typing import Any

import streamlit as st

from app.ui import format as fmt
from app.ui.theme import active_session_css


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


# ---------------------------------------------------------------- 侧栏
def render_sidebar(
    sessions: list[dict[str, Any]],
    active_id: str,
    stats_line: str,
    has_index: bool,
) -> dict[str, Any]:
    """返回 ``{"new": bool, "selected": str | None}``。"""
    action = {"new": False, "selected": None}
    with st.sidebar:
        st.markdown(
            """
            <div class="k-brand">
              <div class="k-mark">K</div>
              <div>
                <div class="k-name">KUMORE</div>
                <div class="k-sub">AGENTIC RAG</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("＋  新对话", type="primary", use_container_width=True, key="new_chat"):
            action["new"] = True

        query = st.text_input(
            "搜索会话",
            placeholder="搜索会话",
            label_visibility="collapsed",
            key="session_search",
        )

        st.markdown('<div class="k-side-label">会话</div>', unsafe_allow_html=True)
        shown = 0
        needle = (query or "").strip().lower()
        for row in sessions:
            title = str(row.get("title") or "新对话")
            if needle and needle not in title.lower():
                continue
            shown += 1
            if st.button(
                title,
                key=f"session_btn_{row['id']}",
                use_container_width=True,
                help=f"{row.get('n_turns', 0)} 轮 · {fmt_time(row.get('updated_at'))}",
            ):
                action["selected"] = str(row["id"])
        if not shown:
            st.caption("还没有会话" if not needle else "没有匹配的会话")

        st.markdown('<div class="k-side-label">状态</div>', unsafe_allow_html=True)
        dot = "var(--k-green)" if has_index else "var(--k-dim)"
        st.markdown(
            f"""
            <div class="k-status">
              <div class="row"><span class="dot" style="background:{dot}"></span>
                {'索引就绪' if has_index else '索引为空'}</div>
              <div class="meta">{esc(stats_line)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:0.55rem"></div>', unsafe_allow_html=True)
        with st.popover("设置", use_container_width=True):
            st.segmented_control(
                "检索深度 top-k",
                options=[3, 5, 8],
                default=5,
                key="ui_top_k",
                label_visibility="visible",
            )
            st.selectbox(
                "引用结构",
                options=["claim", "strict", "tight"],
                index=0,
                key="ui_prompt_variant",
                help="claim：每条陈述带逐字证据与句柄；strict/tight：散文回答。",
            )
            st.toggle("包含旧版本文档", value=False, key="ui_include_superseded")
            st.caption("Agent：最多 10 步 / 120k tokens。单步：检索 top-k 后直接生成。")

    if active_id:
        active_session_css(active_id)
    return action


def fmt_time(value: Any) -> str:
    from app.ui.session_store import relative_time

    return relative_time(str(value or ""))


# ---------------------------------------------------------------- 顶栏 / 空状态
def render_topbar(
    title: str,
    mode: str,
    prompt_variant: str,
    retrieval_mode: str,
    has_index: bool,
) -> None:
    st.markdown(
        f"""
        <div class="k-topbar">
          <div class="k-top-title">{esc(title or "新对话")}</div>
          <div class="k-top-meta">
            <span class="pill">{esc(fmt.mode_label(mode))}</span>
            <span class="pill">{esc(prompt_variant)}</span>
            <span class="pill">{esc(retrieval_mode) if has_index else 'no index'}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(has_index: bool) -> None:
    if not has_index:
        st.markdown(
            """
            <div class="k-hero">
              <div class="k-kicker">KUMORE / SETUP</div>
              <h1>还没有<em>索引</em>。</h1>
              <p>先在终端运行 <code>python -m app ingest</code>，再刷新本页。
              界面不会替代入库，也不做“没证据先答一句”的兜底。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        """
        <div class="k-hero">
          <div class="k-kicker">KUMORE / AGENTIC RAG</div>
          <h1>答案有据，<em>可以回到原文</em>。</h1>
          <p>只检索你的文档。每个结论带逐字证据与 [ID:n] 句柄，
          点引用能看到它落在原文的哪一句。</p>
          <div class="k-feats">
            <div class="k-feat"><div class="n">01</div><div class="t">强制检索</div><div class="d">先取证，再回答；没有证据就拒答</div></div>
            <div class="k-feat"><div class="n">02</div><div class="t">逐字引用</div><div class="d">陈述、引文、句柄三件套</div></div>
            <div class="k-feat"><div class="n">03</div><div class="t">可回原文</div><div class="d">点开引用看上下文与页码</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 对话
def render_user(question: str) -> None:
    st.markdown(
        f'<div class="k-user-row"><div class="k-user">{esc(question)}</div></div>',
        unsafe_allow_html=True,
    )


def render_turn(turn: dict[str, Any], chunks: dict[str, Any], *, expanded: bool = False) -> None:
    mode = fmt.mode_label(str(turn.get("mode") or ""))
    st.markdown(
        f"""
        <div class="k-head">
          <span class="dot"></span><span class="who">库默</span>
          <span class="sep">/</span><span class="meta">{esc(mode)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    trace = turn.get("trace") or {}
    if trace.get("steps"):
        render_steps(trace, expanded=expanded)

    citations = list(turn.get("citations") or [])
    by_number = {int(c.get("marker") or 0): c for c in citations}

    if turn.get("refused"):
        reason = fmt.refusal_label(turn.get("refusal_reason"))
        body = fmt.strip_handles(str(turn.get("answer") or ""))
        st.markdown(
            f"""
            <div class="k-refusal">
              <div class="tag">拒答 · {esc(reason)}</div>
              <div class="body">{esc(body)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif turn.get("claims"):
        render_claims(turn, by_number)
    else:
        # 正文走 st.markdown 默认转义：文档里出现 <script> 也不会被当 HTML 执行。
        st.markdown(format_answer(turn), unsafe_allow_html=False)

    if citations and not turn.get("claims"):
        render_citation_chips(turn, citations, label_files=True)

    warnings = list(turn.get("warnings") or [])
    if warnings:
        with st.expander("诊断", expanded=False):
            for warning in warnings:
                st.markdown(
                    f'<div class="k-warn">{esc(warning)}</div>', unsafe_allow_html=True
                )
            invalid = turn.get("invalid_markers") or []
            if invalid:
                st.markdown(
                    f'<div class="k-warn">越界编号（已从正文删除）：{esc(invalid)}</div>',
                    unsafe_allow_html=True,
                )

    render_turn_footer(turn)


def format_answer(turn: dict[str, Any]) -> str:
    text = str(turn.get("answer") or "")
    cleaned = fmt.strip_handles(text)
    return cleaned or "（空回答）"


def render_turn_footer(turn: dict[str, Any]) -> None:
    meta = turn.get("meta") or {}
    bits = []
    tokens = fmt.format_tokens(meta.get("tokens"))
    if tokens:
        bits.append(tokens)
    if turn.get("elapsed_s"):
        bits.append(f"{turn['elapsed_s']}s")
    if meta.get("model"):
        bits.append(str(meta["model"]))
    if meta.get("n_evidence"):
        bits.append(f"{meta['n_evidence']} 块证据")
    if not bits:
        return
    st.markdown(
        '<div class="k-turn-foot">'
        + "".join(f"<span>{esc(bit)}</span>" for bit in bits)
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 研究过程
def live_steps_html(steps: list[dict[str, Any]]) -> str:
    """运行中的研究过程：不建 widget，直接整块 HTML 覆盖 placeholder。

    为什么不用 ``st.expander``：回调里反复创建 widget 会拿到同一个 key 冲突；
    直接写 HTML 最稳，跑完再 ``st.rerun`` 换成可折叠的正式组件。
    """
    rows: list[str] = []
    for display_no, raw in enumerate(steps, 1):
        step = fmt.step_view(raw)
        step["no"] = display_no
        rows.append(
            textwrap.dedent(
                f"""
                <div class="k-step{'' if step['ok'] else ' fail'}">
                  <span class="no">{step['no']:02d}</span>
                  <span class="tool"><span class="ico">{esc(step['icon'])}</span>{esc(step['label'])}</span>
                  <span class="arg">{esc(step['arg'])}</span>
                  <span class="meta">{esc(step['meta'])}</span>
                  {f'<span class="detail">{esc(step["detail"])}</span>' if step['detail'] else ""}
                </div>
                """
            ).strip()
        )
    return (
        '<div class="k-head"><span class="dot"></span><span class="who">库默</span>'
        '<span class="sep">/</span><span class="meta">研究中</span></div>'
        '<div class="k-steps">' + "".join(rows) + "</div>"
    )


def render_steps(trace: dict[str, Any], *, expanded: bool = False) -> None:
    steps = [fmt.step_view(step) for step in trace.get("steps") or []]
    summary = fmt.trace_summary(trace)
    label_bits = [f"研究过程 · {summary['n_steps']} 步"]
    if summary["duration"]:
        label_bits.append(summary["duration"])
    if summary["outcome"] == "refuse":
        label_bits.append("拒答")
    label = " · ".join(label_bits)

    with st.expander(label, expanded=expanded):
        rows: list[str] = []
        for display_no, step in enumerate(steps, 1):
            step["no"] = display_no
            cls = "" if step["ok"] else " fail"
            meta = step["meta"]
            rows.append(
                textwrap.dedent(
                    f"""
                    <div class="k-step{cls}">
                      <span class="no">{step['no']:02d}</span>
                      <span class="tool"><span class="ico">{esc(step['icon'])}</span>{esc(step['label'])}</span>
                      <span class="arg">{esc(step['arg'])}</span>
                      <span class="meta">{esc(meta)}</span>
                      {f'<span class="detail">{esc(step["detail"])}</span>' if step["detail"] else ""}
                    </div>
                    """
                ).strip()
            )
        st.markdown('<div class="k-steps">' + "".join(rows) + "</div>", unsafe_allow_html=True)

        foot = [
            f"终局 {summary['outcome'] or '—'}",
            f"证据 {summary['n_evidence']} 块",
        ]
        if summary["tokens"]:
            foot.append(summary["tokens"])
        if summary["degradations"]:
            foot.append("降级 " + " / ".join(summary["degradations"]))
        st.markdown(
            '<div class="k-trace-foot">'
            + "".join(f"<span>{esc(bit)}</span>" for bit in foot)
            + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- claim
def render_claims(turn: dict[str, Any], by_number: dict[int, dict[str, Any]]) -> None:
    claims = list(turn.get("claims") or [])
    turn_id = str(turn.get("id") or "t")
    for index, claim in enumerate(claims):
        statement = fmt.strip_handles(str(claim.get("statement") or ""))
        evidence = str(claim.get("evidence") or "").strip()
        badge = "逐字" if claim.get("verbatim") else "未定位"
        badge_cls = " ok" if claim.get("verbatim") else ""
        quote = (
            f'<div class="k-quote"><span class="q-label">证据</span>{esc(evidence)}</div>'
            if evidence
            else '<div class="k-quote"><span class="q-label">证据</span>（模型未给逐字引文）</div>'
        )
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="k-claim">
                  <div class="k-claim-top">
                    <span class="k-claim-no">CLAIM {index + 1:02d}</span>
                    <span class="k-claim-badge{badge_cls}">{esc(badge)}</span>
                  </div>
                  <div class="k-claim-stmt">{esc(statement) or "（空陈述）"}</div>
                  {quote}
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
        numbers = [int(n) for n in (claim.get("numbers") or [])]
        if numbers:
            render_citation_chips(
                turn,
                [by_number[n] for n in numbers if n in by_number],
                label_files=False,
                key_suffix=f"c{index}",
                quote=evidence,
                statement=statement,
            )


# ---------------------------------------------------------------- 引用
def render_citation_chips(
    turn: dict[str, Any],
    citations: list[dict[str, Any]],
    *,
    label_files: bool,
    key_suffix: str = "all",
    quote: str = "",
    statement: str = "",
) -> None:
    if not citations:
        return
    turn_id = str(turn.get("id") or "t")
    with st.container(horizontal=True, key=f"citechips_{turn_id}_{key_suffix}"):
        for citation in citations:
            marker = int(citation.get("marker") or 0)
            label = (
                fmt.citation_label(citation)
                if label_files
                else f"[ID:{marker}]"
            )
            if st.button(
                label,
                key=f"cite_{turn_id}_{key_suffix}_{marker}",
                help=fmt.citation_help(citation),
            ):
                select_evidence(citation, quote=quote, statement=statement)
                st.rerun()


def select_evidence(citation: dict[str, Any], *, quote: str = "", statement: str = "") -> None:
    """选中一条引用；抽屉在页面末尾渲染，靠这个 state 驱动。"""
    st.session_state["evidence"] = {
        "marker": int(citation.get("marker") or 0),
        "handle": str(citation.get("handle") or f"[ID:{citation.get('marker', 0)}]"),
        "chunk_id": str(citation.get("chunk_id") or ""),
        "rel_path": str(citation.get("rel_path") or ""),
        "page": citation.get("page"),
        "page_end": citation.get("page_end"),
        "section": citation.get("section") or "",
        "preview": str(citation.get("text") or ""),
        "quote": quote,
        "statement": statement,
    }


# ---------------------------------------------------------------- 证据抽屉
def render_evidence_drawer(chunks: dict[str, Any]) -> None:
    state = st.session_state.get("evidence")
    if not state:
        return
    st.markdown('<div class="k-shade"></div>', unsafe_allow_html=True)
    with st.container(key="evidence_drawer"):
        left, right = st.columns([5, 1], vertical_alignment="center")
        with left:
            st.markdown('<div class="k-drawer-kicker">EVIDENCE / SOURCE</div>', unsafe_allow_html=True)
        with right:
            if st.button("✕", key="close_evidence", help="关闭"):
                st.session_state["evidence"] = None
                st.rerun()

        chunk = chunks.get(state.get("chunk_id") or "")
        # 父子块：citation 指向命中的子块，模型看到的是父块；抽屉展示父块正文。
        mom_id = getattr(chunk, "mom_id", None)
        if mom_id and mom_id in chunks:
            chunk = chunks[mom_id]
        page = fmt.page_label(state.get("page"), state.get("page_end"))
        rel_path = fmt.mask_private(str(state.get("rel_path") or ""))
        meta = " · ".join(
            bit for bit in (rel_path, page, state.get("section")) if bit
        )
        st.markdown(
            f"""
            <div class="k-source">{esc(fmt.file_name(rel_path, 60))}</div>
            <div class="k-source-meta">{esc(meta)}</div>
            """,
            unsafe_allow_html=True,
        )

        text = str(getattr(chunk, "text", "") or "")
        if not text:
            text = str(state.get("preview") or "")
            note = "当前索引里找不到这个 chunk（只显示引用时的摘要）。"
        else:
            note = ""
        quote = str(state.get("quote") or "")
        body = fmt.highlight_html(text, quote) if quote else html.escape(text)
        st.markdown(f'<div class="k-chunk">{body}</div>', unsafe_allow_html=True)
        if note:
            st.markdown(f'<div class="k-drawer-quote">{esc(note)}</div>', unsafe_allow_html=True)
        if state.get("statement"):
            st.markdown(
                f'<div class="k-drawer-quote">该引用支撑的陈述：{esc(state["statement"])}</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"句柄 {state.get('handle') or ''} · chunk {state.get('chunk_id') or '?'}")


def scroll_to_bottom() -> None:
    """把主滚动容器拉到底。

    Streamlit 的 ``container(autoscroll=True)`` 只对固定高度容器生效；主页面
    的滚动容器是 ``[data-testid="stMain"]``。``st.html`` 默认忽略 JS，这里显式
    打开 ``unsafe_allow_javascript``——脚本是常量，不拼任何用户输入。
    """
    st.html(
        """
        <script>
        const main = document.querySelector('[data-testid="stMain"]');
        if (main) { main.scrollTop = main.scrollHeight; }
        </script>
        """,
        unsafe_allow_javascript=True,
    )


# ---------------------------------------------------------------- 底部输入
def render_composer(disabled: bool, disabled_hint: str) -> str | None:
    with st.bottom:
        c1, c2 = st.columns([1, 1.6], vertical_alignment="center")
        with c1:
            st.segmented_control(
                "模式",
                options=["agent", "single"],
                default="agent",
                format_func=fmt.mode_label,
                key="ui_mode",
                label_visibility="collapsed",
            )
        with c2:
            st.markdown(
                '<div class="k-composer-hint" style="text-align:right">'
                "<b>Enter</b> 发送　<b>Shift+Enter</b> 换行</div>",
                unsafe_allow_html=True,
            )
        prompt = st.chat_input(
            disabled_hint if disabled else "问点什么…",
            disabled=disabled,
            key="ui_prompt",
        )
    return prompt
