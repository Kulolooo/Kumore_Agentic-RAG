"""黑绿主题：只允许黑 / 白灰 / 绿三种色相。

为什么自己注入 CSS：Streamlit 的 ``config.toml`` 只能改主色与背景，
管不到侧栏、步骤行、claim 卡片这些"像 Codex"的细节。CSS 全部集中在这里，
页面文件不写样式，换主题只改一个文件。

**已知脆弱点**：部分选择器依赖 Streamlit 的 ``data-testid``，升级小版本要
重新截图核对（见 ``docs/M6-界面设计.md`` 的维护项）。
"""

from __future__ import annotations

import streamlit as st

#: 设计令牌。黑绿两色：黑是底，绿只做强调，不铺大面积。
TOKENS = {
    "bg": "#070907",
    "bg_elev": "#0A0D0B",
    "panel": "#0D110E",
    "panel_2": "#111713",
    "line": "#1C241E",
    "line_2": "#2A342C",
    "text": "#E8EDE9",
    "text_2": "#A9B3AC",
    "muted": "#77837A",
    "dim": "#4F5C53",
    "green": "#43E08A",
    "green_2": "#25A967",
    "green_soft": "rgba(67, 224, 138, 0.08)",
    "green_line": "rgba(67, 224, 138, 0.30)",
}


def _css() -> str:
    t = TOKENS
    return f"""
<style>
:root {{
  --k-bg: {t['bg']};
  --k-bg-elev: {t['bg_elev']};
  --k-panel: {t['panel']};
  --k-panel-2: {t['panel_2']};
  --k-line: {t['line']};
  --k-line-2: {t['line_2']};
  --k-text: {t['text']};
  --k-text-2: {t['text_2']};
  --k-muted: {t['muted']};
  --k-dim: {t['dim']};
  --k-green: {t['green']};
  --k-green-2: {t['green_2']};
  --k-green-soft: {t['green_soft']};
  --k-green-line: {t['green_line']};
  --k-mono: "Cascadia Code", "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  --k-sans: "Segoe UI Variable Text", "Segoe UI", -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
}}

html, body, [class*="css"] {{
  font-family: var(--k-sans);
  -webkit-font-smoothing: antialiased;
}}

.stApp {{
  background:
    radial-gradient(1100px 520px at 78% -12%, rgba(67, 224, 138, 0.045), transparent 62%),
    var(--k-bg);
  color: var(--k-text);
}}

[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
[data-testid="stStatusWidget"] {{ display: none; }}

/* ---------------- 版心 ---------------- */
.block-container {{
  max-width: 940px;
  padding: 0 2.4rem 11.5rem;
  margin: 0 auto;
}}
@media (max-width: 760px) {{
  .block-container {{ padding: 0 1rem 8.5rem; }}
}}

/* ---------------- 侧栏 ---------------- */
[data-testid="stSidebar"] {{
  background: #080B09;
  border-right: 1px solid var(--k-line);
  min-width: 274px;
  max-width: 274px;
}}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  padding: 1.15rem 0.9rem 1rem;
}}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {{
  color: var(--k-muted);
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.15rem; }}

.k-brand {{
  display: flex; align-items: center; gap: 0.65rem;
  padding: 0.1rem 0.25rem 1.15rem;
}}
.k-mark {{
  width: 27px; height: 27px; border-radius: 8px;
  background: var(--k-green); color: #04150B;
  display: grid; place-items: center;
  font-size: 13px; font-weight: 800; letter-spacing: -0.03em;
  box-shadow: 0 0 0 1px rgba(67,224,138,.35), 0 0 24px rgba(67,224,138,.18);
}}
.k-name {{ font-size: 12px; font-weight: 700; letter-spacing: 0.16em; color: var(--k-text); }}
.k-sub {{ font-size: 9.5px; letter-spacing: 0.10em; color: var(--k-dim); margin-top: 1px; }}

.k-side-label {{
  font-family: var(--k-mono); font-size: 9.5px; letter-spacing: 0.16em;
  color: var(--k-dim);
  /* Streamlit 的 Markdown 容器有 -16px 负边距，会把子元素的下内边距"吃掉"；
     所以高度必须写在**外层 element container** 上（见下一条 :has 规则）。 */
  margin: 0; padding: 0 0 0 0.25rem; height: 34px; line-height: 34px;
}}
/* 让"会话 / 状态"这行真正占 34px，否则下一个按钮会压字（P78 的布局坑） */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.k-side-label) {{
  height: 34px !important;
}}

/* 侧栏会话按钮：左对齐、无填充，选中项由页面注入动态 class 高亮 */
[data-testid="stSidebar"] .stButton > button {{
  justify-content: flex-start; text-align: left;
  background: transparent; border: 1px solid transparent;
  color: var(--k-text-2); font-size: 12.5px;
  padding: 0.42rem 0.6rem; border-radius: 7px; min-height: 2rem;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
  background: var(--k-panel); border-color: var(--k-line); color: var(--k-text);
}}
[data-testid="stSidebar"] .stButton > button p {{ font-size: 12.5px; }}

.k-status {{
  margin-top: 1rem; padding: 0.7rem 0.75rem;
  border: 1px solid var(--k-line); border-radius: 10px;
  background: var(--k-panel);
}}
.k-status .row {{ display: flex; align-items: center; gap: 0.42rem; font-size: 11.5px; color: var(--k-text-2); }}
.k-status .dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--k-green); box-shadow: 0 0 0 3px var(--k-green-soft);
}}
.k-status .meta {{
  margin-top: 0.28rem; font-family: var(--k-mono); font-size: 9.8px;
  color: var(--k-dim); line-height: 1.6;
}}

/* ---------------- 通用控件 ---------------- */
[data-testid="stWidgetLabel"] p {{ color: var(--k-muted); font-size: 11px; }}
.stButton > button {{
  border-radius: 8px; border: 1px solid var(--k-line);
  background: var(--k-panel); color: var(--k-text-2);
  font-size: 12.5px; box-shadow: none;
}}
.stButton > button:hover {{
  border-color: var(--k-line-2); color: var(--k-text); background: var(--k-panel-2);
}}
button[data-testid="stBaseButton-primary"] {{
  background: var(--k-green) !important; border-color: var(--k-green) !important;
  color: #04150B !important; font-weight: 650;
}}
button[data-testid="stBaseButton-primary"]:hover {{
  background: #5CEB9C !important; border-color: #5CEB9C !important;
}}
button[data-testid="stBaseButton-tertiary"] {{
  background: transparent !important; border-color: transparent !important;
}}

[data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
  background: var(--k-panel); border-color: var(--k-line); color: var(--k-text);
  font-size: 12px;
}}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stPopover"]) {{
  margin-top: 0.55rem;
}}
[data-testid="stPopover"] > button {{
  color: var(--k-text-2); font-size: 12px;
  background: var(--k-panel); border: 1px solid var(--k-line); border-radius: 8px;
}}

[data-testid="stSegmentedControl"] {{
  background: var(--k-panel); border: 1px solid var(--k-line);
  border-radius: 9px; padding: 2px;
}}
[data-testid="stSegmentedControl"] button {{
  color: var(--k-muted); font-size: 11.5px; border-radius: 6px; padding: 0.2rem 0.65rem;
}}
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {{
  background: var(--k-green-soft); color: var(--k-green);
  box-shadow: inset 0 0 0 1px var(--k-green-line);
}}

[data-testid="stExpander"] {{
  border: 1px solid var(--k-line); border-radius: 11px;
  background: var(--k-panel); overflow: hidden;
}}
[data-testid="stExpander"] summary {{
  padding: 0.5rem 0.75rem; color: var(--k-text-2); font-size: 12px;
}}
[data-testid="stExpander"] summary:hover {{ color: var(--k-text); background: var(--k-panel-2); }}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ padding: 0 0.75rem 0.85rem; }}
[data-testid="stExpander"] svg {{ color: var(--k-dim); }}

[data-testid="stMarkdownContainer"] p {{ line-height: 1.75; }}
[data-testid="stMarkdownContainer"] code {{
  font-family: var(--k-mono); font-size: 0.86em;
  background: var(--k-panel-2); color: var(--k-green);
  border: 1px solid var(--k-line); border-radius: 5px; padding: 0.05em 0.35em;
}}
[data-testid="stMarkdownContainer"] pre {{
  background: var(--k-panel); border: 1px solid var(--k-line); border-radius: 9px;
}}

/* ---------------- 顶栏 ---------------- */
.k-topbar {{
  position: sticky; top: 0; z-index: 30;
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: 0.75rem 0 0.7rem; margin-bottom: 0.35rem;
  background: linear-gradient(to bottom, rgba(7,9,7,0.96) 62%, rgba(7,9,7,0));
  backdrop-filter: blur(10px);
}}
.k-top-title {{
  font-size: 12.5px; color: var(--k-text-2); font-weight: 600;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 62%;
}}
.k-top-meta {{
  display: flex; align-items: center; gap: 0.55rem;
  font-family: var(--k-mono); font-size: 9.5px; letter-spacing: 0.08em; color: var(--k-dim);
}}
.k-top-meta .pill {{
  border: 1px solid var(--k-line); border-radius: 999px;
  padding: 0.12rem 0.5rem; color: var(--k-muted);
}}

/* ---------------- 空状态 ---------------- */
.k-hero {{ padding: 8.5vh 0 3vh; }}
.k-kicker {{
  font-family: var(--k-mono); font-size: 10px; letter-spacing: 0.2em;
  color: var(--k-green); text-transform: uppercase;
}}
.k-hero h1 {{
  margin: 0.75rem 0 0.55rem; font-size: clamp(28px, 3.6vw, 40px);
  line-height: 1.22; letter-spacing: -0.025em; font-weight: 620; color: var(--k-text);
}}
.k-hero h1 em {{ font-style: normal; color: var(--k-green); }}
.k-hero p {{ margin: 0; max-width: 590px; color: var(--k-muted); font-size: 13.5px; line-height: 1.85; }}
.k-feats {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
  margin-top: 2.2rem; background: var(--k-line);
  border: 1px solid var(--k-line); border-radius: 12px; overflow: hidden;
}}
.k-feat {{ background: var(--k-bg); padding: 0.95rem 1rem; }}
.k-feat .n {{ font-family: var(--k-mono); font-size: 9.5px; color: var(--k-green); letter-spacing: 0.14em; }}
.k-feat .t {{ margin-top: 0.35rem; font-size: 12.5px; color: var(--k-text); }}
.k-feat .d {{ margin-top: 0.2rem; font-size: 11px; color: var(--k-dim); line-height: 1.55; }}
@media (max-width: 760px) {{ .k-feats {{ grid-template-columns: 1fr; }} }}

/* ---------------- 对话流 ---------------- */
.k-user-row {{ display: flex; justify-content: flex-end; margin: 1.6rem 0 0.45rem; }}
.k-user {{
  max-width: 80%; padding: 0.62rem 0.9rem;
  background: var(--k-panel-2); border: 1px solid var(--k-line);
  border-radius: 13px 13px 4px 13px;
  color: var(--k-text); font-size: 13.5px; line-height: 1.7;
  white-space: pre-wrap; overflow-wrap: anywhere;
}}

.k-head {{ display: flex; align-items: center; gap: 0.45rem; margin: 0.95rem 0 0.3rem; }}
.k-head .dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--k-green); box-shadow: 0 0 0 3px var(--k-green-soft);
}}
.k-head .who {{ font-size: 12px; font-weight: 650; color: var(--k-text); letter-spacing: 0.02em; }}
.k-head .sep {{ color: var(--k-line-2); }}
.k-head .meta {{ font-family: var(--k-mono); font-size: 9.5px; color: var(--k-dim); letter-spacing: 0.06em; }}

.k-answer {{ font-size: 14.5px; line-height: 1.82; color: var(--k-text); }}
.k-answer p {{ margin: 0.3rem 0; }}

/* ---------- 研究过程 ---------- */
.k-steps {{ display: flex; flex-direction: column; }}
.k-step {{
  display: grid; grid-template-columns: 24px 76px minmax(0, 1fr) auto;
  gap: 0.45rem; align-items: baseline;
  padding: 0.3rem 0; border-bottom: 1px dashed rgba(42,52,44,0.5);
  font-size: 11.5px;
}}
.k-step:last-child {{ border-bottom: none; }}
.k-step .no {{ font-family: var(--k-mono); font-size: 9.5px; color: var(--k-dim); }}
.k-step .tool {{ color: var(--k-text-2); white-space: nowrap; }}
.k-step .tool .ico {{ color: var(--k-green-2); margin-right: 0.3rem; }}
.k-step .arg {{
  color: var(--k-muted); font-family: var(--k-mono); font-size: 10.8px;
  overflow-wrap: anywhere; line-height: 1.6;
}}
.k-step .meta {{ font-family: var(--k-mono); font-size: 9.3px; color: var(--k-dim); white-space: nowrap; }}
.k-step .detail {{ grid-column: 3 / -1; color: var(--k-dim); font-size: 10.5px; line-height: 1.55; }}
.k-step.fail .tool {{ color: var(--k-green-2); }}
.k-step.fail .tool .ico {{ color: var(--k-dim); }}
.k-trace-foot {{
  display: flex; flex-wrap: wrap; gap: 0.85rem;
  margin-top: 0.55rem; padding: 0.5rem 0 0.1rem;
  border-top: 1px solid var(--k-line);
  font-family: var(--k-mono); font-size: 9.5px; color: var(--k-muted);
}}

/* ---------- claim ---------- */
.k-claim {{
  margin: 0.55rem 0 0.25rem; padding: 0.75rem 0.9rem;
  border: 1px solid var(--k-line); border-radius: 11px;
  background: var(--k-panel);
}}
.k-claim-top {{ display: flex; align-items: center; gap: 0.5rem; }}
.k-claim-no {{ font-family: var(--k-mono); font-size: 9.5px; color: var(--k-green); letter-spacing: 0.1em; }}
.k-claim-badge {{
  margin-left: auto; font-family: var(--k-mono); font-size: 9px;
  color: var(--k-dim); border: 1px solid var(--k-line); border-radius: 999px;
  padding: 0.05rem 0.4rem;
}}
.k-claim-badge.ok {{ color: var(--k-green-2); border-color: var(--k-green-line); }}
.k-claim-stmt {{ margin-top: 0.35rem; font-size: 14px; line-height: 1.75; color: var(--k-text); }}
.k-quote {{
  margin-top: 0.55rem; padding: 0.45rem 0.65rem;
  border-left: 2px solid var(--k-green-line);
  background: rgba(255,255,255,0.012);
  color: var(--k-text-2); font-size: 11.8px; line-height: 1.7;
  white-space: pre-wrap; overflow-wrap: anywhere;
}}
.k-quote .q-label {{
  display: block; font-family: var(--k-mono); font-size: 8.8px;
  letter-spacing: 0.14em; color: var(--k-dim); margin-bottom: 0.15rem;
}}

/* ---------- 拒答 / 诊断 ---------- */
.k-refusal {{
  margin: 0.55rem 0; padding: 0.8rem 0.95rem;
  border: 1px dashed var(--k-line-2); border-radius: 11px;
  background: var(--k-panel);
}}
.k-refusal .tag {{
  font-family: var(--k-mono); font-size: 9px; letter-spacing: 0.14em;
  color: var(--k-green); text-transform: uppercase;
}}
.k-refusal .body {{ margin-top: 0.35rem; font-size: 13px; line-height: 1.7; color: var(--k-text-2); }}
.k-warn {{
  margin: 0.35rem 0; padding: 0.4rem 0.6rem;
  border-left: 2px solid var(--k-green-line);
  color: var(--k-dim); font-size: 10.5px; line-height: 1.6;
}}

.k-turn-foot {{
  display: flex; flex-wrap: wrap; gap: 0.8rem;
  margin: 0.55rem 0 1.4rem;
  font-family: var(--k-mono); font-size: 9.3px; letter-spacing: 0.05em; color: var(--k-dim);
}}
.k-turn-foot .g {{ color: var(--k-green-2); }}

/* ---------------- 引用 chip ---------------- */
div[class*="st-key-citechips_"] [data-testid="stHorizontalBlock"] {{ gap: 0.3rem; }}
div[class*="st-key-citechips_"] button {{
  min-height: 24px; height: 24px; padding: 0 0.55rem;
  background: var(--k-green-soft); border: 1px solid var(--k-green-line);
  border-radius: 999px; color: var(--k-green);
  font-family: var(--k-mono); font-size: 10.5px;
}}
div[class*="st-key-citechips_"] button:hover {{
  background: rgba(67,224,138,0.14); border-color: var(--k-green); color: var(--k-green);
}}
div[class*="st-key-citechips_"] button p {{ font-size: 10.5px; }}

/* ---------------- 底部输入 ---------------- */
[data-testid="stBottom"] {{
  background: linear-gradient(to top, #070907 58%, rgba(7,9,7,0));
  border-top: none;
}}
[data-testid="stBottomBlockContainer"] {{
  max-width: 940px; margin: 0 auto;
  padding: 0.55rem 2.4rem 1.15rem;
}}
@media (max-width: 760px) {{
  [data-testid="stBottomBlockContainer"] {{ padding: 0.4rem 1rem 0.9rem; }}
}}
[data-testid="stChatInput"] {{
  background: var(--k-panel);
  border: 1px solid var(--k-line-2);
  border-radius: 13px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
  transition: border-color .16s ease, box-shadow .16s ease;
}}
[data-testid="stChatInput"]:focus-within {{
  border-color: var(--k-green-line);
  box-shadow: 0 0 0 3px var(--k-green-soft), 0 12px 40px rgba(0,0,0,0.35);
}}
[data-testid="stChatInput"] textarea {{
  color: var(--k-text); font-size: 13.5px; font-family: var(--k-sans);
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--k-dim); }}
[data-testid="stChatInput"] button {{ color: var(--k-green); }}
[data-testid="stChatInput"] button:hover {{ background: var(--k-green-soft); }}
[data-testid="stChatInput"] button svg {{ fill: var(--k-green); color: var(--k-green); }}

.k-composer-row {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.75rem; padding: 0 0.2rem 0.45rem;
}}
.k-composer-hint {{
  font-family: var(--k-mono); font-size: 9.2px; color: var(--k-dim); letter-spacing: 0.04em;
}}
.k-composer-hint b {{ color: var(--k-muted); font-weight: 500; }}

/* ---------------- 证据抽屉 ---------------- */
.k-shade {{
  position: fixed; inset: 0; z-index: 990;
  background: rgba(0,0,0,0.42); backdrop-filter: blur(2px);
}}
.st-key-evidence_drawer {{
  position: fixed; top: 0; right: 0; z-index: 1000;
  width: 450px; max-width: 94vw; height: 100vh;
  padding: 1.05rem 1.1rem 1.4rem;
  background: #090C0A;
  border-left: 1px solid var(--k-line-2);
  box-shadow: -30px 0 70px rgba(0,0,0,0.55);
  overflow-y: auto;
}}
.st-key-evidence_drawer [data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
.k-drawer-head {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; }}
.k-drawer-kicker {{ font-family: var(--k-mono); font-size: 9px; letter-spacing: 0.18em; color: var(--k-green); }}
.k-source {{ margin-top: 0.55rem; font-size: 13.5px; font-weight: 600; color: var(--k-text); line-height: 1.55; }}
.k-source-meta {{ margin-top: 0.2rem; font-family: var(--k-mono); font-size: 9.8px; color: var(--k-dim); line-height: 1.7; }}
.k-chunk {{
  margin-top: 0.7rem; padding: 0.75rem 0.8rem;
  background: var(--k-panel); border: 1px solid var(--k-line); border-radius: 10px;
  color: var(--k-text-2); font-size: 12.3px; line-height: 1.8;
  white-space: pre-wrap; overflow-wrap: anywhere;
  max-height: 56vh; overflow-y: auto;
}}
.k-chunk mark {{
  background: var(--k-green-soft); color: var(--k-green);
  border-bottom: 1px solid var(--k-green); padding: 0 0.08em;
}}
.k-drawer-quote {{
  margin-top: 0.6rem; padding: 0.55rem 0.65rem;
  border: 1px dashed var(--k-line-2); border-radius: 9px;
  font-size: 11px; color: var(--k-muted); line-height: 1.7;
}}

/* ---------------- 滚动条 ---------------- */
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #1D271F; border-radius: 9px; border: 2px solid #070907; }}
::-webkit-scrollbar-thumb:hover {{ background: #2A382D; }}

/* 窄屏隐藏侧栏时，主区留白收紧 */
@media (max-width: 760px) {{
  .k-hero {{ padding-top: 5vh; }}
  .k-user {{ max-width: 92%; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_css(), unsafe_allow_html=True)


def active_session_css(session_id: str) -> None:
    """给当前会话按钮加一层绿色描边（Streamlit 不支持 per-button class）。"""
    if not session_id:
        return
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
    st.markdown(
        f"""<style>
        .st-key-session_btn_{safe} button {{
          background: var(--k-green-soft) !important;
          border-color: var(--k-green-line) !important;
          color: var(--k-text) !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
