"""M6 展示层（Streamlit）。

只做**展示与编排**，不碰检索/生成口径：问答仍走 ``app.agent`` / ``app.generate``
的同一套函数。UI 侧只新增两件事：

1. 会话落盘（``storage/ui_sessions/``，gitignore），刷新页面不丢历史；
2. 实时步回调（``AgentRunner(on_step=...)``），把"研究过程"边跑边画出来。

设计口径见 ``docs/路线图.md`` §M6 与 ``docs/M6-界面设计.md``。
"""
