"""M4 agentic 检索层。

分工：

- ``toolspec``：四个工具的 OpenAI 函数定义（**纯数据**，不含执行逻辑）；
- ``tools``：执行层，全部走 ``Retriever`` 与现有存储接口，命中即向
  ``EvidenceRegistry`` 要 ``[ID:n]``；
- ``prompts``：研究轮 / 充分性判定 / 兜底合成三段提示词；
- ``trace``：每一步的 JSONL 轨迹（双通道渲染的数据源）；
- ``loop``：循环本体（首轮强制检索 → 充分性 → claim 短路 → 兜底合成 → 预算）。

为什么把 spec 与执行分开：闸门 C 要**对着真正要上线的工具定义**测模型能不能稳定发出
``tool_call``，如果 spec 和实现各写一份，测出来的稳定性就不代表上线行为。
"""

from __future__ import annotations

__all__ = ["toolspec", "tools", "prompts", "trace", "loop"]
