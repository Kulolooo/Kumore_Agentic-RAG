"""证据注册表：``[ID:n]`` 句柄的发号、去重与回解。

抄自 RAGFlow 的 ``EvidenceRegistry``（``_ref/ragflow-main/internal/agentic_rag/
evidence_registry.go:27-156``），只抄机制，不抄并发/上下文那套外壳（单机单请求，
一把锁即可，这里连锁都不需要）。

三条与源码一致的约定：

1. **首见序发号**：``stamp`` 返回 ``len(order)``，即第一个被展示的片段拿到 ``[ID:0]``
   （``evidence_registry.go:104-112``）。编号**不是**由内容哈希派生的——
   路线图原来那句写错了，M3 已按源码改正。
2. **去重键是 ``id + \\x1f + sha256(text)[:16]``**（``:76``）：同一个 chunk 被展示两次
   只发一个号；但同一 chunk 的**不同文字窗口**是两个不同的"已服务"记录，
   因为模型读到的内容变了。发号仍按 id 去重（同 id 永远同一个号）。
3. **``resolve(n)`` 回 chunk**，越界返回 ``None``（``:146-156``）。
   空 id 返回 ``-1``，调用方按"不可引用"处理。

为什么编号要在这里发、而不是在渲染器里发：模型只能引用它**看见过**的编号，
所以编号必须在片段抵达模型的那一刻就定下来；否则渲染期换顺序，模型写回的号
就指错了地方（这也是 M3 要替掉"用 hits 下标当编号"的原因——下标会随过滤、
折叠、版本补充而变化）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: 空 id / 未注册时返回的哨兵值。源码同样用 -1 表示"不可引用"。
NOT_CITABLE = -1


def handle(n: int) -> str:
    """``n`` → 模型看到并写回的字面句柄。"""
    return f"[ID:{n}]"


def text_fingerprint(text: str) -> str:
    """``serve_text`` 的去重键里那半段：sha256(正文) 的前 16 位。

    只用于"这段文字模型是不是第一次看到"，不参与发号。
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Served:
    """一次"展示给模型"的结果，便于调用方判断要不要重发正文。"""

    number: int
    #: 这段**文字**是不是本轮第一次出现（同 id 不同窗口也算新）
    fresh: bool


class EvidenceRegistry:
    """一次问答（会话）内的 ``[ID:n]`` 分发中心。"""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._num_of: dict[str, int] = {}
        self._served: set[str] = set()
        #: 发号时对应的 chunk 正文指纹，用于"同号换文"的自检（见 attach_text）
        self._text_of: dict[int, str] = {}

    # ---------------------------------------------------------------- 发号
    @classmethod
    def from_hits(cls, hits) -> "EvidenceRegistry":
        """按上下文顺序给每条命中发号，并记下它被展示的正文。

        ``hits`` 是 ``Retrieved`` 列表；编号顺序与提示词里资料块的顺序**必须**
        一致，否则模型写的号会指到别处。所以这里只按传入顺序发号，不看 ``rank``。
        """
        registry = cls()
        for hit in hits:
            registry.stamp(hit.chunk.chunk_id, text=hit.context_chunk.text)
        return registry

    def stamp(self, chunk_id: str, *, text: str | None = None) -> int:
        """返回 chunk 的稳定句柄；第一次见时分配下一个号。

        空 id 返回 ``-1``（不可引用）。``text`` 只用于记录展示正文，
        不影响发号。
        """
        chunk_id = (chunk_id or "").strip()
        if not chunk_id:
            return NOT_CITABLE
        if chunk_id in self._num_of:
            return self._num_of[chunk_id]
        number = len(self._order)
        self._order.append(chunk_id)
        self._num_of[chunk_id] = number
        if text is not None:
            self._text_of[number] = text
        return number

    def serve_text(self, chunk_id: str, text: str) -> Served:
        """发号 + 报告"这段文字是否第一次出现"。

        对应源码 ``ServeText``（``:68-86``）：ReAct 循环会重复取回同一份文档，
        第一次已经把正文发进上下文，后面只需要点名。单轮问答用不到它，
        但 M4 的多轮/多工具会用到，先按源码把机制放好。
        """
        number = self.stamp(chunk_id)
        if number == NOT_CITABLE:
            return Served(number=NOT_CITABLE, fresh=False)
        key = f"{chunk_id}\x1f{text_fingerprint(text)}"
        fresh = key not in self._served
        self._served.add(key)
        return Served(number=number, fresh=fresh)

    # ---------------------------------------------------------------- 回解
    def resolve(self, n: int) -> str | None:
        """句柄 → chunk_id；越界或未发号返回 ``None``。"""
        if not isinstance(n, int) or n < 0 or n >= len(self._order):
            return None
        return self._order[n]

    def ids(self) -> list[str]:
        """首见序的快照：``ids()[n]`` 就是模型看到的 ``[ID:n]``。"""
        return list(self._order)

    def text_of(self, n: int) -> str | None:
        return self._text_of.get(n)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._num_of
