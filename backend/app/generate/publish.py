"""引用发布：把"展示过的块"压实成"被引用的块"。

抄的是 RAGFlow 的 ``ResolveCitations``（``_ref/ragflow-main/internal/agentic_rag/
citations.go:207-245``）：run 级句柄（``[ID:265]``）不该直接交给前端，答案里引用到的
段落按**首引序**重编号成 ``[ID:0]``、``[ID:1]``…，发布清单只含被引的块；没被引的块
不进 ``reference.chunks``。

为什么这对我们有用（两条都指到已观测的问题）：

1. **发布面 = 被引面**。M3 起我们的 ``[ID:n]`` 是"展示序"，于是"上下文里有 30 块"
   和"答案引用了 6 块"在前端看起来是同一个列表；压实之后两者分开，前端只拿被引块。
2. **首引序对读者更友好**。答案里第一个出现的引用是 ``[ID:0]``，读者从上往下读
   编号就是单调的——RAGFlow 的注释把这叫 "mechanical compaction"。

**这不是去重**：同一个句柄出现多次仍是同一个号（首见定号），与 ``EvidenceRegistry``
的"同 id 同号"一致。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 机械契约句柄。只处理 ``[ID:n]``：agent 路的提示词只写这一种写法，
#: 过渡期写法 ``[n]`` 在这里当正文处理（M3 的兼容路径在 ``answer.py`` 里，不在这里）。
_HANDLE_ID = re.compile(r"\[ID:(\d{1,3})\]")


@dataclass
class Published:
    """压实后的答案与发布清单。"""

    #: 重编号后的答案正文
    text: str
    #: 新号 → 旧号（``order[0]`` 是第一个被引的旧号）
    order: list[int] = field(default_factory=list)
    #: 旧号 → 新号（同一个旧号永远同一个新号）
    mapping: dict[int, int] = field(default_factory=dict)
    #: 重编号时丢弃的越界句柄（正常路径下为空，sanitize 已经先删过）
    dropped: list[int] = field(default_factory=list)

    @property
    def n_references(self) -> int:
        return len(self.order)


def publish_by_citation(text: str, n_context: int) -> Published:
    """把 ``[ID:n]`` 按**首引序**重编号；越界编号删掉并记账。

    ``n_context`` 是"候选块的数量"（0..n_context-1 合法）。越界句柄在这里被删除而不是
    留一个错号——与 ``citations.go:213-215`` 的 "dropped rather than left pointing at a
    wrong index" 同一条规则，也和 ``answer.py`` 的 ``sanitize_answer`` 一致；
    保留这一层是为了让本函数单独可测（即便上游已经清洗过）。
    """
    text = text or ""
    mapping: dict[int, int] = {}
    order: list[int] = []
    dropped: list[int] = []
    pieces: list[str] = []
    cursor = 0
    for match in _HANDLE_ID.finditer(text):
        number = int(match.group(1))
        if number < 0 or number >= n_context:
            # 越界：连同它前面的正文一起保留，只删掉这个标记
            dropped.append(number)
            pieces.append(text[cursor : match.start()])
            cursor = match.end()
            continue
        if number not in mapping:
            mapping[number] = len(order)
            order.append(number)
        pieces.append(text[cursor : match.start()])
        pieces.append(f"[ID:{mapping[number]}]")
        cursor = match.end()
    pieces.append(text[cursor:])
    return Published(
        text="".join(pieces).strip(), order=order, mapping=mapping, dropped=dropped
    )
