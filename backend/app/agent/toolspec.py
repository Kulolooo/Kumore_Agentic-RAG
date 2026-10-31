"""四个工具的 OpenAI 兼容定义。

**只有形状，没有实现**：``tools.py`` 负责执行。分开是因为闸门 C（工具调用稳定性）
要拿真正上线的定义去测——两处各写一份，测出来的稳定率就不代表上线行为。

设计约束（都来自已观测的失败，不是"业界惯例"）：

1. **``search_kb`` 带 ``scope``**：闸门 A 实测 q31 的缺失金标与命中的金标同属一个版本族，
   ``max_per_family=3`` 的限流把它挡在 top-5 外，解除限流后排第 4——所以第二步必须
   有办法把检索**收窄到某一份文档**，否则重复检索只会拿回同样的 5 条。
2. **``grep_kb`` 是字面匹配，不是语义检索**：q28 的两份金标标题都叫 ``README``，
   标题无法区分，但正文里的表名（``100836 ratings``）能。字面匹配是"我已经知道要找什么"
   时最可靠的动作，也是闸门 A 里唯一可能救回这类题的路径。
3. **``read_section`` 按文档读**：闸门 A 的 q25 是"文档进了 top-5，但那一块不含证据串"，
   这时再检索没用，只能往同一份文档的相邻块读下去。
4. **``list_docs`` 用来消歧**：模型需要知道"这个主题下还有哪些文档"，才能决定
   ``scope`` 填什么。它**不返回标题、不返回路径**（见 ``tools.py`` 的隐私说明）。

参数一律用扁平标量 + 明确默认值：嵌套对象会显著抬高模型写错 JSON 的概率，而
"arguments 合法率"是闸门 C 的硬指标。
"""

from __future__ import annotations

from typing import Any

#: 单次工具返回的正文上限（字符）。检索结果太长会把"下一轮"的上下文吃光，
#: 而且长结果对模型判断"够不够"没有帮助——它只需要知道自己读到了什么。
TOOL_TEXT_MAX = 1200

#: 一次 ask 最多发号多少个证据块。超过后新证据只放正文、不再发号（避免编号三位数）。
MAX_EVIDENCE = 40

SEARCH_KB = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "在知识库里按关键词检索相关片段，返回带编号的原文片段。"
            "这是主要的检索方式，可以用不同措辞多检索几次。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词，用自然语言短语或关键词都行"},
                "top_k": {
                    "type": "integer",
                    "description": "返回条数，默认 5，最多 10",
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "可选。只在这份文档里检索，填 list_docs / 检索结果里看到的文档代号（如 D3）。"
                        "用于确知答案在某一份文档里、但检索总被别的文档挤掉时。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

GREP_KB = {
    "type": "function",
    "function": {
        "name": "grep_kb",
        "description": (
            "在知识库全文里按字面查找一段字符串（区分大小写不敏感，忽略空格与换行差异）。"
            "适合已经知道要找的确切数字、型号、专有名词时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "要逐字查找的字符串"},
                "max_hits": {
                    "type": "integer",
                    "description": "最多返回多少处匹配，默认 5，最多 10",
                },
            },
            "required": ["pattern"],
        },
    },
}

READ_SECTION = {
    "type": "function",
    "function": {
        "name": "read_section",
        "description": (
            "读取某一份文档的更多正文。填文档代号（如 D3）时从头读；"
            "再给 chunk 编号（如 ID:7）时从那一块附近读。用于"
            "\"这份文档里应该有，但检索只给了其中一块\"的情况。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc": {"type": "string", "description": "文档代号，如 D3"},
                "near": {
                    "type": "string",
                    "description": "可选。从这一块附近开始读，填 [ID:n] 或 ID:n 里的 n",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多读几块，默认 3，最多 6",
                },
            },
            "required": ["doc"],
        },
    },
}

LIST_DOCS = {
    "type": "function",
    "function": {
        "name": "list_docs",
        "description": (
            "列出知识库里的文档（代号 + 开头一小段正文），可按关键词筛选。"
            "用来了解有哪些资料、或在同名文档之间消歧。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "可选。只列正文含这个词的文档"},
                "limit": {"type": "integer", "description": "最多列几份，默认 10，最多 20"},
            },
            "required": [],
        },
    },
}

#: 顺序即展示顺序。``search_kb`` 必须排第一：首轮强制调用的就是它。
TOOL_SPECS: list[dict[str, Any]] = [SEARCH_KB, GREP_KB, READ_SECTION, LIST_DOCS]

TOOL_NAMES: tuple[str, ...] = tuple(
    spec["function"]["name"] for spec in TOOL_SPECS
)

#: 首轮强制调用的工具名（抄 RAGFlow 的"先检索再说话"，`retrieval_model.go:93-99`）
FIRST_TOOL = "search_kb"

#: 各工具参数的默认值与上限。执行层用它把模型给的数字收进安全区，
#: 避免 ``top_k=10000`` 这种把上下文打爆的调用。
LIMITS = {
    "search_kb": {"top_k": (5, 10), "max": 10},
    "grep_kb": {"max_hits": (5, 10), "max": 10},
    "read_section": {"limit": (3, 6), "max": 6},
    "list_docs": {"limit": (10, 20), "max": 20},
}


def forced(name: str = FIRST_TOOL) -> dict[str, Any]:
    """构造"强制调用某工具"的 ``tool_choice``。

    ``"required"`` 只保证"一定会调某个工具"，不保证调的是哪个；首轮要求必须是
    ``search_kb``，所以这里用具名形态。
    """
    if name not in TOOL_NAMES:
        raise ValueError(f"未知工具 {name!r}，可用 {TOOL_NAMES}")
    return {"type": "function", "function": {"name": name}}
