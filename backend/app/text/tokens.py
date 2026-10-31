"""Token 估算：不装模型分词器，也要能让"512 token"这个参数有确定含义。

## 为什么不用 tiktoken / bge-m3 tokenizer

分块参数要在**任何机器上可复现**：同一份文本、同一组参数，切出来的 chunk 必须逐字相同。
模型 tokenizer 会随版本升级改变切分（P23 的教训：不可复现的结论不能当结论），
而分块一旦不可复现，M1 的 4 组对照就变成"换台机器换个数"。

所以这里用**确定性字符成本**估算：
  - 中日韩汉字：1 字 ≈ 1 token（cl100k 对中文的实测大体在这个量级）；
  - 全角标点：0.6；
  - 空白：0.1；
  - 其余（ASCII、数字、符号）：0.3 token/字符（约 3~4 字符 1 token）。

它**不是**任何模型的分词器，报告里必须写"估算 token"，不能写"模型 token"。
存在的意义是给"512"定一个稳定的量尺，而不是精确复刻某个模型的账单。
"""

from __future__ import annotations

import math


def _char_cost(ch: str) -> float:
    code = ord(ch)
    # 基本汉字 + 扩展 A + 兼容汉字
    if (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    ):
        return 1.0
    # CJK 标点（。、；：！？「」等）
    if 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFF65:
        return 0.6
    if ch.isspace():
        return 0.1
    return 0.3


def estimate_tokens(text: str) -> int:
    """估算 token 数。空文本返回 0，非空至少 1。"""
    if not text:
        return 0
    total = sum(_char_cost(ch) for ch in text)
    return max(1, math.ceil(total))


def char_offset_at_tokens(text: str, budget: int) -> int:
    """从 0 开始累加，返回 token 预算耗尽处的字符下标（至少前进 1）。"""
    if budget <= 0:
        return 0
    cost = 0.0
    for index, ch in enumerate(text):
        cost += _char_cost(ch)
        if cost > budget:
            return max(index, 1)
    return len(text)
