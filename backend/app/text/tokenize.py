"""中文/英文混合分词。

为什么不用 jieba：M0 的目标是拿到一个**能失败、能复现**的基线。字符 n-gram 的切分
规则完全确定，不依赖词典版本，任何人在任何机器上跑出的 BM25 分数都一致；装一个
词典分词器会引入"词典不同 → 结果不同"的变量，让后续对照实验说不清是谁的功劳。

规则：
  1. NFKC 归一化，全角数字/字母转半角（中文标点保留）。
  2. 连续 ASCII 字母数字作为一个词，转小写。
  3. 连续汉字串切 1-gram 和 2-gram 都入索引：
     - 只留 2-gram 时，单字查询（如"泵"）匹配不到"水泵"这样的 2-gram；
     - 只留 1-gram 又丢了"水泵"这类短语的整体性。
     两者都收，单字由 IDF 自然降权，是中文检索里常见且可解释的折中。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

#: 第一组匹配 ASCII 词，第二组匹配汉字串。其余字符（标点、空白、符号）都是分隔符。
_TOKEN_RE = re.compile(
    r"([A-Za-z0-9]+)"  # 英文、数字
    r"|([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)"  # 汉字（含扩展 A 与兼容区）
)


def normalize(text: str) -> str:
    """NFKC 归一化：全角转半角，兼容字符合并。"""
    return unicodedata.normalize("NFKC", text or "")


def _cjk_grams(run: str) -> list[str]:
    grams = list(run)
    if len(run) > 1:
        grams.extend(run[i : i + 2] for i in range(len(run) - 1))
    return grams


def tokenize(text: str) -> list[str]:
    """切词并保留重复（BM25 需要词频）。"""
    tokens: list[str] = []
    for ascii_run, cjk_run in _TOKEN_RE.findall(normalize(text)):
        if ascii_run:
            tokens.append(ascii_run.lower())
        elif cjk_run:
            tokens.extend(_cjk_grams(cjk_run))
    return tokens


def term_frequencies(text: str) -> Counter[str]:
    return Counter(tokenize(text))
