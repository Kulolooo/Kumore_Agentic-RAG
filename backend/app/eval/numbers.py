"""数字与量纲：把"八万五千元"和"85000元"认成同一个事实。

M3 的句级校验器只比阿拉伯数字，于是"C-3 中文数字漏检"成立：正文写"八万五千元"、
陈述写"85000 元"时判不出；反过来陈述照抄中文数字、正文却是阿拉伯数字时会被判不支持。
这个模块补两件事：

1. **中文数字 → 数值**：支持 零〇一二两三四五六七八九十百千万亿 的常见组合
   （"八万五千"、"三千七百七十"、"二十"）。不追求覆盖罕见写法（"壹拾"、大写金额），
   遇到解析不了的整串直接放弃——漏检一次比误判一次便宜。
2. **单位族**：把 `W/瓦/千瓦`、`m/米/厘米`、`%/百分比` 归到同一族，比较时只看
   "值 + 族"。族是**语言特征**，不是语料特征，换领域同样成立；族表写死在下面，
   要扩就加别名，不要加语料专名。

还有一处配套：**引用型数字不算事实**。模型爱写"第 7 页资料给出…""资料[3] 里写…"，
这些数字来自我们的展示顺序，不在正文里，硬比会制造假阴性。``strip_references``
把"第 X 页""第 X–Y 页""第 X 项/条/章/节""ID:n"这类引用片段先摘掉再比。
"""

from __future__ import annotations

import re

#: 零宽字符：提示词渲染时用它把正文里的 ``[ID:n]`` 变体隔开（见 prompt._neutralize_handles），
#: 匹配时一律忽略，否则"模型照抄了零宽字符的引文"会定位失败。
ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_BIG_UNITS = {"万": 10_000, "亿": 100_000_000}

#: 单位别名 → 族。长的别名必须排在前面（"千瓦" 先于 "瓦"、"万元" 先于 "元"）。
_UNIT_ALIASES: dict[str, str] = {
    "千瓦": "power", "兆瓦": "power", "kw": "power", "mw": "power", "瓦": "power", "w": "power",
    "厘米": "length", "毫米": "length", "千米": "length", "公里": "length",
    "cm": "length", "mm": "length", "km": "length", "m": "length", "米": "length",
    "千克": "mass", "公斤": "mass", "kg": "mass", "克": "mass", "g": "mass",
    "吨": "mass", "t": "mass",
    "百分比": "ratio", "%": "ratio", "％": "ratio", "倍": "multiple",
    "分钟": "time", "小时": "time", "min": "time", "秒": "time", "s": "time", "h": "time",
    "万元": "money", "亿元": "money", "元": "money", "块": "money", "万": "money", "亿": "money",
    "个": "count", "项": "count", "种": "count", "类": "count", "条": "count",
    "份": "count", "套": "count", "台": "count", "人": "count", "次": "count",
    "度": "angle", "°": "angle", "℃": "temp",
}

#: 按长度倒序拼成的单位正则（保证"千瓦"先于"瓦"被匹配）。
#: 末尾的 ``(?![A-Za-z])`` 是必需的：否则 "17.56MB" 里的 "M" 会被认成"米"。
_UNIT_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(_UNIT_ALIASES, key=len, reverse=True)
)

_CN_NUMBER = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")
_ARABIC = re.compile(r"\d+(?:\.\d+)?")
_UNIT_AFTER = re.compile(rf"\s*(?P<unit>{_UNIT_PATTERN})(?![A-Za-z])", re.IGNORECASE)
#: 数字与单位之间的量级词："3770**万**千瓦"＝3770×万 千瓦。中文数字里 万/亿 已经被
#: 数词解析吃掉（"三千七百七十万"），这里只服务阿拉伯数字写法。
_SCALE_AFTER = re.compile(r"\s*(?P<scale>[万亿])")

#: 引用型片段：先摘掉再提数字（它们不是事实，比较只会制造假阴性）。
_REFERENCE = re.compile(
    r"(?:第\s*\d+(?:\s*[-–~至]\s*\d+)?\s*[页頁])"
    r"|(?:第\s*[\d一二三四五六七八九十百]+\s*[份条项章节点页節])"
    r"|(?:ID\s*[:：]\s*\d+)"
    r"|(?:\[\s*ID\s*[:：]\s*\d+\s*\])"
)


def _cn_to_value(text: str) -> int | None:
    """把中文数字串转成整数；解析不了返回 None。

    算法：按 亿/万 分三段累加，段内按 千/百/十 累加，末尾补个位。
    "八万五千" → 85000；"三千七百七十" → 3770；"二十" → 20；"十" → 10。
    """
    if not text:
        return None
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
            continue
        if char in _CN_SMALL_UNITS:
            # "十"单独出现表示 10（"十五"=15）
            section += (number or 1) * _CN_SMALL_UNITS[char]
            number = 0
            continue
        if char in _CN_BIG_UNITS:
            section += number
            total += (section or 1) * _CN_BIG_UNITS[char]
            section = 0
            number = 0
            continue
        return None
    return total + section + number


def strip_references(text: str) -> str:
    """摘掉"第 X 页""第 X 项""ID:n"这类引用片段（不是事实数字）。"""
    return _REFERENCE.sub(" ", text or "")


def _unit_family(unit: str | None) -> str:
    if not unit:
        return ""
    return _UNIT_ALIASES.get(unit.lower(), _UNIT_ALIASES.get(unit, ""))


def _suffix_after(source: str, position: int) -> tuple[float, str]:
    """数字后面的"量级 + 单位"，返回 ``(量级, 单位族)``。

    ``"3770万千瓦"`` → ``(10000, "power")``；``"115亿"`` → ``(1e8, "money")``；
    ``"2024年"`` → ``(1, "count")``；什么都没有 → ``(1, "")``。
    """
    scale = 1.0
    cursor = position
    scale_match = _SCALE_AFTER.match(source, cursor)
    if scale_match is not None:
        scale = 10_000 if scale_match.group("scale") == "万" else 100_000_000
        cursor = scale_match.end()
        unit_match = _UNIT_AFTER.match(source, cursor)
        if unit_match is not None:
            return scale, _unit_family(unit_match.group("unit"))
        # "115 亿" 后面没跟具体单位：当作金额量级
        return scale, "money"
    unit_match = _UNIT_AFTER.match(source, cursor)
    return scale, _unit_family(unit_match.group("unit") if unit_match else None)


def _trim_cn_unit_tail(source: str, raw: str, end: int) -> tuple[str, int]:
    """把"被数词吃掉的单位首字"还回去。

    ``"三千七百七十万千瓦"`` 里，数词正则会贪到 ``"……七十千"``（千也在数词字符集里），
    但这里的"千"属于单位"千瓦"。规则：数词末尾若是 十/百/千，且它加上后面的文本
    能组成一个单位，就把这个字还回去。``"三万千米"`` → 数词 "三万"、单位 "千米"。
    """
    while raw and raw[-1] in _CN_SMALL_UNITS:
        if _UNIT_AFTER.match(raw[-1] + source[end:]) is None:
            break
        raw = raw[:-1]
        end -= 1
    return raw, end


def numeric_tokens(text: str, *, keep_references: bool = False) -> set[tuple[float, str]]:
    """句子里的数字，返回 ``{(值, 单位族)}``。

    * 阿拉伯数字：``"2160mm"`` → ``(2160.0, "length")``；无单位 → ``(2160.0, "")``。
    * 中文数字：只有**长度 ≥2 或后面跟单位**时才收（单字"三"太常见，收进来全是噪声）。
    * 值相同的整数与小数视为同一个（``20`` 与 ``20.0``）。
    """
    if not text:
        return set()
    source = text if keep_references else strip_references(text)
    tokens: set[tuple[float, str]] = set()

    for match in _ARABIC.finditer(source):
        scale, family = _suffix_after(source, match.end())
        tokens.add((float(match.group(0)) * scale, family))

    for match in _CN_NUMBER.finditer(source):
        # 前面紧挨着数字的汉字数词是单位的一部分（"3770万千瓦" 的 "万千"），跳过
        if match.start() > 0 and (source[match.start() - 1].isdigit() or source[match.start() - 1] == "."):
            continue
        raw, end = _trim_cn_unit_tail(source, match.group(0), match.end())
        value = _cn_to_value(raw)
        if value is None:
            continue
        scale, family = _suffix_after(source, end)
        # 中文数词已经吃过 万/亿；只有在它后面又跟了单位时才可能再吃一次量级，
        # 但那时数词本身不含 万/亿（"三千七百七十" + "万千瓦"），乘上去正好。
        if len(raw) < 2 and not family:
            continue
        if value == 1 and family == "count":
            # "另一份资料""一个方法""一方面"里的"一"是话语标记，不是数量事实。
            # 这类写法在中文里几乎总是指代而不是计数，收进一致性检查只会制造假阴性。
            continue
        tokens.add((float(value) * scale, family))

    return tokens


def token_label(token: tuple[float, str]) -> str:
    """给报告用的字面：``(85000.0, "money")`` → ``"85000(money)"``。"""
    value, family = token
    shown = int(value) if float(value).is_integer() else value
    return f"{shown}({family})" if family else str(shown)


def missing_tokens(wanted: set[tuple[float, str]], available: set[tuple[float, str]]) -> list[tuple[float, str]]:
    """句子里对不上的数字。

    比对规则：同值且（单位族相同 **或** 任一侧没写单位）。"20 W" 与 "20 瓦" 同族，
    "20" 与 "20 kg" 也算对得上——句子省略单位是常态，不是错。
    """
    missing: list[tuple[float, str]] = []
    for value, family in sorted(wanted):
        if (value, family) in available:
            continue
        if (value, "") in available and family:
            continue
        if family and any(v == value and not f for v, f in available):
            continue
        if any(v == value for v, f in available) and not family:
            continue
        missing.append((value, family))
    return missing
