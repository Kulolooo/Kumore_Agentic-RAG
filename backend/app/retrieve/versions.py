"""版本分组：把"同一份文档的不同版本"归到一起。

M0 的实测缺陷是「版本冲突静默选一版」：问「每年定向吸纳多少人」，语料里
`海丰戍卫.pdf` 写 8 名、`海丰戍卫-校赛版本.pdf` 写 3 名，模型直接答了 8 名、
不提还有另一版。根因是上下文里两个片段看起来一样权威。

修法分两层：
  1. 这里给每份文档算出「版本族」和「版本标签」（如 `校赛版本`、`规整版`、`.pdf`）；
  2. 提示词在检测到同族多版本同时命中时，显式要求"给出不同说法并分别引用"。

刻意用**保守的命名启发式**而不是猜内容差异：
只剥离文件名里的版本标记词（-校赛版本 / （规整版） / -副本 / -无信息 …），
再用 `(父目录, 归一化标题)` 分组。宁可漏判（当成两个不相关文档），不可错判
（把两份不同文档硬说成同一份的不同版本，那会让模型主动编出"差异"）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

#: 出现在文件名末尾、表示"这是同一份东西的另一个版本/副本"的标记。
#: 顺序无关，会反复剥离直到无法再剥。
_VERSION_MARKERS: tuple[str, ...] = (
    "-校赛版本",
    "校赛版本",
    "（规整版）",
    "(规整版)",
    "规整版",
    "-副本",
    " - 副本",
    "副本",
    "（无个人信息）",
    "(无个人信息)",
    "无个人信息",
    "-无信息",
    "无信息",
    "【排版结果】",
    "-三创",
    "三创",
    "（1）",
    "(1)",
    "（2）",
    "(2)",
    "（3）",
    "(3)",
)

_TRAILING = re.compile(r"[-_\s·—]+$")

#: 只在**确实是一次新发布**时才提示冲突。像「规整版」「无个人信息」「副本」「.pdf/.docx」
#: 这类只是排版、脱敏或格式不同，内容通常一致；对它们发冲突警告会让模型无谓地变谨慎
#: （实测：一条误报的横幅能把「直接答 8 人」变成 4/4 拒答）。
_RELEASE_MARKERS: tuple[str, ...] = ("校赛版本",)


def variant_kind(title: str) -> str:
    """``release`` = 可能是内容不同的新版本；``format`` = 只是排版/格式/脱敏差异。"""
    text = (title or "").strip()
    return "release" if any(text.endswith(m) for m in _RELEASE_MARKERS) else "format"


def strip_version_markers(title: str) -> str:
    """剥掉版本标记，得到"这份文档是什么"的名字。"""
    text = (title or "").strip()
    changed = True
    while changed:
        changed = False
        for marker in _VERSION_MARKERS:
            if text.endswith(marker):
                text = text[: -len(marker)].strip()
                changed = True
        trimmed = _TRAILING.sub("", text)
        if trimmed != text:
            text = trimmed
            changed = True
    return text or (title or "").strip()


@dataclass(frozen=True)
class VersionInfo:
    """一份文档在它所属版本族里的位置。"""

    group_key: str  # (父目录, 归一化标题)，用于分组
    base_name: str  # 展示用：文档本名
    label: str  # 展示用：区别于同族其他成员的标签
    siblings: tuple[str, ...] = field(default_factory=tuple)  # 同族其他文档的 rel_path
    #: release = 可能内容不同的新版本；format = 仅排版/格式/脱敏差异
    kind: str = "format"

    @property
    def is_multi_version(self) -> bool:
        return bool(self.siblings)


def _version_label(title: str, base_name: str, ext: str) -> str:
    """从标题里剥出"区别于本名的部分"；剥不出就用扩展名。"""
    rest = (title or "").strip()
    if base_name and rest.startswith(base_name):
        rest = rest[len(base_name) :]
    rest = rest.strip()
    rest = rest.lstrip("-_·—  ")
    rest = rest.strip()
    # 去掉包裹的括号，如「（规整版）」
    while len(rest) >= 2 and rest[0] in "（(" and rest[-1] in "）)":
        rest = rest[1:-1].strip()
    rest = _TRAILING.sub("", rest)
    return rest or ext or "原始版"


def build_version_map(docs) -> dict[str, VersionInfo]:
    """只为"存在多个版本"的文档族生成映射；单版本文档不出现在结果里。

    参数是 ``Iterable[Doc]``；返回 ``{doc_id: VersionInfo}``。
    """
    grouped: dict[tuple[str, str], list] = {}
    for doc in docs:
        if not getattr(doc, "n_chunks", 0):
            continue  # 没进索引的文档不参与，否则会给"看不见的版本"发冲突警告
        rel = doc.rel_path
        parent = str(PurePosixPath(rel).parent)
        base = strip_version_markers(doc.title)
        grouped.setdefault((parent, base), []).append(doc)

    result: dict[str, VersionInfo] = {}
    for (parent, base), members in grouped.items():
        if len(members) < 2:
            continue
        for doc in members:
            others = tuple(m.rel_path for m in members if m.doc_id != doc.doc_id)
            result[doc.doc_id] = VersionInfo(
                group_key=f"{parent}/{base}",
                base_name=base,
                label=_version_label(doc.title, base, doc.ext),
                siblings=others,
                kind=variant_kind(doc.title),
            )
    return result
