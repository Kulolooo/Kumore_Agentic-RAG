"""金标集读写。

字段里用**相对路径**而不是内部 doc_id：doc_id 是路径的哈希，人看不懂，也没法 review。
写题的人只需要知道"答案在哪份文件的哪一页"，路径能直接在磁盘上核对，
这样才谈得上"题目可被证伪"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

#: 允许的题型。加新题型要同时改这里，避免报告里出现无法汇总的类别。
CATEGORIES = (
    "单跳事实",
    "同义改写",
    "跨文档",
    "版本冲突",
    "无答案",
    "追问",
)


@dataclass
class GoldItem:
    qid: str
    question: str
    category: str
    answerable: bool
    gold_rel_paths: list[str] = field(default_factory=list)
    gold_answer: str | None = None
    #: 必须能在被引用片段里原样找到的字符串，用来抽查引用是否真的支持结论
    gold_spans: list[str] = field(default_factory=list)
    #: 证据所在页（可选）。有它才能检查"分块后证据是否完整落在同一块"，
    #: 这也是 `docs/逆向/08-对标复刻清单.md:29` 建议加的那一列。
    page: int | None = None
    #: 证据跨多页时的页码集合（q22 的计分规则横跨 7–14 页）。与 ``page`` 取并集；
    #: 旧题只写 ``page``，新题可以只写这个。
    gold_pages: list[int] = field(default_factory=list)
    #: 仅无答案题用：这些词在全库必须出现 0 次，否则"无答案"这个前提本身不成立（P27）。
    evidence_absent: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "category": self.category,
            "answerable": self.answerable,
            "gold_rel_paths": self.gold_rel_paths,
            "gold_answer": self.gold_answer,
            "gold_spans": self.gold_spans,
            "page": self.page,
            "gold_pages": self.gold_pages,
            "evidence_absent": self.evidence_absent,
            "notes": self.notes,
        }


def load_gold(path: Path) -> list[GoldItem]:
    if not path.is_file():
        raise FileNotFoundError(f"金标集不存在：{path}")

    items: list[GoldItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc

        qid = str(row.get("qid") or f"q{lineno:02d}")
        if qid in seen:
            raise ValueError(f"金标集 qid 重复：{qid}（第 {lineno} 行）")
        seen.add(qid)

        category = str(row.get("category", "")).strip()
        if category not in CATEGORIES:
            raise ValueError(f"{qid} 的 category={category!r} 不在 {CATEGORIES} 里")

        answerable = bool(row.get("answerable", category != "无答案"))
        gold_paths = list(row.get("gold_rel_paths") or [])
        if answerable and not gold_paths:
            raise ValueError(f"{qid} 标为可回答，却没有 gold_rel_paths")

        items.append(
            GoldItem(
                qid=qid,
                question=str(row["question"]).strip(),
                category=category,
                answerable=answerable,
                gold_rel_paths=gold_paths,
                gold_answer=row.get("gold_answer"),
                gold_spans=list(row.get("gold_spans") or []),
                page=row.get("page"),
                gold_pages=[
                    int(p) for p in (row.get("gold_pages") or []) if str(p).isdigit()
                ],
                evidence_absent=list(row.get("evidence_absent") or []),
                notes=row.get("notes"),
            )
        )
    return items


def iter_lines(items: list[GoldItem]) -> Iterator[dict[str, Any]]:
    for item in items:
        yield item.to_dict()
