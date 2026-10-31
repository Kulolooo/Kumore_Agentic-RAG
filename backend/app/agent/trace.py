"""一次 ask 的步骤轨迹（双通道渲染的数据源）。

为什么要有它：M4 的验收第一条是"两跳题的每一步可复现"，第二条是"每次拒答能判为
'没找到'还是'判定不充分'"。没有逐步记录，这两条都只能靠事后读回答猜。

字段口径（**先定口径再看数字**）：

===========  ==========================================================
step         步号，从 1 开始；一次工具调用算一步
tool         工具名（search_kb / grep_kb / read_section / list_docs）；
             非工具轮为 ``sufficiency`` / ``claims`` / ``finalize``
args         模型给出的参数（已回填成 dict；非法 JSON 原样保留在 args_raw）
ok           这一步是否成功（工具异常、参数非法、模型异常都是 False）
n_hits       本步新增的**证据块**数（去重后新发号的条数）
hit_ids      本步新发号的 ``[ID:n]`` 编号
ms           本步耗时（工具执行时长或模型调用时长）
tokens       本步消耗的 prompt + completion
decision     continue / finalize / refuse（这一步结束时循环的判定）
detail       自由文本：错误信息、判定缺失项、降级原因等
===========  ==========================================================

JSONL 落地：一次 ask 一个文件，每行一步。**不写**正文全文（正文已在评测的
``context`` 里），只写足以复现"为什么走这一步"的摘要。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

DECISIONS = ("continue", "finalize", "refuse")


@dataclass
class Step:
    step: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    #: 原始 arguments 字符串：非法 JSON 时 args 为空，靠它事后复盘模型写了什么
    args_raw: str = ""
    ok: bool = True
    n_hits: int = 0
    hit_ids: list[int] = field(default_factory=list)
    #: 本步新发号的块来自哪条召回腿（bm25 / vector / bm25+vector / grep / read）。
    #: 归因维度靠它：只记块号回答不了"这一步是词法还是向量召回的"。
    sources: list[str] = field(default_factory=list)
    ms: float | None = None
    tokens: int | None = None
    decision: str = "continue"
    detail: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision 只能是 {DECISIONS}，当前 {self.decision!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "args_raw": self.args_raw,
            "ok": self.ok,
            "n_hits": self.n_hits,
            "hit_ids": list(self.hit_ids),
            "sources": list(self.sources),
            "ms": self.ms,
            "tokens": self.tokens,
            "decision": self.decision,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Step":
        return cls(
            step=int(data.get("step") or 0),
            tool=str(data.get("tool") or ""),
            args=dict(data.get("args") or {}),
            args_raw=str(data.get("args_raw") or ""),
            ok=bool(data.get("ok", True)),
            n_hits=int(data.get("n_hits") or 0),
            hit_ids=[int(n) for n in (data.get("hit_ids") or [])],
            sources=[str(s) for s in (data.get("sources") or [])],
            ms=data.get("ms"),
            tokens=data.get("tokens"),
            decision=str(data.get("decision") or "continue"),
            detail=str(data.get("detail") or ""),
        )


@dataclass
class Trace:
    """一次 ask 的完整轨迹。"""

    question: str
    steps: list[Step] = field(default_factory=list)
    #: 终局：answer / refuse / degraded
    outcome: str = ""
    #: 拒答归因：no_hit（没找到）/ judged_insufficient（找到了但判定不充分）/ None
    refusal_reason: str | None = None
    #: 依赖失败标记（检索 / 模型 / judge 任一失败）：非空表示这次是降级交付
    degradations: list[str] = field(default_factory=list)
    #: 累计 token（按 turn 结算）
    tokens_used: int = 0
    #: 证据注册表最终有多少块
    n_evidence: int = 0
    started_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )
    #: M6 展示层的实时回调：每追加一步就调一次（CLI 与评测不设置，行为不变）。
    #: **不是**业务逻辑，异常一律吞掉——UI 卡住不能把一次已经付费的问答打断。
    on_step: Callable[[Step], None] | None = field(
        default=None, repr=False, compare=False
    )

    # ---------------------------------------------------------------- 记账
    def add(self, step: Step) -> Step:
        self.steps.append(step)
        if step.tokens:
            self.tokens_used += step.tokens
        if self.on_step is not None:
            try:
                self.on_step(step)
            except Exception:
                # 展示层回调的异常不能影响问答主链路（也不写进轨迹，避免污染评测）。
                pass
        return step

    def degrade(self, reason: str) -> None:
        """记一次依赖失败。同一原因只记一次，避免重试把轨迹刷满。"""
        if reason and reason not in self.degradations:
            self.degradations.append(reason)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tool_steps(self) -> list[Step]:
        return [s for s in self.steps if s.tool in ("search_kb", "grep_kb", "read_section", "list_docs")]

    def ids_used(self) -> list[int]:
        """所有步发出过的编号（去重保序）。"""
        seen: list[int] = []
        for step in self.steps:
            for number in step.hit_ids:
                if number not in seen:
                    seen.append(number)
        return seen

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "outcome": self.outcome,
            "refusal_reason": self.refusal_reason,
            "degradations": list(self.degradations),
            "tokens_used": self.tokens_used,
            "n_evidence": self.n_evidence,
            "n_steps": self.n_steps,
            "started_at": self.started_at,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        trace = cls(
            question=str(data.get("question") or ""),
            outcome=str(data.get("outcome") or ""),
            refusal_reason=data.get("refusal_reason"),
            degradations=list(data.get("degradations") or []),
            tokens_used=int(data.get("tokens_used") or 0),
            n_evidence=int(data.get("n_evidence") or 0),
            started_at=str(data.get("started_at") or ""),
        )
        trace.steps = [Step.from_dict(row) for row in data.get("steps") or []]
        return trace

    def to_markdown(self) -> str:
        """双通道的"研究过程"侧：给人看的逐步表格。"""
        lines = [
            f"问题：{self.question}",
            "",
            "| 步 | 动作 | 参数 | 新增证据 | 耗时 ms | tokens | 判定 | 说明 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for step in self.steps:
            args = json.dumps(step.args, ensure_ascii=False) if step.args else (step.args_raw or "")
            args = args if len(args) <= 60 else args[:59] + "…"
            detail = step.detail if len(step.detail) <= 60 else step.detail[:59] + "…"
            lines.append(
                f"| {step.step} | {step.tool} | {args} | {step.n_hits} | "
                f"{step.ms if step.ms is not None else ''} | "
                f"{step.tokens if step.tokens is not None else ''} | "
                f"{step.decision} | {detail} |"
            )
        lines.append("")
        lines.append(
            f"终局：{self.outcome}｜累计 {self.tokens_used} tokens｜"
            f"证据 {self.n_evidence} 块｜降级 {self.degradations or '无'}"
        )
        return "\n".join(lines)

    # ---------------------------------------------------------------- 落盘
    def save(self, path: Path) -> Path:
        """写成 JSONL：第 1 行是头部（问题/终局/汇总），其后每行一步。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {key: value for key, value in self.to_dict().items() if key != "steps"}
        header["type"] = "header"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for step in self.steps:
                row = step.to_dict()
                row["type"] = "step"
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "Trace":
        """读回一份轨迹。头部缺失时按"全是 step 行"处理。"""
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        header = next((row for row in rows if row.get("type") == "header"), None)
        steps = [row for row in rows if row.get("type") == "step" or "step" in row and "tool" in row]
        if header is None:
            header = {"question": "", "steps": steps}
        else:
            header = {**header, "steps": steps}
        return cls.from_dict(header)

    @classmethod
    def iter_file(cls, path: Path) -> Iterator[dict[str, Any]]:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
