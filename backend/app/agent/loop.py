"""M4 的循环本体。

一发问号（turn）内的顺序，每条都对应 ``docs/路线图.md`` §M4 的一行：

1. **首轮强制检索**：``tool_choice`` 具名强制 ``search_kb``。理由不是"业界都这样"，
   而是一次真实失败：不强制时模型有概率直接凭上下文里没有的东西作答。
2. **充分性 JSON 判定**：``{"answerable": bool, "missing": str, ...}``，解析失败按
   "继续"处理（任务书 d 项）。**不是**散文——散文没法机械解析，等于没有门。
3. **claim 短路**：判定为足够之后，还要过一遍 M3 的逐字校验：每条 claim 的引文必须
   能在它指的块里逐字定位，且 claim 数覆盖问题的要义数。过不了就继续检索。
   这是"抄机制不抄规模"的核心适配（见 ``prompts.py`` 顶部）。
4. **逐段过滤**：本轮**不启用** judge（keepTopN / LLM 逐段打分）。理由：铁律 2
   （没有失败样本不加组件）——claim 覆盖检查是零成本的，先看它够不够；
   rerank 分数当门控的 AUC 0.88–0.90 是"可答/无答案"的区分度，不等于段落相关性。
   这条决定写进报告，M4 不冒充"实现了逐段过滤"。
5. **兜底合成**：独立预算、证据截断、``min_answer_chars`` 重试；任一依赖失败都返回
   降级答案并在轨迹里标记，进程退出码仍是 0。
6. **预算按 turn 结算**：``max_steps`` 与 ``max_tokens_budget`` 都在这里结算，
   超预算时"少检索一次、走短合成"。

终止条件（任一）：
- 短路成功 → 合成；
- 走到 ``max_steps``；
- 预算将超；
- 连续 ``max_unparsed_rounds`` 轮给不出契约 JSON；
- 模型调用失败（降级）。

为什么 ``max_steps`` 按"模型轮次"算而不是按"工具调用次数"算：花钱的是轮次，
一轮里并行两个工具调用只花一次 prompt。按工具次数算会惩罚并行，而并行正是
闸门 C 验证过的能力。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event
from typing import Any

from app.agent import toolspec
from app.agent.prompts import (
    degraded_answer,
    finalize_messages,
    finalize_retry_note,
    research_system,
    research_user,
    sufficiency_nudge,
    truncate_hits,
)
from app.agent.tools import ToolBox, ToolResult
from app.agent.trace import Step, Trace
from app.generate.answer import (
    _HANDLE_ID,
    _build_citations,
    extract_handles,
    is_refusal,
    sanitize_answer,
)
from app.generate.claim import Claim, parse_claims, verify_claims
from app.generate.client import (
    ChatResult,
    DeepSeekClient,
    LLMError,
    assistant_message,
    tool_message,
)
from app.generate.evidence import EvidenceRegistry
from app.generate.publish import publish_by_citation
from app.generate.prompt import REFUSAL_PREFIX
from app.retrieve.retriever import Retriever
from app.schemas import Answer, Retrieved

#: 中文问句里的"要义分隔符"。分号与问号是硬边界；逗号不切，避免把一句话切碎。
_CLAUSE_SPLIT = re.compile(r"[？?；;\n]+")

#: 疑问词。命中即认为这个分句是一个独立的信息需求。
_INTERROGATIVE = re.compile(r"什么|多少|几[个项次类种份篇条台套]|哪|是否|吗|如何|怎么|为什么|谁|多久|分别")

#: "分别/各"意味着一个分句里问了两件事（"票房和推荐分别用了什么数据"）。
#: 只按分句数会把它当成 1 个要点，覆盖判定就会漏。
_DISTRIBUTIVE = re.compile(r"分别|各自|各[一是]?")

#: 判定为"够用"时至少要有几条 claim。设 1 是因为单事实题只该有一条；
#: 多要点题由 ``coverage_gap`` 兜住，不在这里一刀切。
MIN_CLAIMS = 1


@dataclass
class AgentConfig:
    """M4 的预算与开关。默认值 = 路线图 §6 的参数总表。"""

    max_steps: int = 10
    max_tokens_budget: int = 120_000
    #: 兜底合成的独立预算（不计入检索预算，但要如实记账）
    finalize_budget_tokens: int = 40_000
    finalize_max_tokens: int = 2048
    #: 合成轮是否走流式。默认关闭：CLI/评测保持一次性返回；API/React 显式开启。
    #: 开启后只影响**最后一次合成**，研究轮仍是非流式 JSON/tool_calls。
    stream: bool = False
    min_answer_chars: int = 12
    #: 证据截断：超过就整块丢尾部
    evidence_max_chars: int = 40_000
    #: 每个工具步最多带几块进**合成**上下文（改动 1）。
    #: ``0`` = 不收紧（全部发号块进合成，M4 第一版的行为）；``2`` = 每步只要头部两块。
    #:
    #: **默认 0，因为实测收紧是净损失**（`storage/runs/m4_agent.md` §8.7）：
    #: 没有 judge 的收紧会把金标块当成尾巴剪掉——q12 的 BOM 块被剪，答案随即把
    #: CH340G 写成 CH340C。RAGFlow 敢剪是因为它有一个 typed-decision head 先判"这段够不够"，
    #: 我们只抄了 "keepTopN 豁免" 而没抄它的前置条件，这是"抄机制没核对前提"的直接教训。
    synthesis_per_step: int = 0
    top_k: int = 5
    prompt_variant: str = "claim"
    #: P69 修复实验：短路成功时，合成阶段只允许复述已核实的 claim，
    #: 不允许新增 claim/句柄。默认关闭——先证明它不伤可答率再开。
    verified_only: bool = False
    include_paths: bool = False
    include_superseded: bool = False
    temperature: float = 0.0
    #: 连续多少轮给不出契约 JSON 就收尾（每轮都会强制一次工具调用）
    max_unparsed_rounds: int = 2
    #: 研究轮的单轮 max_tokens；只用来输出 JSON 或工具调用，不需要很长
    research_max_tokens: int = 1024
    #: 单轮 prompt 预估上限（预算预判用），低于实际时预算会略晚触发
    min_round_tokens: int = 2_000


@dataclass
class SufficiencyVerdict:
    """充分性判定的解析结果。"""

    answerable: bool
    missing: str = ""
    next_query: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "missing": self.missing,
            "next_query": self.next_query,
            "claims": len(self.claims),
        }


@dataclass
class AgentResult:
    answer: Answer
    trace: Trace
    #: 按注册表顺序排好的命中（``[ID:n]`` 的 n 就是下标）——**全部**发过号的证据，
    #: 供轨迹回放与"claim 回指到步"使用
    hits: list[Retrieved] = field(default_factory=list)
    #: 按**首引序**排好的发布清单（只含答案真正引用的块）。评测的句级解析必须用它，
    #: 因为答案里的 ``[ID:n]`` 在发布后指的就是这份清单的下标。
    published: list[Retrieved] = field(default_factory=list)
    #: 真正进了合成上下文的块（⊆ 全部发号块）。"模型看到了什么"这类判定
    #: （例如 ``gold_span_in_context``）要用它，不能用发布清单——发布清单是被引子集。
    synthesis: list[Retrieved] = field(default_factory=list)
    #: 编号 → 来自哪一步哪一次工具调用（验收"每条 claim 能回指到某一步"）
    evidence: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[SufficiencyVerdict] = field(default_factory=list)
    #: 最后一次充分性判定给出的"还缺什么"
    last_missing: str = ""
    short_circuit: bool = False


# ---------------------------------------------------------------- 判定解析
def parse_verdict(text: str) -> SufficiencyVerdict | None:
    """从模型输出里抠出充分性 JSON。

    容忍三种真实污染：代码块围栏、JSON 前后有散文、尾部有逗号。**不做**语义补救——
    字段缺失或类型不对一律当"解析失败"，由循环按"继续"处理（任务书 d 项的容错方向
    是"多查一次"而不是"提前收尾"）。
    """
    raw = (text or "").strip()
    if not raw:
        return None
    # 去掉代码块围栏
    fenced = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw, flags=re.S).strip()
    candidate = fenced
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # 尾逗号是模型最常见的语法错，单独救一次；其它错误不救
        trimmed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            data = json.loads(trimmed)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict) or "answerable" not in data:
        return None
    answerable = data.get("answerable")
    if not isinstance(answerable, bool):
        if str(answerable).strip().lower() in {"true", "false"}:
            answerable = str(answerable).strip().lower() == "true"
        else:
            return None
    claims = data.get("claims")
    if claims is None:
        claims = []
    if not isinstance(claims, list):
        return None
    return SufficiencyVerdict(
        answerable=answerable,
        missing=str(data.get("missing") or "").strip(),
        next_query=str(data.get("next_query") or "").strip(),
        claims=[item for item in claims if isinstance(item, dict)],
        raw=raw,
    )


def _handles_from(value: Any) -> list[int]:
    """从 ``"[ID:0]"`` / ``"0"`` / ``[0, 3]`` 里取编号（0 基）。"""
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        found = [int(m.group(1)) for m in re.finditer(r"\[ID:(\d{1,3})\]", value)]
        if found:
            return found
        return [int(m.group(1)) for m in re.finditer(r"(?<!\d)(\d{1,3})(?!\d)", value)]
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            out.extend(_handles_from(item))
        return out
    return []


def claims_from_verdict(
    verdict: SufficiencyVerdict, registry: EvidenceRegistry
) -> list[Claim]:
    """把 JSON 里的 claims 转成 ``Claim``，复用 M3 的逐字校验。

    句柄字段名允许 ``handles`` 或 ``handle``；值允许 ``"[ID:0]"`` 或 ``0``——
    都是模型真实会写出来的形态，收窄只会制造解析失败。
    """
    claims: list[Claim] = []
    for item in verdict.claims:
        statement = str(item.get("statement") or item.get("claim") or "").strip()
        if not statement:
            continue
        evidence = str(item.get("evidence") or item.get("quote") or "").strip()
        raw_handles = item.get("handles", item.get("handle"))
        numbers = _handles_from(raw_handles)
        chunk_ids: list[str] = []
        for number in numbers:
            chunk_id = registry.resolve(number)
            if chunk_id is not None and chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
        claims.append(
            Claim(
                statement=statement,
                evidence=evidence,
                chunk_ids=chunk_ids,
                numbers=numbers,
                raw=json.dumps(item, ensure_ascii=False),
            )
        )
    return claims


def sub_question_count(question: str) -> int:
    """问题里有几个独立的信息需求（覆盖判定的分母）。

    只用中文疑问句的通用形态，没有为任何具体题目调参：

    1. 按 ``？?；;`` 切分，数含疑问词的分句；
    2. 每出现一次"分别/各自"再加一（一个分句里问了两件事）。

    它的作用是把"只答了一半"变成可机械检查的差数，**不是**精确理解题意；
    所以它只用于"要不要短路"这个软门，绝不用于判定答案对错。
    """
    parts = [part.strip() for part in _CLAUSE_SPLIT.split(question or "") if part.strip()]
    counted = sum(1 for part in parts if _INTERROGATIVE.search(part))
    base = max(1, counted)
    return base + len(_DISTRIBUTIVE.findall(question or ""))


def coverage_gap(question: str, claims: list[Claim]) -> int:
    """还差几条 claim 才算覆盖了问题的要义（0 = 覆盖）。"""
    return max(0, sub_question_count(question) - len(claims))


def _claims_ok(
    claims: list[Claim],
    report,
    gap: int,
) -> tuple[bool, str]:
    """claim 短路的判据。返回 (是否通过, 不通过的原因)。

    四条同时成立才短路：
    1. 至少一条 claim；
    2. 每条 claim 都给了逐字引文（``with_evidence == 条数``）；
    3. 每条引文都能在它指的块里逐字定位（``verbatim == with_evidence``）；
    4. claim 数覆盖问题要义数（``gap == 0``）。
    """
    if len(claims) < MIN_CLAIMS:
        return False, "没有给出任何 claim"
    if report.with_evidence < len(claims):
        return False, f"{len(claims) - report.with_evidence} 条 claim 没有给出逐字引文"
    if report.verbatim < report.with_evidence:
        return False, (
            f"{report.with_evidence - report.verbatim} 条 claim 的引文在所指块里定位不到"
        )
    if gap > 0:
        return False, f"还有 {gap} 个问题要点没有 claim 覆盖"
    return True, ""


# ---------------------------------------------------------------- 主循环
class AgentRunner:
    """一次 ask 的 agent 循环。构造后可重复调用 ``run``（每个问题独立注册表）。"""

    def __init__(
        self,
        retriever: Retriever,
        client: DeepSeekClient,
        config: AgentConfig | None = None,
        *,
        extra_note: str = "",
        on_step: Callable[[Step], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        cancel_event: Event | None = None,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.config = config or AgentConfig()
        #: 只追加实验约束，不改默认行为（默认空串）
        self.extra_note = extra_note
        #: M6 展示层用的实时步回调（可选）。默认 None 时行为与 M4 完全一致。
        self.on_step = on_step
        #: M6.5 流式合成回调：只收最终答案的增量，不包含研究过程。
        self.on_token = on_token
        #: 客户端断开时由 API 置位；每个研究轮/流式帧之间检查。
        self.cancel_event = cancel_event

    # ------------------------------------------------------------ 单个问题
    def run(self, question: str) -> AgentResult:
        cfg = self.config
        trace = Trace(question=question)
        trace.on_step = self.on_step
        toolbox = ToolBox(
            self.retriever,
            top_k=cfg.top_k,
            include_superseded=cfg.include_superseded,
        )
        verdicts: list[SufficiencyVerdict] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": research_system(self.extra_note)},
            {"role": "user", "content": research_user(question)},
        ]
        force_tool: str | None = toolspec.FIRST_TOOL
        unparsed = 0
        last_missing = ""
        last_tokens = cfg.min_round_tokens
        rounds = 0
        #: "重试一次再降级"的闸门 C 兜底：每种失败形态只重试一次，第二次就降级。
        retried: dict[str, bool] = {}

        while rounds < cfg.max_steps:
            # 客户端已断开：不要再发研究轮（每轮都是钱）。
            if self.cancel_event is not None and self.cancel_event.is_set():
                trace.degrade("cancelled")
                break
            # ---- 预算预判：超了就"少检索一次、走短合成" ----
            # 第一轮永远放行：此时还没花钱，预估值只是"一轮大概要多少"，
            # 拿它去挡第一轮会让预算很小的配置一次都不查（单测覆盖过这个 bug）。
            spent = trace.tokens_used
            if spent > 0 and spent + max(last_tokens, cfg.min_round_tokens) > cfg.max_tokens_budget:
                trace.degrade("budget_exhausted")
                break

            rounds += 1
            choice: Any = toolspec.forced(force_tool) if force_tool else "auto"
            try:
                chat_kwargs: dict[str, Any] = {
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.research_max_tokens,
                    "tools": toolspec.TOOL_SPECS,
                    "tool_choice": choice,
                }
                if self.cancel_event is not None:
                    chat_kwargs["cancel_event"] = self.cancel_event
                result = self.client.chat(messages, **chat_kwargs)
            except LLMError as exc:
                trace.degrade(f"llm_error: {exc}")
                break

            if result.cancelled:
                trace.degrade("cancelled")
                break

            tokens = result.total_tokens
            last_tokens = tokens or cfg.min_round_tokens
            note = (result.text or "").strip()
            note = note if len(note) <= 120 else note[:119] + "…"

            # ---- 分支零：强制调用却没发出 tool_call（闸门 C 的失败形态 1）----
            if not result.tool_calls and force_tool:
                if not retried.get("forced"):
                    retried["forced"] = True
                    trace.add(
                        Step(
                            step=rounds,
                            tool=force_tool,
                            ok=False,
                            tokens=tokens,
                            decision="continue",
                            detail="强制调用没有发出 tool_call，重试一次",
                        )
                    )
                    messages.append(assistant_message(result))
                    messages.append(
                        {
                            "role": "user",
                            "content": f"本轮必须调用 {force_tool} 工具，不要直接回答。",
                        }
                    )
                    continue
                trace.add(
                    Step(
                        step=rounds,
                        tool=force_tool,
                        ok=False,
                        tokens=tokens,
                        decision="continue",
                        detail="重试后仍未发出 tool_call，降级收尾",
                    )
                )
                trace.degrade("first_tool_call_missing")
                break

            # ---- 分支一：模型要调工具 ----
            if result.tool_calls:
                messages.append(assistant_message(result))
                new_total = 0
                arg_errors: list[str] = []
                for call in result.tool_calls:
                    args_raw = call.arguments or ""
                    try:
                        args = call.parsed_arguments()
                        parse_error = ""
                    except ValueError as exc:
                        args = {}
                        parse_error = str(exc)
                        arg_errors.append(call.name)
                    toolbox.current_step = rounds
                    tool_result = toolbox.run(call.name, args, args_raw=args_raw)
                    text = tool_result.text if tool_result.ok else (
                        f"{tool_result.text}\n（本次调用没有拿到结果，请换一种检索方式）"
                    )
                    messages.append(tool_message(call.id, text))
                    new_total += tool_result.n_new
                    step = Step(
                        step=rounds,
                        tool=call.name,
                        args=args,
                        args_raw=args_raw,
                        ok=tool_result.ok and not parse_error,
                        n_hits=tool_result.n_new,
                        hit_ids=list(tool_result.new_ids),
                        sources=list(tool_result.sources),
                        ms=tool_result.ms,
                        tokens=tokens if len(result.tool_calls) == 1 else None,
                        decision="continue",
                        detail="；".join(
                            part
                            for part in (
                                parse_error,
                                tool_result.detail,
                                note and f"模型说：{note}",
                                n_hits_note(tool_result),
                            )
                            if part
                        ),
                    )
                    trace.add(step)
                if tokens and len(result.tool_calls) > 1:
                    # 并行调用时 tokens 是整轮的成本，记在**这一轮的第一步**上，
                    # 其余步留空，避免 trace 的合计把同一轮算 N 次。
                    round_steps = [s for s in trace.steps if s.step == rounds]
                    for step in round_steps:
                        step.tokens = None
                    if round_steps:
                        round_steps[0].tokens = tokens
                    trace.tokens_used += tokens
                # ---- 闸门 C 的失败形态 2：arguments 不是合法 JSON ----
                if arg_errors:
                    if not retried.get("args"):
                        retried["args"] = True
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "上一轮的工具 arguments 不是合法 JSON（工具没能执行）。"
                                    "请重新调用同一个工具，arguments 必须是一个 JSON 对象，"
                                    "字符串用双引号。"
                                ),
                            }
                        )
                        force_tool = arg_errors[0]
                        continue
                    trace.degrade("arguments_invalid")
                    break
                force_tool = None
                continue

            # ---- 分支二：模型给判定 ----
            verdict = parse_verdict(result.text)
            if verdict is None:
                unparsed += 1
                trace.add(
                    Step(
                        step=rounds,
                        tool="sufficiency",
                        args={},
                        args_raw=(result.text or "")[:500],
                        ok=False,
                        tokens=tokens,
                        decision="continue",
                        detail=f"第 {unparsed} 次给不出契约 JSON；已强制再检索一次",
                    )
                )
                if unparsed >= cfg.max_unparsed_rounds:
                    trace.degrade("sufficiency_parse_failed")
                    break
                # 解析失败按"继续"处理：强制它至少再查一次（任务书 d 项）
                force_tool = toolspec.FIRST_TOOL
                messages.append(assistant_message(result))
                messages.append(
                    {
                        "role": "user",
                        "content": sufficiency_nudge(
                            "输出不是约定的 JSON（必须是含 answerable 字段的一行 JSON）"
                        ),
                    }
                )
                continue

            verdicts.append(verdict)
            last_missing = verdict.missing
            if not verdict.answerable:
                trace.add(
                    Step(
                        step=rounds,
                        tool="sufficiency",
                        args=verdict.to_dict(),
                        args_raw=verdict.raw[:500],
                        ok=True,
                        tokens=tokens,
                        decision="continue",
                        detail="判定为不足：" + (verdict.missing or "（未写明缺什么）"),
                    )
                )
                messages.append(assistant_message(result))
                hint = (
                    f"上一轮判定 insufficient，缺：{verdict.missing}。"
                    + (f"下一句可以查：{verdict.next_query}" if verdict.next_query else "")
                )
                messages.append({"role": "user", "content": sufficiency_nudge(hint)})
                # 下一轮若没有明确 next_query 就强制检索一次，避免空转
                force_tool = None if verdict.next_query else toolspec.FIRST_TOOL
                continue

            # 判定足够 → 过 claim 短路这道闸
            claims = claims_from_verdict(verdict, toolbox.registry)
            report = verify_claims(
                claims,
                toolbox.registry_texts(),
                toolbox.registry,
                raw_text="",
            )
            gap = coverage_gap(question, claims)
            ok, reason = _claims_ok(claims, report, gap)
            trace.add(
                Step(
                    step=rounds,
                    tool="claims",
                    args=verdict.to_dict(),
                    args_raw=verdict.raw[:500],
                    ok=ok,
                    tokens=tokens,
                    decision="finalize" if ok else "continue",
                    detail=(
                        f"claim {len(claims)} 条，逐字 {report.verbatim}/{report.with_evidence}，"
                        f"要点缺口 {gap}"
                        + ("" if ok else f"；不短路：{reason}")
                    ),
                )
            )
            if ok and not verdict.missing:
                verified = [
                    {
                        "statement": claim.statement,
                        "handles": list(claim.numbers),
                        "evidence": claim.evidence,
                    }
                    for claim in claims
                ]
                return self._finalize(
                    question, toolbox, trace, verdicts, last_missing, True, "", verified
                )
            messages.append(assistant_message(result))
            messages.append(
                {
                    "role": "user",
                    "content": sufficiency_nudge(
                        reason or (f"missing 非空：{verdict.missing}" if verdict.missing else "")
                    ),
                }
            )
            force_tool = None

        # ---- 收尾（正常走完 / 步数用尽 / 预算用尽 / 依赖失败）----
        return self._finalize(question, toolbox, trace, verdicts, last_missing, False, "")

    # ------------------------------------------------------------ 兜底合成
    def _finalize(
        self,
        question: str,
        toolbox: ToolBox,
        trace: Trace,
        verdicts: list[SufficiencyVerdict],
        last_missing: str,
        short_circuit: bool,
        note: str,
        verified: list[dict] | None = None,
    ) -> AgentResult:
        cfg = self.config
        hits = toolbox.hits_in_registry_order()
        evidence = toolbox.evidence_map()
        #: ---- 改动 1：合成只喂"过门"的块 ----
        #: 过门 = (a) 短路 claim 实际引用过的句柄块 ∪ (b) 每个工具步的前 N 块
        #: （N = ``synthesis_per_step``，等价于 RAGFlow 的 keepTopN 豁免头部——
        #: 它保证"过滤只会缩短尾部，不会把结果集清空"，见 decision_filter.go:299-333）。
        #: 没有短路的题没有 (a)，只剩 (b)+……不够时退回全给，避免把兜底合成饿死。
        selected = self._select_synthesis(toolbox, trace, verified)
        usable, dropped = truncate_hits(
            selected, self.retriever.docs_by_id, cfg.evidence_max_chars
        )
        trace.n_evidence = len(hits)
        #: 合成是一步"额外"的动作，步号接着最后一次模型轮次往下编——
        #: 用 ``len(steps)+1`` 会在并行工具调用时跳号（一轮多步），读轨迹的人会以为丢了步。
        step_no = (trace.steps[-1].step if trace.steps else 0) + 1

        # 一份证据都没有：确定性拒答，不调用模型（M0 起的不变量）
        if not usable:
            trace.add(
                Step(
                    step=step_no,
                    tool="finalize",
                    ok=True,
                    decision="refuse",
                    detail="零证据，未调用模型",
                )
            )
            trace.outcome = "refuse"
            trace.refusal_reason = "no_hit"
            answer = Answer(
                question=question,
                answer=f"{REFUSAL_PREFIX}：知识库中没有检索到与该问题相关的资料。",
                refused=True,
                context_chunk_ids=[],
                warnings=["零证据，未调用模型"] + ([note] if note else []),
                refusal_reason="no_hit",
            )
            return AgentResult(
                answer=answer,
                trace=trace,
                hits=hits,
                # 零证据：既没有合成上下文，也没有发布清单
                synthesis=[],
                published=[],
                evidence=evidence,
                verdicts=verdicts,
                last_missing=last_missing,
                short_circuit=short_circuit,
            )

        messages = finalize_messages(
            question,
            usable,
            self.retriever.docs_by_id,
            self.retriever.versions,
            include_paths=cfg.include_paths,
            variant=cfg.prompt_variant,
            verified_claims=verified,
            verified_only=cfg.verified_only,
        )
        if self.cancel_event is not None and self.cancel_event.is_set():
            chat, error = None, "已取消"
        else:
            chat, error = self._safe_chat(
                messages,
                max_tokens=cfg.finalize_max_tokens,
                trace=trace,
                step_no=step_no,
                on_token=self.on_token,
                cancel_event=self.cancel_event,
            )
        spent = chat.total_tokens if chat is not None else 0
        if chat is not None and chat.cancelled:
            trace.degrade("cancelled")
            chat, error = None, "已取消"
        retried = False
        # 流式已经吐过 token，再重试会让前端看到重复半截；流式路径不做"太短重试"。
        if chat is not None and cfg.min_answer_chars > 0 and not cfg.stream:
            body = (chat.text or "").strip()
            if len(body) < cfg.min_answer_chars and not is_refusal(body):
                retried = True
                messages.append(assistant_message(chat))
                messages.append(
                    {"role": "user", "content": finalize_retry_note(cfg.min_answer_chars, body)}
                )
                chat2, error2 = self._safe_chat(
                    messages,
                    max_tokens=cfg.finalize_max_tokens,
                    trace=trace,
                    step_no=step_no,
                )
                if chat2 is not None:
                    spent += chat2.total_tokens or 0
                    chat = chat2
                else:
                    error = error2 or error

        if chat is None:
            # 依赖失败 → 降级交付（不编内容），退出码仍为 0
            text = degraded_answer(question, usable, error or "模型不可用")
            answer = Answer(
                question=question,
                answer=text,
                refused=True,
                context_chunk_ids=[hit.chunk.chunk_id for hit in usable],
                warnings=[f"依赖失败，已降级：{error}"],
                refusal_reason="degraded_dependency",
            )
            trace.add(
                Step(
                    step=step_no,
                    tool="finalize",
                    ok=False,
                    ms=None,
                    tokens=None,
                    decision="refuse",
                    detail=f"依赖失败，降级答案：{error}",
                )
            )
            trace.outcome = "degraded"
            trace.refusal_reason = "degraded_dependency"
            return AgentResult(
                answer=answer,
                trace=trace,
                hits=hits,
                # 降级答案不做引用发布：它是"原文片段 + 拒答前缀"，没有句柄
                synthesis=usable,
                published=[],
                evidence=evidence,
                verdicts=verdicts,
                last_missing=last_missing,
                short_circuit=short_circuit,
            )

        # ---- 句柄清洗 → 引用回填 → 按首引序发布（与 M3 同一套函数 + 改动 2）----
        allow_legacy = _HANDLE_ID.search(chat.text) is None
        raw_handles = extract_handles(chat.text, include_legacy=allow_legacy)
        raw_invalid = sorted(
            {item.number for item in raw_handles if item.index(len(usable)) is None}
        )
        cleaned, removed = sanitize_answer(chat.text, len(usable), allow_legacy=allow_legacy)
        #: ---- 改动 2：发布清单按"被引"生成 ----
        #: 合成上下文是 0..k-1 的紧凑编号（模型只看得到这一层），但发布给前端/评测的
        #: 引用清单要按**首引序**重编号，且只含被引块——抄 ``citations.go:207-245``。
        #: 重编号之后句柄仍按"新号 = 发布清单下标"走位，所以下游的
        #: ``_build_citations`` / 句级解析 / 评测口径都不用改。
        published_text = publish_by_citation(cleaned, len(usable))
        cleaned = published_text.text
        published_hits = (
            [usable[index] for index in published_text.order]
            if published_text.order
            else []
        )
        if published_text.dropped:
            removed = sorted(set(removed) | set(published_text.dropped))
        citations, _ = _build_citations(
            extract_handles(cleaned, include_legacy=allow_legacy),
            published_hits,
            self.retriever.docs_by_id,
        )
        refused = is_refusal(cleaned)

        warnings: list[str] = []
        if raw_invalid:
            warnings.append(
                f"引用了不存在的资料编号 {raw_invalid}（本次上下文只有 {len(usable)} 条），已从正文删除"
            )
        if not refused and not citations:
            warnings.append("回答未标注任何引用")
        if dropped:
            warnings.append(f"证据超过 {cfg.evidence_max_chars} 字符，整块丢弃尾部 {dropped} 条")
        if retried:
            warnings.append("回答过短，已重试一次")

        # ---- claim 侧打点：注册表按**发布清单**重建（新号 = 发布清单下标）----
        registry = EvidenceRegistry.from_hits(published_hits)
        claims: list = []
        claim_report = None
        claim_fallback = False
        if cfg.prompt_variant == "claim" and not refused:
            claims = parse_claims(cleaned, registry)
            if not claims:
                claim_fallback = True
                warnings.append(
                    "claim 解析失败（没有一行符合 {陈述, 证据, 句柄} 格式），已按普通回答交付"
                )
            else:
                claim_report = verify_claims(
                    claims,
                    {hit.chunk.chunk_id: hit.context_chunk.text for hit in published_hits},
                    registry,
                    raw_text=cleaned,
                )

        answer = Answer(
            question=question,
            answer=cleaned,
            refused=refused,
            citations=citations,
            cited_markers=[c.marker for c in citations],
            invalid_markers=raw_invalid,
            sanitized_markers=removed,
            context_chunk_ids=[hit.chunk.chunk_id for hit in published_hits],
            published_ids=[hit.chunk.chunk_id for hit in published_hits],
            warnings=warnings,
            claims=claims,
            claim_report=claim_report,
            claim_fallback=claim_fallback,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            latency_ms=chat.latency_ms,
            model=chat.model,
            # 归因：有证据却拒答 = 判定不充分（依赖失败在上面单独分支里已归因）
            refusal_reason="judged_insufficient" if refused else None,
        )
        trace.add(
            Step(
                step=step_no,
                tool="finalize",
                ok=True,
                tokens=spent,
                ms=chat.latency_ms,
                decision="refuse" if refused else "finalize",
                detail=(
                    f"合成上下文 {len(usable)}/{len(hits)} 块（截断丢 {dropped}）"
                    f"；发布 {len(published_hits)} 块"
                    + ("；拒答" if refused else f"；引用 {len(citations)} 条")
                    + ("；重试一次" if retried else "")
                ),
            )
        )
        trace.outcome = "refuse" if refused else "answer"
        trace.refusal_reason = answer.refusal_reason
        return AgentResult(
            answer=answer,
            trace=trace,
            hits=hits,
            synthesis=usable,
            published=published_hits,
            evidence=evidence,
            verdicts=verdicts,
            last_missing=last_missing,
            short_circuit=short_circuit,
        )

    # ------------------------------------------------------------ 合成取证
    def _select_synthesis(
        self,
        toolbox: ToolBox,
        trace: Trace,
        verified: list[dict] | None,
    ) -> list[Retrieved]:
        """挑出**可以进合成上下文**的块（改动 1）。

        规则两条，都是"过门"的机械定义：

        1. 短路 claim 里出现过的句柄块——它们是已经逐字校验过的结论所依赖的原文；
        2. 每个工具步的前 ``synthesis_per_step`` 块——保证"即使这一步的块没被 claim 用到，
           它的头部仍然可见"，等价于 RAGFlow 的 keepTopN 豁免头部
           （``decision_filter.go:326-333``：过滤只能剪尾，不能清空结果集）。

        两条都为空时**退回全给**：这种情形只会在"没短路且没有任何工具步"时出现
        （例如全靠一次强制检索就走到 max_steps），那时收紧没有依据，收紧反而会饿死合成。
        """
        hits = toolbox.hits_in_registry_order()
        if not hits:
            return []
        if self.config.synthesis_per_step <= 0:
            #: 开关关闭：全部发号块进合成（M4 第一版行为，也是当前默认）
            return hits
        chosen: set[int] = set()
        for claim in verified or []:
            for number in claim.get("handles") or []:
                if isinstance(number, int) and 0 <= number < len(hits):
                    chosen.add(number)
        for step in trace.tool_steps:
            for number in (step.hit_ids or [])[: self.config.synthesis_per_step]:
                if 0 <= number < len(hits):
                    chosen.add(number)
        if not chosen:
            return hits
        return [hits[number] for number in sorted(chosen)]

    # ------------------------------------------------------------ 一次安全调用
    def _safe_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        trace: Trace,
        step_no: int,
        on_token: Callable[[str], None] | None = None,
        cancel_event: Event | None = None,
    ) -> tuple[ChatResult | None, str]:
        """合成阶段的模型调用：失败只记账，不抛。

        token 由调用方累加进 ``Step.tokens``（``Trace.add`` 负责结算进
        ``tokens_used``），这里**不**重复写 ``trace.tokens_used``。

        流式只在「配置打开 + 客户端真的支持 chat_stream」时启用；假客户端
        （单测）没有该方法，自动退回一次性 ``chat``，测试契约不变。
        """
        try:
            if self.config.stream and hasattr(self.client, "chat_stream"):
                chat = self.client.chat_stream(
                    messages,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                    on_token=on_token,
                    cancel_event=cancel_event,
                )
            else:
                chat = self.client.chat(
                    messages,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                )
        except LLMError as exc:
            trace.degrade(f"finalize_failed: {exc}")
            return None, str(exc)
        return chat, ""


def n_hits_note(result: ToolResult) -> str:
    """工具结果里"新发了几块"的一行说明（trace detail 用）。"""
    if result.n_new:
        return f"新增证据 {result.n_new} 块"
    return "没有新增证据"
