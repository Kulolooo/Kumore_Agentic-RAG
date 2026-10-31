"""M4 的三段提示词：研究轮 / 充分性判定 / 兜底合成。

**全部只用通用特征**：不提任何语料名、题号、专有名词。换一套语料后，这几段话一字不用改。

与 RAGFlow 的对应关系（抄机制不抄规模）：

====================  ======================================  ==========================
我们                  对方                                    出处
====================  ======================================  ==========================
首轮强制 search_kb     ``tool_choice`` 强制检索                ``retrieval_model.go:93-99``
充分性 JSON            ``{"answerable": ..., "missing": ...}``  ``08`` §M2③
claim 短路             检索召回"编译好的 claim 行"              ``tree/claims.go``
证据块 ``[ID:n]``      ``EvidenceRegistry.ServeText``           ``evidence_registry.go:68-86``
====================  ======================================  ==========================

最大的差异在 claim 短路：对方是**查离线编译好的 claim 行**（"陈述+逐字引文+ref"存表），
我们是**让模型在运行时把已召回片段压成 claim，再用 M3 的逐字校验判真假**。
抄的是"claim 够用就短路"这个机制，不是"离线编译 claim 库"那套规模——
130 份文档不值得建编译管线（``08`` 的原话是这份规模不抄）。
"""

from __future__ import annotations

from app.generate.prompt import (
    _CLAIM_FORMAT,
    REFUSAL_PREFIX,
    _CITATION_RULES,
    _build_system_prompt,
    _neutralize_handles,
)
from app.retrieve.versions import VersionInfo
from app.schemas import Doc, Retrieved

#: 充分性判定的 JSON 契约。字段名与路线图 §6 的 ``sufficiency_gate`` 一致。
VERDICT_FIELDS = ("answerable", "missing")

_RESEARCH_RULES = """你是检索助手。你**不能**凭记忆或常识回答问题，只能依据工具返回的原文。

【可用工具】
- search_kb(query, top_k, scope)：按关键词检索，返回带编号的原文片段。scope 填文档代号可只查那一份。
- grep_kb(pattern, max_hits)：全文按字面查找一段字符串（数字、型号、专有名词最有效）。
- read_section(doc, near, limit)：读某一份文档的更多片段。
- list_docs(keyword, limit)：列出知识库里的文档代号和开头一小段正文。

【工作方式】
1. 第一轮必须调用 search_kb。
2. 问题里有几个要点，就分几次检索；不要把所有词塞进一个长查询。
3. 每次拿到结果后判断：问题里的**每一个**要点，都能被已读到的原文直接回答吗？
4. 如果不够，换一个**具体**的检索词再来一次。优先用刚读到的原文里出现过的实体
   （人名、型号、数字、表格名、文件名里的词），**不要**把原问题再问一遍。
   如果某个要点确定在某一类文档里、但检索总被别的文档挤掉，用 scope 或 read_section 深入读。
5. 如果够了，**不要调用工具**，只输出一行 JSON（不要散文、不要代码块围栏）：
   {"answerable": true, "missing": "", "claims": [{"statement": "一句话陈述", "evidence": "从原文逐字复制的连续片段", "handles": ["[ID:0]"]}]}
6. 如果还不够、并且你判断已经查不下去了，输出：
   {"answerable": false, "missing": "还缺哪一项信息", "next_query": "下一句打算查什么"}
7. {"answerable": true} 的前提是：每条 claim 的 evidence 都能在你看到的原文里**逐字**找到，
   且问题问到的每一项都有 claim 覆盖。写不出逐字证据的要点，就放进 missing 里说清楚。
8. 知识库里确实没有相关内容时，也要输出 answerable=false，并在 missing 里写明"知识库中未找到"。
"""


def research_system(extra_note: str = "") -> str:
    """研究轮的 system 提示词。``extra_note`` 留给调用方追加实验性约束。"""
    body = _RESEARCH_RULES.strip()
    if extra_note.strip():
        body += "\n\n【补充约束】\n" + extra_note.strip()
    return body


def research_user(question: str, *, step: int = 1, hint: str = "") -> str:
    """研究轮的首条 user 消息。

    ``hint`` 给"上一轮没按契约输出"这类纠偏用；正常路径下为空。
    """
    lines = [f"【问题】\n{question.strip()}"]
    if step > 1:
        lines.append(f"（第 {step} 轮）")
    if hint.strip():
        lines.append(f"【上一轮的偏差】\n{hint.strip()}")
    return "\n\n".join(lines)


def sufficiency_nudge(reason: str) -> str:
    """判定"不够/不合格"之后，回给模型的纠偏话术。

    ``reason`` 必须是机制性描述（缺几个要点、哪条引文定位不到），不能写"再想想"。
    """
    return (
        "上一轮的判定没有被接受：" + reason + "\n"
        "请继续检索（调用工具），或按契约重新输出 JSON；"
        "claims 的 evidence 必须是原文里逐字连续的一段。"
    )


def finalize_messages(
    question: str,
    hits: list[Retrieved],
    docs_by_id: dict[str, Doc],
    versions: dict[str, VersionInfo] | None = None,
    *,
    include_paths: bool = False,
    variant: str = "claim",
    verified_claims: list[dict] | None = None,
    verified_only: bool = False,
) -> list[dict]:
    """兜底合成用的 messages。

    复用 M3 的 ``build_messages``：资料块格式、``[ID:n]`` 渲染、正文里的句柄中和
    （``_neutralize_handles``）都不另写一份——两处格式一旦不一致，模型抄回去的编号
    就会指错块。

    与 M3 的区别有两处：

    1. ``hits`` 是**按证据注册表顺序**排好的（不是一次检索的 top-k），
       所以 ``[ID:n]`` 的 n 同时是"注册表下标"和"提示词里的块序"。
    2. ``verified_claims`` 是研究阶段**已逐字校验通过**的中间结论。实测（t01）：
       研究阶段明明拿到了 3 条逐字 claim 并短路成功，合成阶段却因为"问题里那句
       '先做过 CIFAR-10 的那门课'指哪门课无法确定"而拒答——因为它从零重推了一遍。
       把已核实的中间结论递过去，短路才有意义；否则 claim 校验只是省了一次检索。
    """
    from app.generate.prompt import build_messages

    messages = build_messages(
        question,
        hits,
        docs_by_id,
        versions,
        include_paths=include_paths,
        variant=variant,
    )
    prefix: list[str] = [
        "【编号约定】下面每段资料开头的 [ID:n] 是**本次回答自己的编号**（从 0 开始连续）。"
        "只能引用这里出现过的编号；检索阶段见过的编号在这次回答里不适用。"
    ]
    if verified_claims:
        prefix.append("【已核实的中间结论】")
        for claim in verified_claims:
            handles = "".join(f"[ID:{number}]" for number in claim.get("handles", []))
            prefix.append(f"- {claim.get('statement', '').strip()}（句柄 {handles or '无'}）")
        prefix.append(
            "以上每条结论的引文都已在原文里逐字核对通过，可以直接作为回答的中间步骤；"
            "使用时仍要标注它对应的句柄。不要因为它们没有逐字复述问题而判为不足。"
        )
        if verified_only:
            #: P69 修复实验：实测多步答案比 M3 多出约 27% 的引用，其中一部分是
            #: "有据但不是金标那条"的额外子事实。收窄上下文（改动 1）伤过答案，
            #: 这里换一个更窄的动作：合成只复述已核实结论，不再扩展新 claim。
            prefix.append(
                "【交付约束】本次合成只能使用上面【已核实的中间结论】：不要新增 claim，"
                "不要引入新的句柄，也不要把【资料】里其他内容写成新事实。"
                "每条结论只保留它自己对应的句柄；无法确认的内容不要写。"
            )
    messages[1]["content"] = "\n".join(prefix) + "\n\n" + messages[1]["content"]
    return messages


def evidence_block_chars(hits: list[Retrieved], docs_by_id: dict[str, Doc]) -> list[int]:
    """每块资料在提示词里的字符数（用于 40k 截断时决定保留前几块）。"""
    sizes: list[int] = []
    for hit in hits:
        chunk = hit.context_chunk
        doc = docs_by_id.get(chunk.doc_id)
        head = len(str(chunk.page or "")) + len(chunk.section or "") + len(doc.title if doc else "")
        sizes.append(head + len(chunk.text or "") + 32)
    return sizes


def truncate_hits(
    hits: list[Retrieved], docs_by_id: dict[str, Doc], max_chars: int
) -> tuple[list[Retrieved], int]:
    """把证据截到 ``max_chars`` 以内，**从尾部整块丢弃**（不切半块）。

    为什么整块丢：切半块会让逐字校验在"被切断的引文"上失败，凭空制造逐字率噪声。
    丢弃的块仍留在注册表里（编号不回收），但不在上下文里——所以最终合成的
    ``n_context`` 要用截断后的长度，越界判断才正确。
    """
    sizes = evidence_block_chars(hits, docs_by_id)
    kept = 0
    total = 0
    for size in sizes:
        if total + size > max_chars and kept > 0:
            break
        total += size
        kept += 1
    return hits[:kept], len(hits) - kept


def finalize_retry_note(min_chars: int, got: str) -> str:
    """回答过短时的重试话术（``min_answer_chars`` 重试）。"""
    return (
        f"上一轮回答只有 {len(got)} 个字符，不足 {min_chars} 个字符，判定为无效输出。"
        "请按既定格式重新作答：能确认的写出来并标注编号；确实不足就只输出一行"
        f"「{REFUSAL_PREFIX}：<缺少什么>」。"
    )


def degraded_answer(question: str, hits: list[Retrieved], reason: str, *, max_chars: int = 600) -> str:
    """依赖失败时的降级答案。

    **不编内容**：只把已检索到的原文片段原样附上，并说明这次没有完成归纳。
    这与"宁可拒答也不编造"的产品不变量一致——降级交付的是"可核对的原料"，
    不是"看起来像答案的东西"。
    """
    lines = [
        f"{REFUSAL_PREFIX}：本次问答过程中出现依赖故障（{reason}），未能完成归纳。",
        "以下是已经检索到的相关原文片段，未经模型整理，供人工核对：",
    ]
    total = 0
    for hit in hits:
        chunk = hit.context_chunk
        text = _neutralize_handles(_clip(chunk.text.strip(), 200))
        lines.append(f"- {text}")
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = [
    "VERDICT_FIELDS",
    "research_system",
    "research_user",
    "sufficiency_nudge",
    "finalize_messages",
    "evidence_block_chars",
    "truncate_hits",
    "finalize_retry_note",
    "degraded_answer",
    "REFUSAL_PREFIX",
    "_build_system_prompt",
    "_CLAIM_FORMAT",
    "_CITATION_RULES",
]
