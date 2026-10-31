"""文档内容指纹：判断"两份文件是不是同一份内容"。

为什么需要它（`storage/runs/adversarial_gold40.md` 的 B-1）：

RAGFlow 把"是不是同一份"的身份放在 **`document.content_hash`**（文件字节的 xxh128，
`document_upload.go:341-342`、`entity/document.go:42`），证据渲染层再按
`sha256(text)` 去重（`ServeText`，`docs/逆向/notes/B-retrieval.md:459`）。

我们的语料里重复不是"同一个文件"，而是**同一份材料的不同格式/不同版本**
（docx / pdf / 无信息副本 / 规整版）。字节哈希区分不了它们，所以把身份定在
**抽取后的全文**上：按 chunk 的字符区间拼回全文（去重叠、去父子重复）、去空白，
取 blake2b(16 hex)。这样同一份文档在 fixed / token / section / parent_child
四种分块下指纹一致，严格口径的计数才可比。

它只是派生信息，**不入库**，也不改 `chunk_id` / `chunk.content_hash` / 向量索引：

- 评测的严格口径用它当计数单位："这个证据串出现在几份**内容不同**的文档里"，
  而不是"出现在几个 chunk 行里"。同一段话被 chunk overlap 切成 3 个窗口、
  被 3 个副本各存一份时，chunk 计数会算 9 次，内容指纹只算 1~3 次。
- 报告里可以指出来"哪些文档其实是同一份内容"，便于人工判断版本冲突题是否成立。

**它不做的事**：不合并文件、不改变检索结果、不把同内容文档标成 `duplicate`。
版本冲突题恰恰需要多份"内容相近但不相同"的版本同时存在，合并会毁掉这类题。
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

_WS = re.compile(r"\s+")

#: 指纹长度（hex 字符数）。8 字节足够区分几千份文档，短到可以贴进报告。
_FP_SIZE = 8


def _flat(text: str) -> str:
    return _WS.sub("", text or "")


def _document_text(chunks: list) -> str:
    """把同一文档的 chunk 拼回全文，**按字符区间去重**。

    不能直接拼 ``c.text``：固定窗口有 150 字重叠，父子块策略里父块又完整包含子块，
    直接拼会把同一段文字算两次甚至三次。更严重的是，这样得到的"文档指纹"会随
    **分块参数**变化——同一份文档在 fixed 与 token 下会被认成两份不同内容，
    严格口径的 span 频次也跟着变，M1/M2 的分块对照就失去了同一把尺子。

    按 ``char_start/char_end`` 取并集：每段只保留游标之后的字符。实测同一份文档在
    fixed / token / section / parent_child 四种切法下，去空白后的指纹完全一致。
    """
    ordered = sorted(chunks, key=lambda c: (c.char_start, c.ordinal))
    parts: list[str] = []
    cursor = 0
    for chunk in ordered:
        start = max(chunk.char_start, cursor)
        if chunk.char_end <= start:
            continue
        skip = max(0, start - chunk.char_start)
        if skip >= len(chunk.text):
            continue
        piece = chunk.text[skip:]
        if piece:
            parts.append(piece)
            cursor = max(cursor, chunk.char_end)
    return "".join(parts)


def document_fingerprints(chunks: Iterable) -> dict[str, str]:
    """``doc_id -> 内容指纹``。

    指纹 = 文档全文（去空白）的 blake2b。它**与分块策略无关**：
    先按 ``char_start/char_end`` 把 chunk 拼回全文（去掉重叠与父子重复），再哈希。
    同一份材料的 docx / pdf / 无信息副本可能抽出相同的文字，于是得到同一指纹，
    这是有意的——严格口径的计数单位是"内容不同的文档"，不是文件名。
    """
    by_doc: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    fingerprints: dict[str, str] = {}
    for doc_id, doc_chunks in by_doc.items():
        if not any(c.char_end for c in doc_chunks):
            # 老数据没有字符区间（全部为 0）：退回按 ordinal 拼接。
            # 不做这个兜底，所有文档都会拼成空串、指纹全相同，重复计数会静默归零。
            text = "".join(
                _flat(c.text) for c in sorted(doc_chunks, key=lambda c: c.ordinal)
            )
        else:
            text = _flat(_document_text(doc_chunks))
        fingerprints[doc_id] = hashlib.blake2b(
            text.encode("utf-8"), digest_size=_FP_SIZE
        ).hexdigest()
    return fingerprints
