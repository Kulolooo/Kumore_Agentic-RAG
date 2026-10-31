"""数据结构。

这些对象会序列化进 JSONL / SQLite，也会跨模块传递，所以字段一旦定下就尽量只增不改。
所有 ``to_dict`` 都保证是纯 JSON 可序列化类型（Path 转 str、frozenset 不出现）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_doc_id(rel_path: str) -> str:
    """文档 id 只由相对路径决定。

    这样同一份文件重新扫描得到同一个 id，增量更新时可以直接比对；
    不用内容 hash 是因为改了内容后 id 变了会让历史引用失效。
    """
    normalized = rel_path.replace("\\", "/").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def make_chunk_id(doc_id: str, ordinal: int) -> str:
    return f"{doc_id}:{ordinal:04d}"


_WS = re.compile(r"\s+")


def make_content_hash(text: str) -> str:
    """片段的内容指纹：只看文字，不看它在哪份文件里。

    为什么不用它当主键：RAGFlow 的 chunk id 有意带 `doc_id` 作用域，
    目的是**避免跨文档同文本互相覆盖**（`docs/逆向/notes/A-ingestion.md:379`）。
    我们保留 `chunk_id = doc_id:ordinal` 的写隔离语义，另存一列内容指纹，
    这样"哪几段其实是同一段话"才查得出来（`docs/M0-2-逆向自查.md` §2）。

    归一化只做"去掉所有空白"：PDF/PPTX 抽出来的换行会把同一句话切成两段，
    不去空白就认不出重复；但保留大小写与标点，避免把不同内容误合并。
    """
    normalized = _WS.sub("", text or "")
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()


@dataclass
class Doc:
    """一份被扫描到的文档。"""

    doc_id: str
    rel_path: str
    title: str
    ext: str
    sha256: str
    size: int
    mtime: float
    #: active = 当前版本；superseded = 被同目录下同名的更新版本取代
    status: str = "active"
    #: 解析是否成功。失败也保留记录，便于统计"解析通过率"
    parse_ok: bool = True
    parse_error: str | None = None
    n_chunks: int = 0
    #: 生成这份文档 chunk 集时用的配置指纹（分块参数 + 解析后端）。
    #: 增量重建靠它判断"这份文档的 chunk 是不是拿旧参数切的"，比存一个状态枚举管用。
    chunk_config: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """检索的最小单位。"""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    page: int | None = None
    #: 片段跨越页边界时，记录终点页；用于把引用写成「第 2–3 页」而不是只写起点页
    page_end: int | None = None
    section: str | None = None
    char_start: int = 0
    char_end: int = 0
    #: 内容指纹（见 ``make_content_hash``）。用于跨文档识别"同一段话"与增量比对。
    content_hash: str = ""
    #: 父子块策略（small-to-big）：父块本身不参与检索，只用来回填上下文。
    #: ``is_parent=True`` 的块不建 BM25、不编码向量；子块的 ``mom_id`` 指向父块。
    is_parent: bool = False
    mom_id: str | None = None
    #: 关联的页面图/图片 id；文本 chunk 也可指向它，用于“文字找图”和懒编码。
    image_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def preview(self) -> str:
        """单行摘要，用于日志和评测表。"""
        flat = " ".join(self.text.split())
        return flat[:80] + ("…" if len(flat) > 80 else "")


@dataclass
class ImageRecord:
    """图片/页面图单元；视觉索引只认 image_id，不认 chunk_id。"""

    image_id: str
    doc_id: str
    rel_path: str
    page: int
    image_file: str
    sha256: str
    width: int
    height: int
    text_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlsoIn:
    """同一段话的另一个来源。

    折叠重复片段时，主结果只留排名最高的那条；其余来源记在这里，
    这样引用仍然能回填到"这句话也出现在哪份文件的哪一页"。
    """

    chunk_id: str
    doc_id: str
    rel_path: str = ""
    page: int | None = None


@dataclass
class Retrieved:
    """一条检索结果。"""

    chunk: Chunk
    score: float
    rank: int = 0
    #: 召回来源：bm25 / vector / bm25+vector。回答"这条到底是谁召回的"，
    #: 是 RAGFlow 逆向报告补的第一条验收标准。
    source: str = ""
    #: 与这条内容完全相同的其他片段（折叠掉了，只保留来源信息）
    also_in: list[AlsoIn] = field(default_factory=list)
    #: 父子块策略下，命中子块时回填的父块。父块正文更大，回答与引用展示都用它；
    #: 检索指标仍然按子块算（否则 Hit@5 会被父块的粗粒度虚高）。
    parent: Chunk | None = None

    @property
    def context_chunk(self) -> Chunk:
        """真正送进模型的块：有父块用父块，否则用命中的块。"""
        return self.parent or self.chunk


@dataclass
class Citation:
    """回答中出现的引用编号，以及它实际指向的片段。"""

    #: 字面写出来的数字。``[ID:n]`` 是 0 基，过渡期的 ``[n]`` 是 1 基；
    #: 要判断是哪种、要怎么显示，看 ``handle``。
    marker: int
    chunk_id: str
    doc_id: str
    rel_path: str
    page: int | None
    section: str | None
    page_end: int | None = None
    text: str = ""
    #: 模型写出来的那个字面量（``[ID:0]`` / ``[3]``）。展示层以它为准，
    #: 因为 M3 迁移期两种写法可能同时存在，光看数字分不出来。
    handle: str = ""


@dataclass
class Answer:
    """生成结果。"""

    question: str
    answer: str
    #: True 表示模型明确表示资料不足，没有编造
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    cited_markers: list[int] = field(default_factory=list)
    #: 引用了不存在的编号，或格式串号，属于生成质量问题
    invalid_markers: list[int] = field(default_factory=list)
    #: 生成时出现过、但已被从回答正文里删掉的越界编号（保留记录以便统计）
    sanitized_markers: list[int] = field(default_factory=list)
    context_chunk_ids: list[str] = field(default_factory=list)
    #: 发布清单：答案里 ``[ID:n]`` 的 n 就是它的下标，**只含被引用的块**（首引序）。
    #: M4 起 agent 路会做"发布清单按被引生成"（抄 RAGFlow 的 ResolveCitations）；
    #: 单步路仍按"展示序"发号，这两个字段此时等于 context_chunk_ids。
    published_ids: list[str] = field(default_factory=list)
    #: 生成质量问题：比如没有拒答却一个引用都没标、引用了编号范围外的资料
    warnings: list[str] = field(default_factory=list)
    #: 拒答归因。M3 的评测在 ``EvalRow`` 里按"上下文是否为空"推导；M4 的 agent 可能
    #: 在**有证据**的情况下因为依赖故障降级，光看上下文分不出来，所以由生成侧直接给。
    #: 取值：``no_hit``（没找到）/ ``judged_insufficient``（找到了但判定不充分）/
    #: ``degraded_dependency``（依赖失败降级，M4 新增）。非拒答时为 None。
    refusal_reason: str | None = None
    #: claim 结构的解析结果（M3）。类型是 ``app.generate.claim.Claim``，
    #: 这里用 Any 是为了让 schemas 保持"叶子模块"——它不 import 任何业务模块，
    #: 否则 generate ↔ schemas 会绕成环。
    claims: list[Any] = field(default_factory=list)
    #: claim 侧打点（逐字率等），普通回答为 None
    claim_report: Any = None
    #: claim 一行都没解析出来、已降级成普通回答
    claim_fallback: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    model: str | None = None


def write_jsonl(path, rows: Iterable[dict[str, Any]]) -> int:
    """覆盖写 JSONL，返回写入行数。ensure_ascii=False 便于人工检查中文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path) -> Iterator[dict[str, Any]]:
    """逐行读 JSONL，容忍空行。"""
    if not path.is_file():
        return iter(())
    return _iter_jsonl(path)


def _iter_jsonl(path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_docs(path) -> list[Doc]:
    return [Doc(**row) for row in read_jsonl(path)]


def load_chunks(path) -> list[Chunk]:
    return [Chunk(**row) for row in read_jsonl(path)]
