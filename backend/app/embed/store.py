"""向量索引：内存里的暴力检索 + 磁盘落盘。

为什么不上一套专门的向量库：当前是 1,949 个 chunk × 1024 维，
矩阵乘法一次约 2 MFLOP，耗时在毫秒级——SQLite + numpy 完全够用，
省掉一个 Docker 服务。到十万级 chunk 再谈 Weaviate / Milvus。
（RAGFlow 逆向报告的结论也是"130 份文档用 faiss/numpy 暴力检索就够"。）

落盘用 ``.npz``：chunk_ids 与向量一起存，避免两份文件不同步。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VectorIndex:
    chunk_ids: list[str]
    vectors: np.ndarray  # (n, dim) float32，已 L2 归一化
    model_name: str = ""
    built_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    #: 每行向量对应的 chunk 内容指纹。**复用向量必须靠它**，不能靠 chunk_id：
    #: M1 改了解析/分块后，chunk_id（doc_id:ordinal）可能没变而正文已经变了，
    #: 只按 id 复用会把旧文字的向量安到新文字上，检索结果对不上却毫无报错。
    content_hashes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors 必须是二维矩阵，当前 {self.vectors.shape}")
        if len(self.chunk_ids) != self.vectors.shape[0]:
            raise ValueError(
                f"chunk_ids 数量 {len(self.chunk_ids)} 与向量行数 {self.vectors.shape[0]} 不一致"
            )
        if self.content_hashes and len(self.content_hashes) != len(self.chunk_ids):
            raise ValueError(
                f"content_hashes 数量 {len(self.content_hashes)} 与向量行数 "
                f"{self.vectors.shape[0]} 不一致"
            )

    # ---- 构建 ----
    @classmethod
    def build(
        cls,
        chunk_ids: list[str],
        vectors: np.ndarray,
        model_name: str,
        content_hashes: list[str] | None = None,
    ) -> "VectorIndex":
        return cls(
            chunk_ids=list(chunk_ids),
            vectors=np.asarray(vectors, dtype=np.float32),
            model_name=model_name,
            content_hashes=list(content_hashes or []),
        )

    # ---- 查询 ----
    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        mask: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """余弦相似度 top-k。向量已归一化，所以点积即余弦。

        ``mask``：可选的布尔数组（长度 n），只在为 True 的行里选 top-k。
        这是"元数据过滤下推"的落点——先按允许集合缩小候选，再截断，
        而不是取回 top-k 之后再把不合格的丢掉（那样 k 条可能被丢光）。
        """
        if self.n == 0 or top_k <= 0:
            return []
        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"查询维度 {query.shape[0]} 与索引维度 {self.dim} 不一致")
        scores = self.vectors @ query
        if mask is None:
            allowed = None
            pool = scores
        else:
            allowed = np.flatnonzero(np.asarray(mask, dtype=bool))
            if allowed.size == 0:
                return []
            pool = scores[allowed]
        k = min(top_k, pool.shape[0])
        # 先取前 k 个（无序），再排序，避免全量排序
        candidates = np.argpartition(-pool, k - 1)[:k]
        if allowed is None:
            ordered = candidates[np.argsort(-scores[candidates])]
        else:
            ordered = allowed[candidates][np.argsort(-scores[allowed[candidates]])]
        return [(self.chunk_ids[int(i)], float(scores[int(i)])) for i in ordered]

    # ---- 落盘 ----
    def save(self, path: Path) -> Path:
        """原子写：先写 .tmp 再 replace，避免中断留下半截索引。

        （对应 RAGFlow 的 ``compensateFailedIndexWrite``：宁可没有，也不要半成品。）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "built_at": self.built_at,
            "n": self.n,
            "dim": self.dim,
        }
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez(
                fh,
                chunk_ids=np.array(self.chunk_ids, dtype=object),
                vectors=self.vectors,
                content_hashes=np.array(self.content_hashes, dtype=object),
                meta=np.array(json.dumps(meta, ensure_ascii=False)),
            )
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"向量索引不存在：{path}")
        with np.load(path, allow_pickle=True) as data:
            meta = json.loads(str(data["meta"]))
            hashes = (
                [str(x) for x in data["content_hashes"]]
                if "content_hashes" in data
                else []
            )
            index = cls(
                chunk_ids=[str(x) for x in data["chunk_ids"]],
                vectors=data["vectors"],
                model_name=meta.get("model_name", ""),
                built_at=meta.get("built_at", ""),
                content_hashes=hashes,
            )
        if index.dim != int(meta.get("dim", index.dim)):
            raise ValueError(
                f"索引维度与元数据不符：{index.dim} vs {meta.get('dim')}（文件可能损坏）"
            )
        return index

    def stats(self) -> dict[str, object]:
        return {
            "chunks": self.n,
            "dim": self.dim,
            "model": self.model_name,
            "built_at": self.built_at,
        }


def reuse_by_content(
    chunks: list, old: VectorIndex, model_name: str
) -> dict[str, np.ndarray]:
    """把旧索引里的向量按**内容指纹**认领回来。

    为什么必须按内容、不能按 chunk_id：chunk_id 是 `doc_id:ordinal`，M1 改了
    解析/分块后，同一序号可能还是同一份文档的第 N 个片段，但正文已经变了
    （表格压平 → Markdown、页眉页脚被删）。只按 id 认领会把旧文字的向量安到
    新文字上——检索能跑、结果不对，而且没有任何报错。旧索引没有内容指纹时
    一条都不复用，宁可多算 30 分钟，也不引入这种静默错。

    模型不同则一条都不复用——向量空间都变了，复用等于引入错误。
    """
    if not old.model_name or old.model_name != model_name:
        return {}
    if not old.content_hashes or len(old.content_hashes) != len(old.chunk_ids):
        return {}
    by_content = {
        digest: old.vectors[index]
        for index, digest in enumerate(old.content_hashes)
        if digest
    }
    reused: dict[str, np.ndarray] = {}
    for chunk in chunks:
        vector = by_content.get(chunk.content_hash)
        if vector is not None:
            reused[chunk.content_hash] = vector
    return reused
