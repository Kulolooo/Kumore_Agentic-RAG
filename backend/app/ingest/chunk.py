"""分块入口（兼容层）。

M0 只有固定窗口一种切法，``chunk_blocks`` 就是全部实现；M1 之后四种策略都在
``app.ingest.chunkers``。这里保留旧签名，是为了不改动已有调用与测试，
同时让 ``from app.ingest.chunk import Block`` 这条老路径继续可用。
"""

from __future__ import annotations

from app.ingest.blocks import Block
from app.ingest.chunkers import ChunkStrategy, chunk_document

__all__ = ["Block", "ChunkStrategy", "chunk_blocks", "chunk_document"]


def chunk_blocks(
    doc_id: str,
    blocks: list[Block],
    chunk_size: int,
    chunk_overlap: int,
) -> list:
    """固定窗口分块（M0 基线），等价于 ``ChunkStrategy(name="fixed")``。"""
    config = ChunkStrategy(
        name="fixed", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return chunk_document(doc_id, blocks, config)
