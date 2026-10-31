"""本地 embedding：把文本变成向量，全程不出本机。"""

from app.embed.local import EmbedderUnavailable, LocalEmbedder
from app.embed.store import VectorIndex

__all__ = ["LocalEmbedder", "EmbedderUnavailable", "VectorIndex"]
