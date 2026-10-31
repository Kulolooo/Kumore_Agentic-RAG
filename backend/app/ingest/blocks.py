"""解析产物 Block：一段带来源元数据的文本。

单独一个模块是为了让 ``parse`` 与 ``chunkers`` 都能引用它而不形成循环依赖
（``chunk.py`` 仍把名字再导出一次，老的 import 路径保持可用）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Block:
    """解析产物：一段带来源元数据的文本。

    ``kind`` 只区分 ``text`` / ``table``：分块时表格尽量整块保留，
    否则 512 token 的窗口会把表头切到上一个 chunk，数据行留给下一个。
    """

    text: str
    page: int | None = None
    section: str | None = None
    kind: str = "text"
