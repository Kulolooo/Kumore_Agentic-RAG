"""模型响应缓存。

解决的是实测缺陷「同一问题逐字不可复现」：``temperature=0`` 并不能让远端模型逐字稳定
（实测 6 次里出过 2 种文本）。这个随机性在**客户端无法消除**，只能：

1. 缓存 —— 同一请求内容直接返回上次结果，重复运行逐字一致，也省钱；
2. 多次取样报告一致率 —— 承认方差存在，别拿单次数字当结论。

缓存键 = (模型, 请求体) 的哈希，包含 messages 全文与所有采样参数；
因此语料或提示词变了，键就变了，不会拿到过期答案。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS responses (
                       key        TEXT PRIMARY KEY,
                       payload    TEXT NOT NULL,
                       created_at TEXT NOT NULL DEFAULT (datetime('now'))
                   )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    @staticmethod
    def make_key(model: str, payload: dict) -> str:
        blob = json.dumps(
            {"model": model, "payload": payload}, ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, result: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses(key, payload) VALUES (?, ?)",
                (key, json.dumps(result, ensure_ascii=False)),
            )

    def size(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
