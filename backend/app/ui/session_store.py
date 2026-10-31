"""会话落盘：一个会话一个 JSON 文件，放 ``storage/ui_sessions/``。

为什么不用数据库：M6 是单机、单用户的演示层，会话量按几十算；JSON 文件
足够，还能直接打开看。**不存语料绝对路径**，只存 ``rel_path``——隐私红线
（AGENTS.md 铁律 5）在展示层同样成立。

口径：
- 文件损坏 / schema 不认识 → ``load`` 返回 None，``list`` 跳过，不让 UI 崩；
- 写入用临时文件 + ``replace``，避免写到一半刷新留下半个 JSON；
- 标题取第一个提问的前 28 个字符，没有提问时为「新对话」。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
DEFAULT_TITLE = "新对话"
TITLE_MAX_CHARS = 28

_WS = re.compile(r"\s+")
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
LOCK_STALE_S = 300


def new_session(now: datetime | None = None) -> dict[str, Any]:
    """建一个空会话。``id`` 带时间前缀，目录按文件名排序就是大致的时间序。"""
    stamp = now or datetime.now().astimezone()
    return {
        "schema": SCHEMA_VERSION,
        "id": f"{stamp:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        "title": DEFAULT_TITLE,
        "created_at": _iso(stamp),
        "updated_at": _iso(stamp),
        "archived_at": None,
        "turns": [],
    }


def session_title(question: str) -> str:
    flat = _WS.sub(" ", (question or "").strip())
    if not flat:
        return DEFAULT_TITLE
    return flat[:TITLE_MAX_CHARS] + ("…" if len(flat) > TITLE_MAX_CHARS else "")


def append_turn(session: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    """把一轮问答追加进会话，并把标题/更新时间刷新。返回同一个 dict。"""
    session.setdefault("turns", []).append(turn)
    if session.get("title") in {"", DEFAULT_TITLE}:
        session["title"] = session_title(str(turn.get("question") or ""))
    session["updated_at"] = _iso(datetime.now().astimezone())
    return session


def save_session(session: dict[str, Any], base_dir: Path) -> Path:
    """原子写。只写 schema 认识的顶层字段，避免把运行时的临时键带进磁盘。"""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "id": str(session.get("id") or ""),
        "title": str(session.get("title") or DEFAULT_TITLE),
        "created_at": str(session.get("created_at") or ""),
        "updated_at": str(session.get("updated_at") or ""),
        "archived_at": session.get("archived_at"),
        "turns": list(session.get("turns") or []),
    }
    if not payload["id"]:
        raise ValueError("会话缺少 id")
    path = base_dir / f"{payload['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


@contextmanager
def _session_lock(session_id: str, base_dir: Path, timeout: float = 5.0) -> Iterator[None]:
    """进程内锁 + 会话级锁文件；跨 FastAPI/Streamlit 两个进程也串行化。

    锁文件放在 ``.locks/``，``list_sessions`` 的 ``*.json`` 扫描不会把它当会话。
    """
    key = f"{Path(base_dir).resolve()}::{session_id}"
    with _LOCKS_GUARD:
        thread_lock = _LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        lock_dir = Path(base_dir) / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{session_id}.lock"
        deadline = time.time() + timeout
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "at": datetime.now().isoformat()}).encode())
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                except OSError:
                    age = 0
                if age > LOCK_STALE_S:
                    try:
                        lock_path.unlink()
                    except OSError:
                        pass
                    continue
                if time.time() > deadline:
                    raise TimeoutError(f"会话正在被另一个进程写入：{session_id}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                lock_path.unlink()
            except OSError:
                pass


def append_turn_atomic(
    session_id: str,
    turn: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any] | None:
    """读改写在同一把锁内完成；返回追加后的会话，文件不存在返回 None。"""
    with _session_lock(session_id, base_dir):
        session = load_session(session_id, base_dir)
        if session is None:
            return None
        append_turn(session, turn)
        save_session(session, base_dir)
        return session


def load_session(session_id: str, base_dir: Path) -> dict[str, Any] | None:
    if not session_id or "/" in session_id or "\\" in session_id:
        return None
    path = Path(base_dir) / f"{session_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
        return None
    if not isinstance(data.get("turns"), list):
        return None
    return data


def list_sessions(
    base_dir: Path,
    limit: int = 40,
    *,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """按更新时间倒序返回会话摘要；默认只返回未归档，坏文件跳过。"""
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in base_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
            continue
        if bool(data.get("archived_at")) != archived:
            continue
        rows.append(
            {
                "id": str(data.get("id") or path.stem),
                "title": str(data.get("title") or DEFAULT_TITLE),
                "updated_at": str(data.get("updated_at") or ""),
                "archived_at": data.get("archived_at"),
                "n_turns": len(data.get("turns") or []),
            }
        )
    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    return rows[:limit]


def set_archived(
    session_id: str,
    base_dir: Path,
    *,
    archived: bool,
) -> dict[str, Any] | None:
    """归档/还原；读改写同一把锁，重复操作幂等。"""
    with _session_lock(session_id, base_dir):
        session = load_session(session_id, base_dir)
        if session is None:
            return None
        session["archived_at"] = _iso(datetime.now().astimezone()) if archived else None
        save_session(session, base_dir)
        return session


def delete_session(session_id: str, base_dir: Path) -> bool:
    """删除会话文件；文件不存在返回 False。"""
    with _session_lock(session_id, base_dir):
        path = Path(base_dir) / f"{session_id}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True


def relative_time(value: str, now: datetime | None = None) -> str:
    """``2026-10-30T01:20:00+08:00`` → 「刚刚 / 12 分钟前 / 昨天 / 10-28」。"""
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return ""
    current = now or datetime.now().astimezone()
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    seconds = max(0, int((current - stamp).total_seconds()))
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{seconds // 60} 分钟前"
    if seconds < 86400:
        return f"{seconds // 3600} 小时前"
    if seconds < 2 * 86400:
        return "昨天"
    if seconds < 7 * 86400:
        return f"{seconds // 86400} 天前"
    return stamp.strftime("%m-%d")


def _iso(stamp: datetime) -> str:
    return stamp.isoformat(timespec="seconds")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
