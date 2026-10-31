"""多知识库：注册、代际索引、原子重建与语料文件操作。

这一层只解决"当前用哪个库、索引放在哪、重建失败怎么办"；检索/生成仍走
``Retriever`` / ``AgentRunner``。CLI、FastAPI、Streamlit 都从这里解析设置，
避免评测读全局索引、UI 读另一个库。

目录结构（``storage/`` 本身 gitignore）::

    storage/
      models/                         # 共享权重，不按库复制
      kbs/
        active.json                   # {"id": "..."}
        <kb_id>/
          kb.json                     # 名称、语料目录、时间
          current.json                # {"generation": "gen-..."}
          gen-20261030-.../           # 不可变代际
            docs.jsonl
            chunks.jsonl
            index.sqlite
            vectors/
            ingest_report.json
          corpus/                     # 新建库的受管语料（可选）
          staging-.../                # 重建中间目录，失败即删

重建语义：新代际写到 staging，成功后才把 current.json 指向它；旧代际保留，
所以中途失败/指针写失败时旧索引仍然可读。没有采用"先删旧文件再写新文件"。
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas import Chunk, Doc

KB_SCHEMA = 1
GENERATION_PREFIX = "gen-"
STAGING_PREFIX = "staging-"
LOCK_STALE_S = 30 * 60
_MAX_NAME = 80
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,79}")

#: 进程内锁：CLI/uvicorn 多线程时避免同一库同时重建。跨进程用锁文件。
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@dataclass
class KnowledgeBase:
    id: str
    name: str
    corpus_root: Path
    managed_corpus: bool
    created_at: str
    updated_at: str
    dir: Path
    #: 每库检索模式；空 = 沿用全局 .env。只允许 bm25/vector/hybrid。
    retrieval_mode: str | None = None

    def to_dict(self, *, include_path: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "managed_corpus": self.managed_corpus,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "retrieval_mode": self.retrieval_mode or "",
        }
        if include_path:
            data["corpus_root"] = str(self.corpus_root)
        return data


# ---------------------------------------------------------------- 基础 IO
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def kb_root(storage_dir: Path | str) -> Path:
    return Path(storage_dir) / "kbs"


def _kb_dir(storage_dir: Path | str, kb_id: str) -> Path:
    return kb_root(storage_dir) / kb_id


def _load_kb(storage_dir: Path, kb_id: str) -> KnowledgeBase | None:
    if not _ID_RE.fullmatch(kb_id or ""):
        return None
    directory = _kb_dir(storage_dir, kb_id)
    data = _read_json(directory / "kb.json")
    if not data:
        return None
    try:
        corpus = Path(str(data["corpus_root"]))
    except (KeyError, TypeError):
        return None
    return KnowledgeBase(
        id=kb_id,
        name=str(data.get("name") or kb_id),
        corpus_root=corpus,
        managed_corpus=bool(data.get("managed_corpus")),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
        dir=directory,
        retrieval_mode=(
            str(data.get("retrieval_mode"))
            if data.get("retrieval_mode") in {"bm25", "vector", "hybrid"}
            else None
        ),
    )


def list_kbs(storage_dir: Path | str) -> list[KnowledgeBase]:
    root = kb_root(storage_dir)
    if not root.is_dir():
        return []
    rows: list[KnowledgeBase] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        kb = _load_kb(Path(storage_dir), directory.name)
        if kb is not None:
            rows.append(kb)
    rows.sort(key=lambda item: (item.updated_at, item.name), reverse=True)
    return rows


def _active_path(storage_dir: Path) -> Path:
    return kb_root(storage_dir) / "active.json"


def active_kb_id(storage_dir: Path | str) -> str | None:
    data = _read_json(_active_path(Path(storage_dir)))
    return str(data["id"]) if data and data.get("id") else None


def set_active(storage_dir: Path | str, kb_id: str) -> KnowledgeBase:
    kb = _load_kb(Path(storage_dir), kb_id)
    if kb is None:
        raise FileNotFoundError(f"知识库不存在：{kb_id}")
    _write_json(_active_path(Path(storage_dir)), {"id": kb.id})
    return kb


def set_retrieval_mode(
    storage_dir: Path | str,
    kb_id: str,
    mode: str | None,
) -> KnowledgeBase:
    """每库覆盖检索模式；None/空 = 沿用全局 .env。"""
    if mode not in {None, "", "bm25", "vector", "hybrid"}:
        raise ValueError(f"不支持的检索模式：{mode}")
    kb = _load_kb(Path(storage_dir), kb_id)
    if kb is None:
        raise FileNotFoundError(f"知识库不存在：{kb_id}")
    data = _read_json(kb.dir / "kb.json") or {}
    if mode:
        data["retrieval_mode"] = mode
    else:
        data.pop("retrieval_mode", None)
    data["updated_at"] = _now()
    _write_json(kb.dir / "kb.json", data)
    return _load_kb(Path(storage_dir), kb_id)  # type: ignore[return-value]


# ---------------------------------------------------------------- 代际
def current_generation(kb: KnowledgeBase) -> Path | None:
    data = _read_json(kb.dir / "current.json")
    if not data or not data.get("generation"):
        return None
    path = kb.dir / str(data["generation"])
    return path if path.is_dir() else None


def generation_settings(kb: KnowledgeBase, base: Settings) -> Settings:
    generation = current_generation(kb)
    if generation is None:
        raise FileNotFoundError(f"知识库还没有索引：{kb.id}")
    return dataclasses.replace(
        base,
        corpus_root=kb.corpus_root,
        storage_dir=generation,
        # 权重共享，避免每个库重复下载 embedding 模型。
        model_dir=base.model_dir,
        retrieval_mode=kb.retrieval_mode or base.retrieval_mode,
    )


def resolve_kb(storage_dir: Path | str, kb_id: str | None = None, base: Settings | None = None) -> KnowledgeBase:
    base = base or Settings.from_env()
    storage_dir = Path(storage_dir)
    _ensure_registry(storage_dir, base)
    if kb_id:
        kb = _load_kb(storage_dir, kb_id)
        if kb is None:
            raise FileNotFoundError(f"知识库不存在：{kb_id}")
        return kb
    chosen = active_kb_id(storage_dir)
    if chosen:
        kb = _load_kb(storage_dir, chosen)
        if kb is not None:
            return kb
    rows = list_kbs(storage_dir)
    if not rows:
        raise FileNotFoundError("没有可用知识库")
    set_active(storage_dir, rows[0].id)
    return rows[0]


def resolve_settings(
    kb_id: str | None = None,
    base: Settings | None = None,
) -> tuple[Settings, KnowledgeBase]:
    base = base or Settings.from_env()
    kb = resolve_kb(base.storage_dir, kb_id, base)
    return generation_settings(kb, base), kb


# ---------------------------------------------------------------- 初始化/迁移
def _ensure_registry(storage_dir: Path, base: Settings) -> None:
    root = kb_root(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    if list_kbs(storage_dir):
        chosen = active_kb_id(storage_dir)
        if not chosen or _load_kb(storage_dir, chosen) is None:
            set_active(storage_dir, list_kbs(storage_dir)[0].id)
        return

    legacy_has_index = any(
        (storage_dir / name).is_file()
        for name in ("index.sqlite", "docs.jsonl", "chunks.jsonl")
    )
    if legacy_has_index:
        _adopt_legacy(storage_dir, base)
    else:
        # 新机器：建一个受管默认库，语料目录为空；上传后重建即可。
        create_kb("默认知识库", storage_dir=storage_dir, base=base, make_active=True)


def _adopt_legacy(storage_dir: Path, base: Settings) -> KnowledgeBase:
    """把旧的单库索引复制进 ``kbs/default``，不删除旧文件（可回滚）。

    只复制索引，不复制语料；``corpus_root`` 沿用旧配置，所以 verify 仍能对账。
    """
    kb_id = "default"
    directory = _kb_dir(storage_dir, kb_id)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    generation = directory / _gen_name()
    generation.mkdir(parents=True, exist_ok=True)
    for name in ("docs.jsonl", "chunks.jsonl", "index.sqlite", "ingest_report.json"):
        source = storage_dir / name
        if source.is_file():
            shutil.copy2(source, generation / name)
    vectors = storage_dir / "vectors"
    if vectors.is_dir():
        shutil.copytree(vectors, generation / "vectors", dirs_exist_ok=True)

    _write_json(directory / "current.json", {"generation": generation.name})
    now = _now()
    _write_json(
        directory / "kb.json",
        {
            "schema": KB_SCHEMA,
            "id": kb_id,
            "name": "默认知识库",
            "corpus_root": str(base.corpus_root),
            "managed_corpus": False,
            "created_at": now,
            "updated_at": now,
        },
    )
    set_active(storage_dir, kb_id)
    return _load_kb(storage_dir, kb_id)  # type: ignore[return-value]


def create_kb(
    name: str,
    *,
    storage_dir: Path | str,
    base: Settings | None = None,
    corpus_root: Path | str | None = None,
    make_active: bool = True,
) -> KnowledgeBase:
    base = base or Settings.from_env()
    storage_dir = Path(storage_dir)
    root = kb_root(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    clean_name = (name or "").strip()[:_MAX_NAME] or "新知识库"
    kb_id = f"kb-{_stamp()}-{uuid.uuid4().hex[:4]}"
    directory = root / kb_id
    directory.mkdir(parents=True, exist_ok=False)

    if corpus_root is None:
        corpus = directory / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        managed = True
    else:
        corpus = Path(corpus_root).expanduser().resolve()
        if not corpus.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            raise FileNotFoundError(f"语料目录不存在：{corpus}")
        managed = False

    _write_json(
        directory / "kb.json",
        {
            "schema": KB_SCHEMA,
            "id": kb_id,
            "name": clean_name,
            "corpus_root": str(corpus),
            "managed_corpus": managed,
            "created_at": _now(),
            "updated_at": _now(),
        },
    )
    kb = _load_kb(storage_dir, kb_id)
    if kb is None:
        raise RuntimeError("知识库创建失败")
    _init_empty_generation(kb, base)
    if make_active:
        set_active(storage_dir, kb.id)
    return kb


def _init_empty_generation(kb: KnowledgeBase, base: Settings) -> None:
    from app.ingest.store import save_index

    generation = kb.dir / _gen_name()
    generation.mkdir(parents=True, exist_ok=True)
    staging = dataclasses.replace(
        base, corpus_root=kb.corpus_root, storage_dir=generation, model_dir=base.model_dir
    )
    save_index(staging, [], [])
    _write_json(kb.dir / "current.json", {"generation": generation.name})


# ---------------------------------------------------------------- 重建
class KBLockError(RuntimeError):
    pass


class _FileLock:
    """跨进程锁文件。超过 30 分钟视为残留（机器休眠/进程被杀）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 5
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                self.handle = fd
                os.write(fd, json.dumps({"pid": os.getpid(), "at": _now()}).encode("utf-8"))
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except OSError:
                    age = 0
                if age > LOCK_STALE_S:
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
                if time.time() > deadline:
                    raise KBLockError(f"知识库正在被另一个进程重建：{self.path}")
                time.sleep(0.1)

    def __exit__(self, *args: object) -> None:
        if self.handle is not None:
            try:
                os.close(self.handle)
            except OSError:
                pass
            self.handle = None
        try:
            self.path.unlink()
        except OSError:
            pass


def _lock_for(kb: KnowledgeBase) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(kb.id, threading.Lock())


def rebuild(
    kb_id: str | None = None,
    *,
    base: Settings | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> tuple[Any, Path]:
    """重建一代索引。成功返回 ``(stats, generation_dir)``。

    ``limit`` 只构建不提升（试跑语义，保留旧的 ``_limit_N`` 行为）。
    失败时 staging 删除，current.json 不变。
    """
    from app.ingest.pipeline import ingest

    base = base or Settings.from_env()
    kb = resolve_kb(base.storage_dir, kb_id, base)
    process_lock = _lock_for(kb)
    lock_path = kb.dir / "rebuild.lock"
    with process_lock, _FileLock(lock_path):
        previous = current_generation(kb)
        tag = _stamp()
        build_name = f"{STAGING_PREFIX}{tag}-{os.getpid()}"
        build_dir = kb.dir / build_name
        build_dir.mkdir(parents=True, exist_ok=False)
        # 向量按内容 hash 复用：把旧代的 vectors/ 带进 staging，embed 只编码新增/变更。
        if previous is not None and (previous / "vectors").is_dir():
            shutil.copytree(previous / "vectors", build_dir / "vectors", dirs_exist_ok=True)
        staging_settings = dataclasses.replace(
            base,
            corpus_root=kb.corpus_root,
            storage_dir=build_dir,
            model_dir=base.model_dir,
        )
        try:
            stats = ingest(
                staging_settings,
                limit=limit,
                verbose=verbose,
                previous_storage_dir=previous,
            )
            if limit is not None:
                return stats, build_dir
            generation = kb.dir / _gen_name()
            build_dir.replace(generation)
            _write_json(kb.dir / "current.json", {"generation": generation.name})
            data = _read_json(kb.dir / "kb.json") or {}
            data["updated_at"] = _now()
            _write_json(kb.dir / "kb.json", data)
            return stats, generation
        except Exception:
            shutil.rmtree(build_dir, ignore_errors=True)
            raise


def rollback(kb_id: str, generation: str, *, storage_dir: Path | str | None = None) -> None:
    base = Settings.from_env() if storage_dir is None else dataclasses.replace(
        Settings.from_env(), storage_dir=Path(storage_dir)
    )
    kb = resolve_kb(base.storage_dir, kb_id, base)
    target = kb.dir / generation
    if not target.is_dir():
        raise FileNotFoundError(f"代际不存在：{generation}")
    _write_json(kb.dir / "current.json", {"generation": target.name})


# ---------------------------------------------------------------- 语料文件
TEXT_EXTS = {".md", ".txt", ".csv", ".json", ".yml", ".yaml", ".py", ".html", ".htm", ".xml"}


def safe_corpus_path(kb: KnowledgeBase, rel_path: str) -> Path:
    raw = rel_path or ""
    if raw.startswith(("/", "\\")) or Path(raw).is_absolute():
        raise ValueError("不允许绝对路径")
    rel = raw.replace("\\", "/")
    if not rel or ".." in Path(rel).parts:
        raise ValueError("非法文件路径")
    root = kb.corpus_root.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError("文件路径越界")
    return target


def list_files(kb: KnowledgeBase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = kb.corpus_root
    if not root.is_dir():
        return rows
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "name": path.name,
                "ext": path.suffix.lower(),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "editable": path.suffix.lower() in TEXT_EXTS,
            }
        )
    rows.sort(key=lambda row: row["path"])
    return rows


def read_file(kb: KnowledgeBase, rel_path: str, *, max_bytes: int = 2_000_000) -> bytes:
    path = safe_corpus_path(kb, rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"文件过大：{path.stat().st_size} bytes")
    return path.read_bytes()


def write_file(kb: KnowledgeBase, rel_path: str, data: bytes, *, overwrite: bool = True) -> Path:
    path = safe_corpus_path(kb, rel_path)
    if path.exists() and not overwrite:
        raise FileExistsError(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def delete_file(kb: KnowledgeBase, rel_path: str) -> None:
    path = safe_corpus_path(kb, rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    path.unlink()


def rename_file(kb: KnowledgeBase, rel_path: str, new_path: str) -> Path:
    source = safe_corpus_path(kb, rel_path)
    target = safe_corpus_path(kb, new_path)
    if not source.is_file():
        raise FileNotFoundError(rel_path)
    if target.exists():
        raise FileExistsError(new_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    return target


def read_text(kb: KnowledgeBase, rel_path: str) -> str:
    path = safe_corpus_path(kb, rel_path)
    if path.suffix.lower() not in TEXT_EXTS:
        raise ValueError("不是可编辑文本文件")
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(kb: KnowledgeBase, rel_path: str, text: str) -> Path:
    path = safe_corpus_path(kb, rel_path)
    if path.suffix.lower() not in TEXT_EXTS:
        raise ValueError("不是可编辑文本文件")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- 统计
def index_stats(kb: KnowledgeBase, base: Settings) -> dict[str, Any]:
    try:
        settings = generation_settings(kb, base)
    except FileNotFoundError:
        return {"ready": False, "docs": 0, "chunks": 0}
    from app.ingest.store import load_index

    docs, chunks = load_index(settings)
    try:
        from app.visual_index import status as visual_status

        visual = visual_status(settings)
    except Exception:
        visual = {}
    return {
        "ready": True,
        "generation": settings.storage_dir.name,
        "docs": len(docs),
        "chunks": len([c for c in chunks if not c.is_parent]),
        "images": int(visual.get("total", 0) or 0),
        "visual": visual,
        "mode": settings.retrieval_mode,
    }


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _gen_name() -> str:
    # 同一秒内创建空代际 + 立刻重建会撞名；加随机后缀，避免目录已存在。
    return f"{GENERATION_PREFIX}{_stamp()}-{uuid.uuid4().hex[:4]}"
