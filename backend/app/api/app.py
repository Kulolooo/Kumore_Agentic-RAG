"""M6.5：给 React 前端的 FastAPI 薄层。

边界（和 Streamlit 版一致）：

- **不重写**检索/生成：直接调 ``app.ui.service.run_turn``，引用/拒答口径不变；
- **不落敏感路径**：turn 里的 ``rel_path`` 已由 ``mask_private`` 处理；
- **不向前端发绝对路径**；``/api/chunks`` 只按 chunk_id 回原文；
- 研究过程用现有 ``AgentRunner(on_step=...)`` 转成 SSE ``step`` 事件；
- 最终答案目前仍是一次性返回（``answer`` 事件），token 流式留给下一轮。

启动：
    python -m app web --port 8000
    # 前端开发：cd web && pnpm dev  （Vite 代理 /api → 8000）
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app import kb as kb_service
from app.config import PROJECT_ROOT, Settings
from app.ingest.store import load_images
from app.retrieve.retriever import Retriever
from app.ui import session_store
from app.ui.format import mask_private
from app import visual_index as visual_service
from app.ui.service import (
    UIConfig,
    build_client,
    build_retriever,
    ensure_session_dir,
    run_turn,
)

WEB_DIST = PROJECT_ROOT / "web" / "dist"


# ---------------------------------------------------------------- 运行时
@dataclass
class Runtime:
    settings: Settings
    retriever: Retriever
    client: Any | None  # DeepSeekClient；无 key 时为 None
    kb: kb_service.KnowledgeBase
    visual_mtime: float = 0.0


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    """当前知识库的索引只加载一次；切库/重建后显式 cache_clear（见 _reload_runtime）。"""
    settings, kb = kb_service.resolve_settings()
    retriever = build_retriever(settings)
    client = build_client(settings) if settings.deepseek_api_key else None
    visual_mtime = (
        settings.visual_vector_index_path.stat().st_mtime
        if settings.visual_vector_index_path.is_file()
        else 0.0
    )
    return Runtime(
        settings=settings,
        retriever=retriever,
        client=client,
        kb=kb,
        visual_mtime=visual_mtime,
    )


def _reload_runtime() -> None:
    get_runtime.cache_clear()
    _prewarm_async(get_runtime())


def _reload_if_visual_changed() -> None:
    """worker 原子替换视觉索引后，让下一次查询用上新段。"""
    try:
        runtime = get_runtime()
        path = runtime.settings.visual_vector_index_path
        mtime = path.stat().st_mtime if path.is_file() else 0.0
        if mtime > runtime.visual_mtime:
            _reload_runtime()
    except Exception:  # noqa: BLE001 - 状态轮询不能因重建/空库失败
        return


# ---------------------------------------------------------------- 请求体
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["agent", "single"] = "agent"
    top_k: int = Field(default=5, ge=1, le=20)
    prompt_variant: Literal["claim", "strict", "tight"] = "claim"
    include_superseded: bool = False
    #: 合成输出上限。默认 1024 足够 claim 答案（实测 30–260 tok），
    #: 只是防止模型话痨；大于 4096 没有业务收益，直接在协议层拒绝。
    max_output_tokens: int = Field(default=1024, ge=128, le=4096)
    #: 单 turn 预算（研究轮 + 合成）；默认沿用 M4 的 120k。
    max_tokens_budget: int = Field(default=120_000, ge=10_000, le=200_000)
    #: 研究轮上限；设置页可调，默认沿用 M4 的 10。
    max_steps: int = Field(default=10, ge=1, le=20)
    #: 可选模型覆盖；空 = 用 .env 的默认模型。
    model: str | None = Field(default=None, max_length=80)
    #: 服务端会话 id；为空则本次不落盘（兼容旧客户端/临时问答）。
    session_id: str | None = Field(default=None, max_length=80)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("问题不能为空")
        return text

    def to_config(self) -> UIConfig:
        return UIConfig(
            mode=self.mode,
            top_k=self.top_k,
            prompt_variant=self.prompt_variant,
            include_superseded=self.include_superseded,
            finalize_max_tokens=self.max_output_tokens,
            max_tokens=self.max_tokens_budget,
            max_steps=self.max_steps,
            stream=True,
        )


# ---------------------------------------------------------------- 应用
app = FastAPI(title="Kumore API", version="0.1.0")


_PREWARM_STARTED: set[int] = set()


def _prewarm_async(runtime: Runtime) -> None:
    """后台预热 bge-m3；重复调用去重，测试/显式关闭时不启动。"""
    if os.environ.get("KUMORE_DISABLE_PREWARM") == "1":
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    key = id(runtime)
    if key in _PREWARM_STARTED:
        return
    _PREWARM_STARTED.add(key)

    def work() -> None:
        try:
            runtime.retriever.prewarm()
        except Exception:
            # 空库/缺依赖时保持 BM25 可用；真实错误由 health 暴露。
            return

    threading.Thread(target=work, name="kumore-text-prewarm", daemon=True).start()


@app.on_event("startup")
def _prewarm_text_tower() -> None:
    try:
        _prewarm_async(get_runtime())
    except Exception:
        return


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ready_runtime() -> Runtime:
    try:
        runtime = get_runtime()
    except Exception as exc:  # 索引为空 / 一致性失败 / 依赖缺失
        raise HTTPException(status_code=503, detail=f"索引未就绪：{exc}") from exc
    if not runtime.retriever.chunks:
        raise HTTPException(status_code=503, detail="索引为空，请先运行 python -m app ingest")
    return runtime


def _require_client(runtime: Runtime) -> Any:
    if runtime.client is None:
        raise HTTPException(status_code=503, detail="未配置 DEEPSEEK_API_KEY")
    return runtime.client


def _client_for(payload: AskRequest, runtime: Runtime) -> Any:
    """模型覆盖只影响本次 turn；不重建索引、不写 .env。"""
    base = _require_client(runtime)
    model = (payload.model or "").strip()
    if not model or model == runtime.settings.deepseek_model:
        return base
    # 同一个 key/base/timeout，只换 model；失败由 ask 的 LLMError 分支处理。
    return build_client(
        dataclasses.replace(runtime.settings, deepseek_model=model)
    )


# ---------------------------------------------------------------- 会话
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,79}")


def _session_dir() -> Path:
    # 不依赖索引：会话列表/新建在索引坏掉时也应该能读。
    return ensure_session_dir(Settings.from_env().storage_dir)


def _valid_session_id(session_id: str) -> bool:
    return bool(_SESSION_ID.fullmatch(session_id or ""))


def _session_or_404(session_id: str) -> dict[str, Any]:
    if not _valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="非法会话 id")
    session = session_store.load_session(session_id, _session_dir())
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


def _summary(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(session.get("id") or ""),
        "title": str(session.get("title") or "新对话"),
        "created_at": str(session.get("created_at") or ""),
        "updated_at": str(session.get("updated_at") or ""),
        "archived_at": session.get("archived_at"),
        "n_turns": len(session.get("turns") or []),
    }


def _persist_turn(session_id: str | None, turn: dict[str, Any]) -> str:
    """把 turn 追加到服务端会话；失败只返回原因，不吞掉已经生成的答案。

    读改写由 ``session_store.append_turn_atomic`` 的进程内锁 + 会话锁文件保护，
    Streamlit/另一个 API 进程同时追加不会后写覆盖。
    """
    if not session_id:
        return ""
    if not _valid_session_id(session_id):
        return "非法会话 id，未保存"
    try:
        session = session_store.append_turn_atomic(session_id, turn, _session_dir())
    except TimeoutError as exc:
        return f"{exc}，未保存"
    if session is None:
        return "会话不存在，未保存"
    return ""


class KBCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    corpus_root: str | None = Field(default=None, max_length=1000)


class KeyUpdate(BaseModel):
    # 只允许本地客户端写；不回显、不写日志。
    api_key: str = Field(default="", max_length=300)


def _write_env_key(key: str) -> None:
    """把 DEEPSEEK_API_KEY 写进项目 .env，保留其它行。"""
    env_path = PROJECT_ROOT / ".env"
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    found = False
    output: list[str] = []
    for line in lines:
        if line.strip().startswith("DEEPSEEK_API_KEY="):
            output.append(f"DEEPSEEK_API_KEY={key}")
            found = True
        else:
            output.append(line)
    if not found:
        output.append(f"DEEPSEEK_API_KEY={key}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = env_path.with_name(".env.tmp")
    tmp.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    tmp.replace(env_path)


@app.post("/api/settings/key")
def api_set_key(payload: KeyUpdate) -> dict[str, Any]:
    """本地配置 API Key；空字符串 = 清除。"""
    key = payload.api_key.strip()
    try:
        _write_env_key(key)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"写入 .env 失败：{exc}") from exc
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
    else:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        _reload_runtime()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重新加载模型失败：{exc}") from exc
    return {"configured": bool(key), "api_ready": bool(key)}


@app.get("/api/images/{image_id}")
def api_image(image_id: str) -> FileResponse:
    runtime = _ready_runtime()
    images = {i.image_id: i for i in load_images(runtime.settings)}
    image = images.get(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    path = runtime.settings.assets_dir / image.image_file
    if not path.is_file():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    return FileResponse(path, media_type="image/png")


@app.get("/api/kbs")
def api_list_kbs() -> dict[str, Any]:
    _reload_if_visual_changed()
    base = Settings.from_env()
    kb_service.resolve_kb(base.storage_dir, None, base)  # 首次迁移/建默认库
    rows = kb_service.list_kbs(base.storage_dir)
    active = kb_service.active_kb_id(base.storage_dir)
    return {
        "active": active,
        "kbs": [
            {**kb.to_dict(), "index": kb_service.index_stats(kb, base)}
            for kb in rows
        ],
    }


@app.post("/api/kbs")
def api_create_kb(payload: KBCreate) -> dict[str, Any]:
    base = Settings.from_env()
    try:
        kb = kb_service.create_kb(
            payload.name,
            storage_dir=base.storage_dir,
            base=base,
            corpus_root=payload.corpus_root or None,
            make_active=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
    _reload_runtime()
    return {**kb.to_dict(), "index": kb_service.index_stats(kb, base)}


@app.post("/api/kbs/{kb_id}/activate")
def api_activate_kb(kb_id: str) -> dict[str, Any]:
    base = Settings.from_env()
    try:
        kb = kb_service.set_active(base.storage_dir, kb_id)
        _reload_runtime()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {**kb.to_dict(), "index": kb_service.index_stats(kb, base)}


@app.get("/api/kbs/{kb_id}/visual")
def api_visual_status(kb_id: str) -> dict[str, Any]:
    _reload_if_visual_changed()
    base, kb = _kb_or_404(kb_id)
    settings = kb_service.generation_settings(kb, base)
    data = visual_service.status(settings)
    data["kb"] = kb.id
    return data


def _schedule_visual_worker(kb_id: str) -> None:
    """懒编码：独立进程；不阻塞当前 turn。"""
    env = dict(os.environ)
    env.setdefault("KUMORE_VISUAL_MMAP", "0")
    subprocess.Popen(
        [sys.executable, "-m", "app.visual_worker", "--kb", kb_id],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _enqueue_visual(runtime: Runtime, turn: dict[str, Any]) -> None:
    try:
        citations = turn.get("citations") or []
        chunk_ids = [str(c.get("chunk_id")) for c in citations if c.get("chunk_id")]
        # 懒编码优先整个 top-K/上下文，不只被引用的那几条。
        chunk_ids.extend(str(cid) for cid in (turn.get("context_chunk_ids") or []))
        missing = visual_service.enqueue_from_chunks(runtime.settings, chunk_ids)
        if missing:
            _schedule_visual_worker(runtime.kb.id)
    except Exception:  # noqa: BLE001 - 懒编码失败不能影响回答
        return


@app.get("/api/kbs/{kb_id}")
def api_get_kb(kb_id: str) -> dict[str, Any]:
    base = Settings.from_env()
    try:
        kb = kb_service.resolve_kb(base.storage_dir, kb_id, base)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {**kb.to_dict(), "index": kb_service.index_stats(kb, base)}


@app.post("/api/kbs/{kb_id}/rebuild")
def api_rebuild_kb(kb_id: str) -> dict[str, Any]:
    base = Settings.from_env()
    try:
        stats, generation = kb_service.rebuild(kb_id, base=base, verbose=False)
        kb_service.set_active(base.storage_dir, kb_id)
        _reload_runtime()
    except kb_service.KBLockError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    kb = kb_service.resolve_kb(base.storage_dir, kb_id, base)
    return {
        "ok": True,
        "stats": stats.to_dict(),
        "generation": generation.name,
        "kb": {**kb.to_dict(), "index": kb_service.index_stats(kb, base)},
    }


class KBRename(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    new_path: str = Field(min_length=1, max_length=1000)


class KBText(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    text: str = Field(max_length=5_000_000)


def _kb_or_404(kb_id: str) -> tuple[Settings, Any]:
    base = Settings.from_env()
    try:
        kb = kb_service.resolve_kb(base.storage_dir, kb_id, base)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return base, kb


def _after_change(kb_id: str, base: Settings, rebuild: bool) -> dict[str, Any]:
    if not rebuild:
        return {"ok": True, "rebuild_error": ""}
    try:
        stats, generation = kb_service.rebuild(kb_id, base=base, verbose=False)
        kb_service.set_active(base.storage_dir, kb_id)
        _reload_runtime()
        return {
            "ok": True,
            "rebuild_error": "",
            "stats": stats.to_dict(),
            "generation": generation.name,
        }
    except Exception as exc:
        # 文件已经写入/删除，但旧索引仍可读；把失败原因返回给 UI，不伪装成功。
        return {"ok": True, "rebuild_error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/kbs/{kb_id}/files")
def api_list_files(kb_id: str) -> dict[str, Any]:
    base, kb = _kb_or_404(kb_id)
    return {
        "kb": {**kb.to_dict(), "index": kb_service.index_stats(kb, base)},
        "files": kb_service.list_files(kb),
    }


@app.get("/api/kbs/{kb_id}/files/text")
def api_read_text(kb_id: str, path: str) -> dict[str, Any]:
    _, kb = _kb_or_404(kb_id)
    try:
        return {"path": path, "text": kb_service.read_text(kb, path)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/kbs/{kb_id}/files/text")
def api_write_text(kb_id: str, payload: KBText, rebuild: bool = True) -> dict[str, Any]:
    base, kb = _kb_or_404(kb_id)
    try:
        kb_service.write_text(kb, payload.path, payload.text)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": payload.path, **_after_change(kb_id, base, rebuild)}


@app.get("/api/kbs/{kb_id}/files/download")
def api_download_file(kb_id: str, path: str) -> FileResponse:
    _, kb = _kb_or_404(kb_id)
    try:
        target = kb_service.safe_corpus_path(kb, path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target, filename=target.name)


@app.post("/api/kbs/{kb_id}/files/upload")
async def api_upload_file(
    kb_id: str,
    file: UploadFile = File(...),
    path: str | None = Form(default=None),
    rebuild: bool = Form(default=True),
) -> dict[str, Any]:
    base, kb = _kb_or_404(kb_id)
    name = Path(file.filename or "upload.bin").name
    rel = (path or name).replace("\\", "/")
    if rel.endswith("/"):
        rel = rel + name
    if Path(rel).suffix.lower() not in base.ext_whitelist:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{Path(rel).suffix}")
    data = await file.read()
    if len(data) > base.max_file_bytes:
        raise HTTPException(status_code=413, detail="文件超过配置的大小上限")
    try:
        kb_service.write_file(kb, rel, data, overwrite=True)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": rel, **_after_change(kb_id, base, rebuild)}


@app.post("/api/kbs/{kb_id}/files/rename")
def api_rename_file(kb_id: str, payload: KBRename, rebuild: bool = True) -> dict[str, Any]:
    base, kb = _kb_or_404(kb_id)
    try:
        kb_service.rename_file(kb, payload.path, payload.new_path)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": payload.new_path, **_after_change(kb_id, base, rebuild)}


@app.delete("/api/kbs/{kb_id}/files")
def api_delete_file(kb_id: str, path: str, rebuild: bool = True) -> dict[str, Any]:
    base, kb = _kb_or_404(kb_id)
    try:
        kb_service.delete_file(kb, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": path, **_after_change(kb_id, base, rebuild)}


class SessionCreate(BaseModel):
    title: str = Field(default="新对话", max_length=80)


@app.get("/api/sessions")
def list_sessions(archived: bool = False) -> dict[str, Any]:
    rows = session_store.list_sessions(_session_dir(), limit=50, archived=archived)
    return {"sessions": rows, "archived": archived}


@app.post("/api/sessions")
def create_session(payload: SessionCreate | None = None) -> dict[str, Any]:
    session = session_store.new_session()
    if payload and payload.title.strip():
        session["title"] = payload.title.strip()[:80]
    session_store.save_session(session, _session_dir())
    return _summary(session)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    return _session_or_404(session_id)


@app.post("/api/sessions/{session_id}/archive")
def archive_session(session_id: str) -> dict[str, Any]:
    session = session_store.set_archived(session_id, _session_dir(), archived=True)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, **_summary(session)}


@app.post("/api/sessions/{session_id}/unarchive")
def unarchive_session(session_id: str) -> dict[str, Any]:
    session = session_store.set_archived(session_id, _session_dir(), archived=False)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, **_summary(session)}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    session = _session_or_404(session_id)
    if not session_store.delete_session(session["id"], _session_dir()):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, "id": session["id"]}


@app.get("/api/health")
def health() -> dict[str, Any]:
    """返回前端需要的状态；索引坏掉也返回 200，让前端能显示 setup 而不是白屏。"""
    try:
        runtime = get_runtime()
    except Exception as exc:
        return {
            "ok": False,
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "api_ready": False,
            "frontend_built": WEB_DIST.is_dir(),
        }
    retriever = runtime.retriever
    return {
        "ok": True,
        "ready": bool(retriever.chunks),
        "api_ready": runtime.client is not None,
        "frontend_built": WEB_DIST.is_dir(),
        "docs": len(retriever.docs_by_id),
        "chunks": len([c for c in retriever.chunks if not c.is_parent]),
        "mode": retriever.mode,
        "semantic_ready": bool(retriever.text_ready),
        "model": runtime.settings.deepseek_model,
        "kb": runtime.kb.id,
        "kb_name": runtime.kb.name,
        "generation": runtime.settings.storage_dir.name,
    }


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    """同步一问一答（测试/降级用）；React 主路径走 SSE。"""
    runtime = _ready_runtime()
    client = _client_for(payload, runtime)
    if payload.session_id:
        _session_or_404(payload.session_id)
    try:
        turn = run_turn(payload.question, runtime.retriever, client, payload.to_config())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    _enqueue_visual(runtime, turn)
    warning = _persist_turn(payload.session_id, turn)
    if warning:
        turn.setdefault("warnings", []).append(warning)
    return turn


def _sse(event: str, data: dict[str, Any]) -> str:
    body = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


@app.post("/api/ask/stream")
async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    """SSE：meta → step* → token* → answer → done。

    agent 循环在独立线程里跑；步骤/合成 token 通过回调塞进 asyncio 队列。
    客户端断开时置 ``cancel_event``，流式合成会在下一帧停止继续读取
    （研究轮的模型调用无法中途打断；已知限制）。
    """
    runtime = _ready_runtime()
    client = _client_for(payload, runtime)
    if payload.session_id:
        _session_or_404(payload.session_id)
    config = payload.to_config()
    question = payload.question

    async def event_stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        cancel_event = threading.Event()
        started = time.monotonic()

        def on_step(step: Any) -> None:
            # 线程 → 事件循环；put_nowait 的队列不设上限，不会阻塞 agent。
            loop.call_soon_threadsafe(queue.put_nowait, ("step", step.to_dict()))

        def on_token(text: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("token", {"text": text}))

        def worker() -> None:
            try:
                turn = run_turn(
                    question,
                    runtime.retriever,
                    client,
                    config,
                    on_step=on_step,
                    on_token=on_token,
                    cancel_event=cancel_event,
                )
                # 客户端已取消：不要把降级/半截 turn 写进会话，也不发 answer。
                if cancel_event.is_set():
                    return
                _enqueue_visual(runtime, turn)
                # 先把 turn 落盘，再发 answer：前端收到 answer 后刷新会话列表，
                # 不会出现"答案到了但列表还是旧标题"。
                warning = _persist_turn(payload.session_id, turn)
                if warning:
                    turn.setdefault("warnings", []).append(warning)
                loop.call_soon_threadsafe(queue.put_nowait, ("answer", turn))
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("error", {"message": f"{type(exc).__name__}: {exc}"}),
                )
            finally:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("done", {"elapsed_s": round(time.monotonic() - started, 2)}),
                )

        disconnected = asyncio.Event()

        async def watch_disconnect() -> None:
            """不能把 ``is_disconnected()`` 放主循环里：它会阻塞等 ASGI disconnect
            消息，一旦主循环 await 它，step/token 就发不出去（v1 踩过）。"""
            try:
                while not disconnected.is_set():
                    if await request.is_disconnected():
                        disconnected.set()
                        cancel_event.set()
                        return
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:
                return

        yield _sse(
            "meta",
            {
                "question": question,
                "mode": config.mode,
                "top_k": config.top_k,
                "prompt_variant": config.prompt_variant,
                "model": getattr(client, "model", ""),
            },
        )
        thread = threading.Thread(target=worker, name="kumore-ask", daemon=True)
        thread.start()
        watcher = asyncio.create_task(watch_disconnect())

        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    # 研究轮可能一次模型调用就 30s+；发注释行保活，别让代理掐断。
                    if disconnected.is_set():
                        break
                    yield ": ping\n\n"
                    continue
                if disconnected.is_set():
                    break
                yield _sse(event, data)
                if event == "done":
                    break
        finally:
            # 线程是 daemon；断开后让流式合成在下一帧停止，尽量少烧 token。
            cancel_event.set()
            watcher.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/chunks/{chunk_id}")
def chunk_detail(chunk_id: str) -> dict[str, Any]:
    """按 chunk_id 回原文，供证据抽屉用。

    父子块策略下 ``chunk_id`` 是命中的子块，但模型看到的是父块；抽屉必须展示
    父块正文，否则用户核对的上下文比模型少一截（M6/M6.5 的 B-1）。
    """
    runtime = _ready_runtime()
    chunks = runtime.retriever.chunks
    chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk 不在当前索引里")
    shown = chunk
    if chunk.mom_id:
        parent = next((c for c in chunks if c.chunk_id == chunk.mom_id), None)
        if parent is not None:
            shown = parent
    doc = runtime.retriever.docs_by_id.get(chunk.doc_id)
    image_url = f"/api/images/{chunk.image_id}" if chunk.image_id else None
    return {
        "chunk_id": chunk.chunk_id,
        "context_chunk_id": shown.chunk_id,
        "image_id": chunk.image_id,
        "image_url": image_url,
        "doc_id": chunk.doc_id,
        "rel_path": mask_private(doc.rel_path if doc else ""),
        "page": shown.page,
        "page_end": shown.page_end,
        "section": shown.section,
        "text": shown.text,
        "is_parent": bool(shown.is_parent),
    }


# ---------------------------------------------------------------- 静态前端
if WEB_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    assets = WEB_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        target = (WEB_DIST / full_path).resolve()
        try:
            target.relative_to(WEB_DIST.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="not found")
        if full_path and target.is_file():
            return FileResponse(target)
        index = WEB_DIST / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="frontend not built")
        # index 引用的 JS/CSS 带 hash；index 自身不缓存，避免构建后仍拿旧入口。
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

else:

    @app.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "message": "API 已启动；前端未构建。开发时运行 cd web && pnpm dev（5173）。",
                "docs": "/docs",
            }
        )
