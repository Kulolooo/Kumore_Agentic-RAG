"""重排（rerank）：对融合后的前 N 条做一次交叉编码打分。

为什么只重排前 N 条（默认 20）：cross-encoder 要把"查询 + 每条正文"拼起来过一遍模型，
成本与条数线性相关。RAGFlow 的取值是 20（`03-检索与排序.md` §5），我们先用同一个起点，
再决定要不要扩到 64。

**fail-open 是本项目相对 RAGFlow 的差异点**：RAGFlow 的 rerank 是唯一没有 fail-open
的环节，provider 抖动会把整次检索打成 error（`retrieval.go:199-202`）。
我们的约定是：重排模型加载失败、推理抛错、返回条数对不上——一律**保持检索原序**，
把失败原因记在 ``SearchResult.rerank_status`` / ``rerank_error`` 里，检索照常返回。
理由：重排是"锦上添花"的一层，它坏掉不该让一个本来能答的问题变成 500。

**做不到的事**：没有单次调用超时。CPU 上模型若卡死，进程会卡住而不是降级——
真要防这个需要子进程隔离，代价大于收益，先记为已知限制。
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Protocol

from app.embed.local import prepare_hf_env

log = logging.getLogger(__name__)


class RerankUnavailable(RuntimeError):
    """缺少依赖或权重；上层据此决定 fail-open 还是直接报错。"""


class Reranker(Protocol):
    """重排器的接口。测试里用一个假类就能覆盖 fail-open 分支，不必加载模型。"""

    name: str

    def rerank(self, query: str, texts: list[str]) -> list[float]:  # pragma: no cover - 协议
        ...


class CrossEncoderReranker:
    """本地 cross-encoder（默认 bge-reranker-base），CPU 推理、离线加载。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        model_dir: Path | None = None,
    ) -> None:
        self.name = model_name
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.model_dir = Path(model_dir) if model_dir else None
        self._model = None

    # ---- 懒加载：bm25 模式不装 torch 也能跑 ----
    def _load(self) -> None:
        if self._model is not None:
            return
        cache_dir = prepare_hf_env(self.model_dir)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RerankUnavailable(
                "未安装 sentence-transformers / torch，无法使用本地重排。"
                "安装：pip install -r requirements-embed.txt"
            ) from exc

        from app.embed.local import LocalEmbedder

        LocalEmbedder._apply_thread_limit()
        log.info("加载重排模型 %s（device=%s）", self.model_name, self.device)
        started = time.perf_counter()
        try:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        except Exception as exc:  # noqa: BLE001 - 网络/缓存/版本错误都归成一条可执行提示
            raise RerankUnavailable(
                f"重排模型加载失败：{type(exc).__name__}: {exc}\n"
                "默认离线（HF_HUB_OFFLINE=1）。首次使用需先下载权重：\n"
                "  $env:KUMORE_ALLOW_MODEL_DOWNLOAD='1'; python tools/m2_matrix.py --download-reranker\n"
                "或把模型放到 KUMORE_MODEL_DIR 指向的缓存目录。"
            ) from exc
        log.info("重排模型就绪，加载耗时 %.1fs", time.perf_counter() - started)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """返回与 ``texts`` 等长的分数；分数越高越相关。"""
        if not texts:
            return []
        self._load()
        pairs = [(query, text) for text in texts]
        raw = self._model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores: list[float] = []
        for value in list(raw):
            score = float(value)
            # NaN 会让排序变成不可预测的顺序；当成最低分，让 fail-open 的语义一致。
            scores.append(score if math.isfinite(score) else -1e9)
        if len(scores) != len(texts):
            raise ValueError(f"重排返回 {len(scores)} 条分数，候选 {len(texts)} 条")
        return scores

    @property
    def loaded(self) -> bool:
        return self._model is not None
