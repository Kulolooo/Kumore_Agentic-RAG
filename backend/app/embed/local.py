"""本地 embedding 模型（默认 bge-visualized-m3），CPU 推理。

为什么值得单独一层：M0 的隐私边界是「文件名不外泄，正文照发」。一旦上向量检索，
如果 embedding 也走云端，出网量会从"每次 5 段"变成"全库每个 chunk 一次"。
放到本地跑，索引就完全不出门，只有问答那一刻的 top-k 片段出网。

两个后端：

1. **bge-m3**（``BAAI/bge-m3``）：纯文本，sentence-transformers 直接加载；
2. **bge-visualized-m3**（``BAAI/bge-visualized-m3``）：BGE-M3 文本塔 + EVA-CLIP
   视觉塔的联合模型，权重是 ``Visualized_m3.pth``，不能走 sentence-transformers，
   必须用内嵌的 ``visual_bge.modeling.Visualized_BGE`` 加载。

为什么换到 visualized 版：纯文本塔与 bge-m3 是同一套权重（CLS pooling + L2 归一化），
所以文本检索的行为不变；换的收益是同一个向量空间里还能编码图片，给后续
"扫描件 / 图片型文档"入库留出通路。代价是权重 1.75 GB、fp32 加载后常驻约 3.5 GB，
比纯文本模型多一个视觉塔。

刻意**不**给查询加指令前缀：bge-m3 不像 bge-v1.5 那样要求 "为这个句子生成表示…" 之类的
query instruction，加了反而偏离模型卡片的用法。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: bge-visualized-m3 的组合方式：视觉权重 + 这个文本塔。
#: 上游 ``Visualized_BGE`` 靠子串 ``bge-m3`` 选 EVA02-CLIP-L-14，不能随意改名。
VISUAL_TEXT_MODEL = "BAAI/bge-m3"

#: 视觉权重文件名。HF 仓库是 ``BAAI/bge-visualized``（不是 ``bge-visualized-m3``），
#: 模型名里的 ``-m3`` 对应仓库里的这个文件。
VISUAL_WEIGHT_FILENAME = "Visualized_m3.pth"

VISUAL_WEIGHT_URL = (
    "https://hf-mirror.com/BAAI/bge-visualized/resolve/main/Visualized_m3.pth"
)


class EmbedderUnavailable(RuntimeError):
    """缺少依赖或权重时的明确报错，而不是让 ImportError 冒到上层。"""


def is_visualized_model(model_name: str | None) -> bool:
    """判断是否走 Visualized-BGE 后端。

    用子串而不是白名单：上游 HF 仓库叫 ``BAAI/bge-visualized``，模型名叫
    ``BAAI/bge-visualized-m3``，本地目录名还可能带路径分隔符。只要名字里有
    ``visualized`` 就按视觉模型处理，避免"换了个写法就静默走错后端"。
    """
    return "visualized" in (model_name or "").lower()


def prepare_hf_env(model_dir: Path | None) -> Path | None:
    """统一 HF 缓存目录与离线开关，返回实际使用的 cache_dir。

    抽出来给 embedding 与 reranker 共用：两者都要满足"索引侧不出网"，
    任何一处漏设都会让模型在加载时去 huggingface.co 发 HEAD 请求（P38）。
    """
    cache_dir = (Path(model_dir) / "hf") if model_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        # SentenceTransformer 用 cache_folder 参数找权重，而 Visualized-BGE 走
        # transformers 的 AutoConfig/AutoTokenizer，后者只认 HF_HUB_CACHE。
        # 不设这一条时，即使 bge-m3 已经下载在项目缓存里，离线加载仍会报
        # "couldn't find it in the cached files"（P86）。
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    # 关掉匿名使用统计与符号链接警告（本项目的原则：默认不外发）
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # **默认离线**：模型库加载时会向 HF 发 HEAD 请求检查更新
    # （实测 modules.json 超时后重试 5 次，每次 10s，索引建到一半像卡死）。
    # 权重已缓存在 model_dir 时直接读缓存；确实要首次下载才显式开
    # KUMORE_ALLOW_MODEL_DOWNLOAD=1。
    if os.environ.get("KUMORE_ALLOW_MODEL_DOWNLOAD", "0").strip() not in {
        "1", "true", "yes",
    }:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return cache_dir


def _build_visualized_bge(weight: Path):
    """构造 Visualized-BGE，并用 mmap 读权重。

    上游 ``Visualized_BGE.load_model`` 直接 ``torch.load(weight)``：先把 1.75 GB
    权重读进匿名内存，再拷进 fp32 参数（约 3.5 GB）。加一层 mmap 让权重走文件
    映射，峰值少一份 1.75 GB——对 8 GB 内存的机器是"能加载 / OOM"的区别。

    用子类覆盖而不是改内嵌源码：内嵌快照与上游的差异保持最小
    （见 ``app/embed/visual_bge/README.md``）。
    """
    import torch

    from app.embed.visual_bge.modeling import Visualized_BGE

    class _VisualizedBGE(Visualized_BGE):
        def load_model(self, model_weight):  # noqa: D102 - 覆盖上游同名方法
            # Windows 上 torch.load(mmap=True) 曾在 1.75GB 权重上 access violation；
            # KUMORE_VISUAL_MMAP=0 退回普通读入（多占一份内存，但更稳）。
            use_mmap = os.environ.get("KUMORE_VISUAL_MMAP", "1").strip() not in {
                "0", "false", "no",
            }
            log.info("加载视觉权重 %s（mmap=%s）", model_weight, use_mmap)
            state = torch.load(
                model_weight,
                map_location="cpu",
                mmap=use_mmap,
                weights_only=True,
            )
            log.info("视觉权重读取完成，load_state_dict…")
            self.load_state_dict(state)
            log.info("视觉权重 load_state_dict 完成")

    return _VisualizedBGE(
        model_name_bge=VISUAL_TEXT_MODEL,
        model_weight=str(weight),
    )


class LocalEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-visualized-m3",
        device: str = "cpu",
        batch_size: int = 8,
        model_dir: Path | None = None,
        visual_weight: Path | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.model_dir = Path(model_dir) if model_dir else None
        #: 视觉权重路径。None 时取 ``model_dir/visual/Visualized_m3.pth``。
        self.visual_weight = Path(visual_weight) if visual_weight else None
        self._model = None
        self._dim: int | None = None
        self._load_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def prewarm(self) -> None:
        """后台预热；与第一次 encode 共享同一把加载锁。"""
        self._load()

    # ---- 懒加载：不装 torch 的机器上，bm25 模式仍能跑 ----
    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if is_visualized_model(self.model_name):
                self._load_visual()
            else:
                self._load_sentence_transformer()

    @property
    def visual_weight_path(self) -> Path:
        """视觉权重实际路径：显式配置优先，否则放模型缓存目录下。"""
        if self.visual_weight is not None:
            return self.visual_weight
        if self.model_dir is None:
            raise EmbedderUnavailable(
                "加载 bge-visualized 需要知道权重放在哪：请设置 KUMORE_MODEL_DIR "
                "或 KUMORE_EMBED_VISUAL_WEIGHT。"
            )
        return self.model_dir / "visual" / VISUAL_WEIGHT_FILENAME

    def _load_sentence_transformer(self) -> None:
        cache_dir = prepare_hf_env(self.model_dir)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise EmbedderUnavailable(
                "未安装 sentence-transformers / torch，无法使用本地 embedding。"
                "安装：pip install -r requirements-embed.txt"
            ) from exc

        log.info("加载本地 embedding 模型 %s（device=%s）", self.model_name, self.device)
        started = time.perf_counter()
        self._apply_thread_limit()
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(cache_dir) if cache_dir else None,
            )
        except Exception as exc:  # noqa: BLE001 - 网络/缓存/版本错误都归成一条可执行提示
            raise EmbedderUnavailable(
                f"本地 embedding 模型加载失败：{type(exc).__name__}: {exc}\n"
                "默认离线（HF_HUB_OFFLINE=1）。首次使用请先下载权重：\n"
                "  $env:KUMORE_ALLOW_MODEL_DOWNLOAD='1'; python -m app embed --rebuild\n"
                "或把模型放到 KUMORE_MODEL_DIR 指向的缓存目录。"
            ) from exc
        self._dim = int(self._model.get_sentence_embedding_dimension())
        log.info("模型就绪，维度=%d，加载耗时 %.1fs", self._dim, time.perf_counter() - started)

    def _load_visual(self) -> None:
        """加载 Visualized-BGE（bge-visualized-m3）。

        与文本模型的三处不同，都必须处理：
        - 权重不是一个 HF 仓库，而是 ``Visualized_m3.pth`` 单文件，路径要显式给；
        - 加载类来自内嵌的 ``app.embed.visual_bge``（上游未发布 PyPI 包，P86）；
        - 文本塔的 config/tokenizer 仍要按 ``BAAI/bge-m3`` 从项目缓存离线加载。
        """
        weight = self.visual_weight_path
        if not weight.is_file():
            raise EmbedderUnavailable(
                f"找不到视觉权重：{weight}\n"
                "bge-visualized-m3 = BAAI/bge-m3 文本塔 + Visualized_m3.pth 视觉权重。\n"
                "首次使用先下载（约 1.75 GB）：\n"
                f'  curl -L -o "{weight}" {VISUAL_WEIGHT_URL}\n'
                "文本塔权重仍需按常规方式下载（KUMORE_ALLOW_MODEL_DOWNLOAD=1）。"
            )
        if self.model_dir is None:
            raise EmbedderUnavailable(
                "加载 bge-visualized 需要 KUMORE_MODEL_DIR 指向模型缓存目录。"
            )

        cache_dir = prepare_hf_env(self.model_dir)
        try:
            from app.embed.visual_bge.modeling import Visualized_BGE  # noqa: F401
        except ImportError as exc:
            raise EmbedderUnavailable(
                "加载 bge-visualized 需要 timm / torchvision / einops / ftfy，"
                "安装：pip install -r requirements-embed.txt"
            ) from exc

        log.info("加载本地视觉 embedding 模型 %s（权重 %s）", self.model_name, weight)
        started = time.perf_counter()
        self._apply_thread_limit()
        try:
            model = _build_visualized_bge(weight)
            # 上游类在检测到 CUDA 时会自行搬到 GPU；这里以配置为准，搬回来。
            model.to(self.device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - 权重/缓存/显存错误都归成一条可执行提示
            raise EmbedderUnavailable(
                f"视觉 embedding 模型加载失败：{type(exc).__name__}: {exc}\n"
                f"权重：{weight}\n"
                "文本塔需要 BAAI/bge-m3 的 config/tokenizer 在 "
                f"{self.model_dir} 缓存中。"
            ) from exc
        self._model = model
        # 视觉模型的 hidden_dim 与 bge-m3 一致（1024），但不要写死：
        # 换成 base 版（768）时维度必须跟着变，否则落盘的索引维度对不上。
        self._dim = int(getattr(model, "hidden_dim", 1024))
        log.info(
            "视觉模型就绪，维度=%d，加载耗时 %.1fs",
            self._dim,
            time.perf_counter() - started,
        )

    @property
    def supports_images(self) -> bool:
        """当前后端能否编码图片。纯文本后端为 False。"""
        return is_visualized_model(self.model_name)

    @staticmethod
    def _apply_thread_limit() -> None:
        """限制 CPU 线程数。

        实测教训：默认不限制时 torch 在本机起了 **56 个线程**跑 8 核 CPU，
        严重超订导致 1,949 个 chunk 编码 45 分钟仍未结束。限制到物理核数量级
        （默认 min(8, 核数)）反而快得多。
        """
        cores = os.cpu_count() or 4
        threads = int(os.environ.get("KUMORE_EMBED_THREADS", str(min(8, cores))))
        threads = max(1, threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        try:
            import torch

            torch.set_num_threads(threads)
            log.info("embedding 线程数=%d（物理核 %d）", threads, cores)
        except Exception:  # pragma: no cover - 仅优化，失败不影响功能
            pass

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """返回 (n, dim) 的 float32 矩阵，已做 L2 归一化（点积即余弦）。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        if is_visualized_model(self.model_name):
            return self._encode_visual(
                texts,
                batch_size=batch_size or self.batch_size,
                show_progress=show_progress,
            )
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"embedding 输出维度异常：{matrix.shape}")
        return matrix

    def _encode_visual(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Visualized-BGE 的分批编码。

        上游 ``encode`` 一次吃一个 batch 并返回 tensor：没有 sentence-transformers
        的 batch_size/normalize 参数，也不做 dtype 转换。整库一次喂进去会直接把
        内存打满，所以这里自己切 batch、自己转 numpy，输出契约与文本后端一致。
        """
        import torch

        bars = range(0, len(texts), max(1, batch_size))
        if show_progress:
            try:
                from tqdm.auto import tqdm

                bars = tqdm(bars, desc="embed", unit="batch")
            except ImportError:  # pragma: no cover - tqdm 是常规依赖，缺了也不影响功能
                log.info("未安装 tqdm，省略进度条")
        pieces: list[np.ndarray] = []
        for start in bars:
            batch = texts[start : start + max(1, batch_size)]
            with torch.inference_mode():
                vectors = self._model.encode(text=batch)  # type: ignore[union-attr]
            pieces.append(vectors.detach().to("cpu", torch.float32).numpy())
        matrix = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        if matrix.ndim != 2:
            raise ValueError(f"embedding 输出维度异常：{matrix.shape}")
        if self._dim is not None and matrix.shape[1] != self._dim:
            raise ValueError(
                f"视觉模型输出维度 {matrix.shape[1]} 与声明的 {self._dim} 不一致"
            )
        return matrix

    def encode_images(self, paths: list[Path | str]) -> np.ndarray:
        """编码图片（仅视觉后端）。

        当前 ingest 还不产出图片块——扫描件/图片型文档仍按"解析失败"记录，
        真正接入要等渲染与图片入库。这里先把模型侧通路留好，调用方拿到的是
        与文本同一个向量空间的 (n, dim) 矩阵，可直接和文本向量比较。
        """
        if not self.supports_images:
            raise ValueError(
                f"模型 {self.model_name} 不支持图片；请用 BAAI/bge-visualized-m3"
            )
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        import torch

        pieces: list[np.ndarray] = []
        for path in paths:
            with torch.inference_mode():
                vector = self._model.encode(image=str(path))  # type: ignore[union-attr]
            pieces.append(vector.detach().to("cpu", torch.float32).numpy())
        matrix = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        if matrix.ndim != 2:
            raise ValueError(f"图片 embedding 输出维度异常：{matrix.shape}")
        return matrix

    def encode_query(self, query: str) -> np.ndarray:
        query = (query or "").strip()
        if not query:
            raise ValueError("查询为空，不应调用 embedding")
        return self.encode([query])[0]
