"""集中配置。

约定：路径、阈值、模型参数只在这里出现一次，其他模块一律 ``from app.config import settings``。
敏感信息（语料路径、API key）只从环境变量或本地 ``.env`` 读取，不写死在代码里，
这样仓库可以直接公开，也不会泄露本机目录结构。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """最小 .env 读取器。

    只处理 ``KEY=VALUE``、``#`` 注释和可选的成对引号；已存在的环境变量优先，
    因此命令行里临时设置的变量不会被文件覆盖。不引入 python-dotenv，少一个依赖。
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # 配置写错时立刻报错，而不是悄悄用默认值
        raise ValueError(f"环境变量 {name} 必须是整数，当前为 {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字，当前为 {raw!r}") from exc


def _env_path(name: str, default: Path | None) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else default


#: 可入库格式；图片走 OCR（RapidOCR），扫描 PDF 按页 OCR。
#: 代码/编译产物仍跳过；否则语料目录里几十万张训练图片会把入库过程拖死。
DEFAULT_EXT_WHITELIST = (
    ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".md", ".txt",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
)


@dataclass(frozen=True)
class Settings:
    """运行时配置快照。"""

    corpus_root: Path
    storage_dir: Path
    data_dir: Path
    ext_whitelist: frozenset[str]
    chunk_size: int
    chunk_overlap: int
    #: 分块策略：fixed（M0 基线）/ token（RAGFlow general）/ section / parent_child。
    chunk_strategy: str
    #: token 策略的单块预算（估算 token，见 app/text/tokens.py）。
    chunk_token_size: int
    #: section 策略的目标块大小。
    section_tokens: int
    #: 父子块策略的子块 / 父块预算。
    child_tokens: int
    parent_tokens: int
    top_k: int
    #: 检索分数的相对截断：丢掉低于 (最高分 × 该值) 的命中。0 表示不裁。
    retrieve_cutoff: float
    #: 同一"版本族"里最多取几个 chunk（0 = 不限）。防止同一份材料的副本占满 top-k。
    max_per_family: int
    bm25_k1: float
    bm25_b: float
    #: 检索模式：bm25（M0 基线）/ vector / hybrid
    retrieval_mode: str
    #: 融合方式：rrf（免调参）或 weighted（RAGFlow 默认的加权和）
    fusion: str
    #: weighted 融合时向量分量的权重（RAGFlow 默认 0.3）
    vector_weight: float
    #: RRF 的 k（RAGFlow 结构导航里用 60）
    rrf_k: int
    #: 检索候选池大小：先取这么多条，再过滤/折叠/限流。RAGFlow 大库用 1024，
    #: 08 报告建议我们取 100（`:250`）。太小会在折叠重复段之后不够 top_k。
    knn_top_k: int
    #: 是否启用重排（M2）。默认关闭：先证明"重排能救回融合救不回的题"再谈默认值。
    rerank: bool
    #: 重排模型。bge-reranker-base（278M / 1.1 GB）而不是 v2-m3（568M / 2.3 GB）：
    #: 前者中英双语、CPU 可跑，后者对当前 45 题的增益没有证据。
    #: **注意它的上限**：xlm-roberta 的 `max_position_embeddings=514`。
    #: 本语料 chunk 中位 488 token、p90 544，实测 67% 的 (query, chunk) 对超过 512，
    #: 尾部会被截断——所以 M2 的负结论只对"512 截断下的重排"成立，
    #: 换长上下文模型（v2-m3 支持 8k）需要重测。
    rerank_model: str
    rerank_device: str
    rerank_batch_size: int
    #: 只重排融合后的前 N 条。RAGFlow 用 20，再往上收益递减、延迟线性涨。
    rerank_candidates: int
    #: cross-encoder 的截断长度。**不要以为 512 覆盖了 735 字的 chunk**：
    #: 实测中位 488 token、p90 544、max 648，67% 的对会被截断。
    #: bge-reranker-base 的硬上限是 514，调大没有意义；要覆盖全文得换模型。
    rerank_max_length: int
    #: 重排失败时是否退回"保持检索原序"。RAGFlow 缺这个，是它最明显的可靠性缺口
    #: （`03-检索与排序.md` §5、`retrieval.go:199-202`）。我们默认 fail-open。
    rerank_fail_open: bool
    #: 文本塔。默认 bge-m3；只编码文本 chunk。
    embedding_model: str
    #: cpu / cuda。M0-1 默认 cpu——8 GB 显存留给将来的 reranker。
    embedding_device: str
    embedding_batch_size: int
    #: bge-visualized 的单文件权重（Visualized_m3.pth）。None 表示按
    #: ``model_dir/visual/Visualized_m3.pth`` 找；纯文本模型不用这一项。
    embedding_visual_weight: Path | None
    #: 模型权重缓存目录（默认在 storage/models 下，已被 gitignore）
    model_dir: Path
    #: 解析策略。M0 固定 fast（无模型依赖）；hi_res 留给 M1。
    parse_strategy: str
    #: 是否把文档路径/标题一起发给模型。默认关闭：路径里常有姓名与学号，
    #: 而模型只需要编号来引用，不需要知道文件在哪。
    send_source_paths: bool
    #: 是否缓存模型响应（按请求内容做键）。开启后重复提问逐字一致，也省钱。
    llm_cache: bool
    #: 提示词变体：strict（默认，资料不足即拒答）/ tight（只在本就找不到相关句时拒答）。
    prompt_variant: str
    #: 解析后端：native（python-docx/pdfminer/openpyxl，纯离线）或 unstructured。
    parser_backend: str
    #: 超过该大小的文件跳过解析（0 表示不限制）。
    #: 实测 131 MB 的 pptx 解析只要 981 ms——文件大几乎都是内嵌图片/视频，
    #: 文字 XML 只有几 MB。所以这个上限只用来防"指到一个几 GB 的怪物"，
    #: 不该拿来判断"值不值得解析"。
    max_file_bytes: int
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    request_timeout_s: float
    max_files: int | None
    #: 视觉塔。只编码图片/扫描页，和文本索引完全分开。
    visual_embedding_model: str = "BAAI/bge-visualized-m3"

    @classmethod
    def from_env(cls) -> "Settings":
        whitelist_raw = os.environ.get("KUMORE_EXT_WHITELIST", "").strip()
        if whitelist_raw:
            exts = frozenset(
                (ext if ext.startswith(".") else f".{ext}").lower()
                for ext in (part.strip() for part in whitelist_raw.split(","))
                if ext
            )
        else:
            exts = frozenset(DEFAULT_EXT_WHITELIST)

        max_files_raw = os.environ.get("KUMORE_MAX_FILES", "").strip()
        max_files = _env_int("KUMORE_MAX_FILES", 0) or None if max_files_raw else None

        settings = cls(
            corpus_root=_env_path("KUMORE_CORPUS_ROOT", PROJECT_ROOT / "data" / "corpus"),
            storage_dir=_env_path("KUMORE_STORAGE_DIR", PROJECT_ROOT / "storage"),
            data_dir=_env_path("KUMORE_DATA_DIR", PROJECT_ROOT / "data"),
            ext_whitelist=exts,
            chunk_size=_env_int("KUMORE_CHUNK_SIZE", 800),
            chunk_overlap=_env_int("KUMORE_CHUNK_OVERLAP", 150),
            chunk_strategy=os.environ.get("KUMORE_CHUNK_STRATEGY", "fixed").strip() or "fixed",
            chunk_token_size=_env_int("KUMORE_CHUNK_TOKEN_SIZE", 512),
            section_tokens=_env_int("KUMORE_SECTION_TOKENS", 1800),
            child_tokens=_env_int("KUMORE_CHILD_TOKENS", 512),
            parent_tokens=_env_int("KUMORE_PARENT_TOKENS", 1800),
            top_k=_env_int("KUMORE_TOP_K", 5),
            retrieve_cutoff=_env_float("KUMORE_RETRIEVE_CUTOFF", 0.0),
            max_per_family=_env_int("KUMORE_MAX_PER_FAMILY", 3),
            bm25_k1=_env_float("KUMORE_BM25_K1", 1.5),
            bm25_b=_env_float("KUMORE_BM25_B", 0.75),
            retrieval_mode=os.environ.get("KUMORE_RETRIEVAL_MODE", "bm25").strip() or "bm25",
            fusion=os.environ.get("KUMORE_FUSION", "rrf").strip() or "rrf",
            vector_weight=_env_float("KUMORE_VECTOR_WEIGHT", 0.3),
            rrf_k=_env_int("KUMORE_RRF_K", 60),
            knn_top_k=_env_int("KUMORE_KNN_TOP_K", 100),
            rerank=os.environ.get("KUMORE_RERANK", "0").strip() in {"1", "true", "yes"},
            rerank_model=os.environ.get("KUMORE_RERANK_MODEL", "BAAI/bge-reranker-base").strip()
            or "BAAI/bge-reranker-base",
            rerank_device=os.environ.get("KUMORE_RERANK_DEVICE", "cpu").strip() or "cpu",
            rerank_batch_size=_env_int("KUMORE_RERANK_BATCH", 8),
            rerank_candidates=_env_int("KUMORE_RERANK_CANDIDATES", 20),
            rerank_max_length=_env_int("KUMORE_RERANK_MAX_LENGTH", 512),
            rerank_fail_open=os.environ.get("KUMORE_RERANK_FAIL_OPEN", "1").strip()
            in {"1", "true", "yes"},
            embedding_model=os.environ.get("KUMORE_EMBED_MODEL", "BAAI/bge-m3").strip()
            or "BAAI/bge-m3",
            visual_embedding_model=os.environ.get(
                "KUMORE_VISUAL_EMBED_MODEL", "BAAI/bge-visualized-m3"
            ).strip()
            or "BAAI/bge-visualized-m3",
            embedding_device=os.environ.get("KUMORE_EMBED_DEVICE", "cpu").strip() or "cpu",
            embedding_batch_size=_env_int("KUMORE_EMBED_BATCH", 8),
            embedding_visual_weight=_env_path("KUMORE_EMBED_VISUAL_WEIGHT", None),
            model_dir=_env_path("KUMORE_MODEL_DIR", PROJECT_ROOT / "storage" / "models"),
            parse_strategy=os.environ.get("KUMORE_PARSE_STRATEGY", "fast").strip() or "fast",
            send_source_paths=os.environ.get("KUMORE_SEND_PATHS", "0").strip() in {"1", "true", "yes"},
            llm_cache=os.environ.get("KUMORE_LLM_CACHE", "0").strip() in {"1", "true", "yes"},
            prompt_variant=os.environ.get("KUMORE_PROMPT_VARIANT", "strict").strip() or "strict",
            parser_backend=os.environ.get("KUMORE_PARSER", "native").strip() or "native",
            max_file_bytes=_env_int("KUMORE_MAX_FILE_MB", 300) * 1024 * 1024,
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip() or None,
            deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip(),
            request_timeout_s=_env_float("KUMORE_REQUEST_TIMEOUT", 120.0),
            max_files=max_files,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        from app.ingest.chunkers import ChunkStrategy

        if self.chunk_strategy not in ("fixed", "token", "section", "parent_child"):
            raise ValueError(
                f"chunk_strategy 只能是 fixed/token/section/parent_child，当前 {self.chunk_strategy!r}"
            )
        # 用构造一次的方式校验：四种预算、父子块大小关系都在 chunkers 里定义，
        # 避免 config 与 chunkers 各维护一份合法值列表。
        ChunkStrategy(
            name=self.chunk_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            token_size=self.chunk_token_size,
            section_tokens=self.section_tokens,
            child_tokens=self.child_tokens,
            parent_tokens=self.parent_tokens,
        )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正数，当前 {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"chunk_overlap 必须满足 0 <= overlap < chunk_size，"
                f"当前 overlap={self.chunk_overlap}, chunk_size={self.chunk_size}"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k 必须为正数，当前 {self.top_k}")
        if self.retrieval_mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError(
                f"retrieval_mode 只能是 bm25 / vector / hybrid，当前 {self.retrieval_mode!r}"
            )
        if self.fusion not in {"rrf", "weighted"}:
            raise ValueError(f"fusion 只能是 rrf / weighted，当前 {self.fusion!r}")
        if not 0.0 <= self.vector_weight <= 1.0:
            raise ValueError(f"vector_weight 必须在 [0,1]，当前 {self.vector_weight}")
        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k 必须为正数，当前 {self.rrf_k}")
        if self.knn_top_k <= 0:
            raise ValueError(f"knn_top_k 必须为正数，当前 {self.knn_top_k}")
        if self.rerank_candidates <= 0:
            raise ValueError(f"rerank_candidates 必须为正数，当前 {self.rerank_candidates}")
        if self.rerank_batch_size <= 0:
            raise ValueError(f"rerank_batch_size 必须为正数，当前 {self.rerank_batch_size}")
        if self.rerank_max_length <= 0:
            raise ValueError(f"rerank_max_length 必须为正数，当前 {self.rerank_max_length}")
        if self.prompt_variant not in ("strict", "tight", "claim"):
            raise ValueError(
                f"prompt_variant 只能是 strict / tight / claim，当前 {self.prompt_variant!r}"
            )

    # ---- 派生路径 ----
    @property
    def docs_path(self) -> Path:
        return self.storage_dir / "docs.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.jsonl"

    @property
    def sqlite_path(self) -> Path:
        return self.storage_dir / "index.sqlite"

    @property
    def runs_dir(self) -> Path:
        return self.storage_dir / "runs"

    @property
    def gold_path(self) -> Path:
        return self.data_dir / "gold" / "m0_gold.jsonl"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    def resolve_gold(self, requested: str | Path | None = None) -> Path:
        """默认金标集：``data/gold/`` 下题量最多的一份。

        为什么不是写死 ``m0_gold.jsonl``：10 题与 45 题两份文件长期并存，
        M1 起所有对照实验都跑 45 题。裸 ``python -m app eval`` 若默认读 10 题，
        两份报告并排看就会被误读成"退化"（M2-0 的 C-1）。
        规则取"题量最多"而不是文件名：换语料、换题库都不用改代码。
        显式传 ``requested`` 时以调用方为准。
        """
        if requested:
            return Path(requested)
        candidates = sorted(self.gold_dir.glob("*.jsonl"))
        if not candidates:
            return self.gold_path

        def count(path: Path) -> int:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return sum(1 for line in fh if line.strip())
            except OSError:
                return -1

        return max(candidates, key=lambda p: (count(p), p.name))

    @property
    def vector_index_path(self) -> Path:
        """文本向量索引；模型名分文件，换模型不会读到旧向量。"""
        slug = self.embedding_model.replace("/", "__").replace(":", "_")
        return self.storage_dir / "vectors" / f"{slug}.npz"

    @property
    def visual_vector_index_path(self) -> Path:
        """视觉向量索引；单独文件，绝不与文本索引合并。"""
        slug = self.visual_embedding_model.replace("/", "__").replace(":", "_")
        return self.storage_dir / "vectors" / f"{slug}-visual.npz"

    @property
    def images_path(self) -> Path:
        return self.storage_dir / "images.jsonl"

    @property
    def assets_dir(self) -> Path:
        # generation 的父目录是 kb 目录；资产不随代际复制/删除。
        return self.storage_dir.parent / "assets"

    @property
    def visual_status_path(self) -> Path:
        return self.storage_dir / "visual_status.json"

    @property
    def visual_queue_path(self) -> Path:
        return self.storage_dir / "visual_queue.json"


settings = Settings.from_env()
