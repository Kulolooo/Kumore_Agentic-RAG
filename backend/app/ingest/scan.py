"""语料扫描。

只做一件事：按扩展名白名单把目录里**可解析的文本文件**找出来，并给出可复现的顺序。

为什么白名单是硬要求：一份课程资料目录实测约 49 万个文件、15.2 GB，其中
48 万张是 ``计算机视觉/`` 下的训练图片。不先按扩展名过滤，入库会直接卡死在 IO 上。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: 这些目录名一律不进：版本库、依赖目录、缓存、回收站。
IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        ".idea",
        ".vscode",
        "__pycache__",
        ".pytest_cache",
        ".ipynb_checkpoints",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "$recycle.bin",
        "system volume information",
    }
)


@dataclass
class ScanResult:
    """扫描结果。统计数字用于报告"为什么只入了这些文件"。"""

    files: list[Path] = field(default_factory=list)
    seen_files: int = 0
    skipped_by_ext: int = 0
    skipped_dirs: int = 0
    skipped_temp: int = 0
    unreadable_dirs: list[str] = field(default_factory=list)
    truncated: bool = False

    def by_ext(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in self.files:
            ext = path.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1
        return dict(sorted(counts.items()))


def scan_corpus(
    root: Path,
    ext_whitelist: frozenset[str],
    max_files: int | None = None,
    ignore_dirs: frozenset[str] = IGNORED_DIR_NAMES,
) -> ScanResult:
    """递归扫描 ``root``，返回按相对路径排序的候选文件。

    顺序固定（排序后）是为了让"同一批输入 → 同一批 chunk_id"可复现。
    """
    root = Path(root)
    result = ScanResult()
    if not root.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{root}")

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=None):
        # 原地修改 dirnames 才能真的剪枝，避免走进 .git / node_modules / 图片数据集
        kept = []
        for name in sorted(dirnames):
            if name.lower() in ignore_dirs:
                result.skipped_dirs += 1
                continue
            kept.append(name)
        dirnames[:] = kept

        for name in sorted(filenames):
            result.seen_files += 1
            ext = os.path.splitext(name)[1].lower()
            if ext not in ext_whitelist:
                result.skipped_by_ext += 1
                continue
            # Office 打开文档时会留下 ~$xxx.docx 锁文件，读它必然破坏；
            # 以点开头的隐藏文件同理，都没有入库价值。
            if name.startswith("~$") or name.startswith("."):
                result.skipped_temp += 1
                continue
            result.files.append(Path(dirpath) / name)
            if max_files is not None and len(result.files) >= max_files:
                result.truncated = True
                result.files.sort(key=lambda p: str(p).lower())
                return result

    result.files.sort(key=lambda p: str(p).lower())
    return result


def rel_path_of(path: Path, root: Path) -> str:
    """相对路径，统一用正斜杠。跨盘符时退回绝对路径。"""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
