# 内嵌的 Visualized-BGE（bge-visualized-m3 的加载代码）

## 来源

- 仓库：<https://github.com/FlagOpen/FlagEmbedding>
- 目录：`research/visual_bge/visual_bge/`
- 快照：`git clone --depth 1 --filter=blob:none --sparse`，HEAD
  `897baa67c0ffddbfccf4e6776ff55310d67809d4`（2026-10-30 取）
- 许可：MIT，Copyright (c) 2022 staoxiao，原文见同目录 `LICENSE`

## 为什么内嵌

1. 上游未发布到 PyPI（`pip install visual_bge` 不存在）；
2. 上游 `setup.py` 的 `find_packages()` 装出来是空 wheel——包目录缺 `__init__.py`，
   实测产物 1,164 字节，`import visual_bge` 报 ModuleNotFoundError；
3. 本项目的索引侧默认离线（`HF_HUB_OFFLINE=1`），运行期不应依赖 git 克隆或联网安装。

## 相对上游的改动（只有 3 处，均不涉及算法）

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `modeling.py:12` | `from visual_bge.eva_clip import …` → `from .eva_clip import …` | 改成包内相对导入，避免与顶层 `visual_bge` 同名冲突 |
| `eva_clip/modified_resnet.py:7` | `from visual_bge.eva_clip.utils import …` → `from .utils import …` | 同上 |
| 新增 `__init__.py` | 空包声明 + 来源说明 | 让它成为真正的 Python 包 |

算法、权重键名、预处理参数一律未改；`Visualized_m3.pth` 原样加载。
