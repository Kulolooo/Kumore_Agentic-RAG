"""Visualized-BGE 推理代码（内嵌快照）。

上游：FlagOpen/FlagEmbedding `research/visual_bge`，MIT 许可，见同目录 ``LICENSE``。
改动与原因见 ``README.md``。为什么内嵌而不是 pip 安装：

- 上游没有发布到 PyPI；
- 它的 ``setup.py`` 用 ``find_packages()``，而包目录没有 ``__init__.py``，
  装出来是个空 wheel（实测 1,164 字节，``import visual_bge`` 直接失败）；
- 本项目索引侧默认离线，运行时不联网、也不依赖构建期的 git 克隆。

这里只放推理需要的代码；训练脚本（``train.py`` 等）不在内嵌范围内。
"""
