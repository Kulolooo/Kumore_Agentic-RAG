"""图片/页面图资产：与文本 chunk 分开存，视觉索引只引用 image_id。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.schemas import ImageRecord


def image_id_for(doc_id: str, page: int) -> str:
    return f"{doc_id}:p{page:04d}"


def save_image_asset(
    settings: Settings,
    *,
    doc_id: str,
    rel_path: str,
    page: int,
    image: Image.Image,
) -> ImageRecord:
    """把 PIL 图保存到 KB 级 assets/（不随代际删除），返回记录。"""
    assets = settings.assets_dir
    assets.mkdir(parents=True, exist_ok=True)
    filename = f"{doc_id}-p{page:04d}.png"
    path = assets / filename
    tmp = path.with_name(path.name + ".tmp")
    image.convert("RGB").save(tmp, format="PNG")
    data = tmp.read_bytes()
    tmp.replace(path)
    return ImageRecord(
        image_id=image_id_for(doc_id, page),
        doc_id=doc_id,
        rel_path=rel_path,
        page=page,
        image_file=filename,
        sha256=hashlib.sha256(data).hexdigest(),
        width=int(image.width),
        height=int(image.height),
    )


def save_image_file(
    settings: Settings,
    *,
    doc_id: str,
    rel_path: str,
    source: Path,
) -> ImageRecord:
    with Image.open(source) as image:
        return save_image_asset(
            settings, doc_id=doc_id, rel_path=rel_path, page=1, image=image
        )
