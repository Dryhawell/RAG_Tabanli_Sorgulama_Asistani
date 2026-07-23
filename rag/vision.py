"""Görüntü OCR ve tablo-odaklı sorgu yardımcıları."""

from __future__ import annotations

import io
import os
import re
from typing import List, Optional, Sequence, Tuple

from app.config import OCR_LANGS
from rag.types import RetrievedChunk

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

_TABLE_Q_RE = re.compile(
    r"(tablo|table|satır|sutun|sütun|kolon|column|\brow\b|\bcell\b|hücre|markdown\s*tablo)",
    re.IGNORECASE,
)


def is_image_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_table_question(question: str) -> bool:
    return bool(_TABLE_Q_RE.search(question or ""))


def ocr_image_bytes(
    data: bytes,
    *,
    langs: str = OCR_LANGS,
    filename: str = "image.png",
) -> str:
    """PIL + pytesseract ile görüntü metni. Tesseract yoksa boş döner."""
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow gerekli: pip install pillow") from exc

    try:
        import pytesseract
    except Exception as exc:
        raise RuntimeError("pytesseract gerekli: pip install pytesseract") from exc

    from shutil import which

    if which("tesseract") is None:
        return ""

    img = Image.open(io.BytesIO(data))
    if img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")
    text = pytesseract.image_to_string(img, lang=langs.replace("+", "+"))
    return (text or "").strip()


def ocr_image_file(path: str, *, langs: str = OCR_LANGS) -> str:
    with open(path, "rb") as f:
        return ocr_image_bytes(f.read(), langs=langs, filename=os.path.basename(path))


def image_query_context(data: bytes, filename: str = "image.png") -> str:
    """Soru ile birlikte kullanılacak OCR bağlamı."""
    try:
        text = ocr_image_bytes(data, filename=filename)
    except Exception as exc:
        return f"[Görüntü OCR başarısız: {exc}]"
    if not text:
        return f"[Görüntü: {filename} — OCR metni yok veya Tesseract kurulu değil]"
    return f"[Görüntü OCR: {filename}]\n{text}"


def chunk_has_table(chunk: RetrievedChunk) -> bool:
    text = chunk.text or ""
    return "[Tablo]" in text or "|" in text and "---" in text


def prioritize_table_chunks(
    chunks: Sequence[RetrievedChunk],
    *,
    question: str,
) -> List[RetrievedChunk]:
    """Tablo sorularında tablo içeren chunk'ları öne alır."""
    items = list(chunks)
    if not is_table_question(question) or not items:
        return items
    tables = [c for c in items if chunk_has_table(c)]
    others = [c for c in items if not chunk_has_table(c)]
    if not tables:
        return items
    return tables + others


def merge_image_into_question(question: str, image_context: str) -> str:
    q = (question or "").strip()
    ctx = (image_context or "").strip()
    if not ctx:
        return q
    if not q:
        return f"Bu görüntüdeki metin/tablo hakkında bilgi ver.\n\n{ctx}"
    return f"{q}\n\n{ctx}"
