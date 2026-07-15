import re
from typing import List, Optional, Sequence, Tuple

from app.config import CHUNK_OVERLAP_RATIO, CHUNK_SIZE_WORDS, MIN_CHUNK_WORDS
from .types import ChunkMetadata

_HEADING_RE = re.compile(
    r"^(?:"
    r"[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s\-\./:]{2,80}"  # HEPSİ BÜYÜK başlık
    r"|\d+(?:\.\d+)*\s+.+"  # 1. / 1.2 Başlık
    r"|#{1,6}\s+.+"  # markdown
    r")$"
)


def _split_words(text: str) -> List[str]:
    return [w for w in text.replace("\n", " ").split() if w]


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if _HEADING_RE.match(s):
        return True
    # Kısa satır ve sonunda nokta yoksa başlık adayı
    return len(s.split()) <= 12 and not s.endswith((".", "!", "?", ":", ";"))


def _paragraphs_from_page(page_text: str) -> List[Tuple[str, Optional[str]]]:
    """Sayfayı (paragraf, heading) çiftlerine böler."""
    lines = page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paragraphs: List[Tuple[str, Optional[str]]] = []
    buf: List[str] = []
    current_heading: Optional[str] = None

    def flush():
        nonlocal buf
        text = " ".join(x.strip() for x in buf if x.strip()).strip()
        if text:
            paragraphs.append((text, current_heading))
        buf = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if _is_heading(line) and len(_split_words(line)) <= 12:
            flush()
            current_heading = line.lstrip("#").strip()
            continue
        buf.append(line)
    flush()
    return paragraphs


def _windows_from_words(
    words: List[str],
    chunk_size_words: int,
    overlap: int,
    min_chunk_words: int,
) -> List[List[str]]:
    if not words:
        return []
    windows: List[List[str]] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        window = words[start:end]
        if end == len(words) and len(window) < min_chunk_words and windows:
            # Son küçük parçayı önceki chunk ile birleştir (tercihen tek parça)
            prev = windows.pop()
            windows.append(prev + window)
            break
        if end == len(words) and len(window) < min_chunk_words and not windows:
            windows.append(window)
            break
        windows.append(window)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return windows


def chunk_pages(
    source_file: str,
    pages: List[str],
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    min_chunk_words: int = MIN_CHUNK_WORDS,
    folder: str = "",
    tags: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[ChunkMetadata]]:
    """Paragraf/başlık sınırlarını gözeten chunking.

    - Önce sayfa içi paragrafları çıkarır
    - Kısa kalıntıları mümkünse yan paragraflarla birleştirir
    - Sözcük penceresi + overlap uygular
    """
    texts: List[str] = []
    metas: List[ChunkMetadata] = []
    chunk_id = 0
    overlap = max(1, int(chunk_size_words * overlap_ratio))
    tag_list = list(tags or [])

    # (page_no 1-based, text, heading)
    units: List[Tuple[int, str, Optional[str]]] = []
    for page_idx, page_text in enumerate(pages):
        paras = _paragraphs_from_page(page_text or "")
        if not paras:
            words = _split_words(page_text or "")
            if words:
                units.append((page_idx + 1, " ".join(words), None))
            continue
        for para, heading in paras:
            units.append((page_idx + 1, para, heading))

    # Kısa paragrafları birleştirerek daha anlamlı bloklar üret
    merged_units: List[Tuple[int, int, str, Optional[str]]] = []  # p_start, p_end, text, heading
    buf_text = ""
    buf_heading: Optional[str] = None
    buf_start: Optional[int] = None
    buf_end: Optional[int] = None

    def flush_buf():
        nonlocal buf_text, buf_heading, buf_start, buf_end
        if buf_text and buf_start is not None and buf_end is not None:
            merged_units.append((buf_start, buf_end, buf_text.strip(), buf_heading))
        buf_text = ""
        buf_heading = None
        buf_start = None
        buf_end = None

    for page_no, para, heading in units:
        wc = len(_split_words(para))
        if not buf_text:
            buf_text = para
            buf_heading = heading
            buf_start = page_no
            buf_end = page_no
            if wc >= min_chunk_words:
                flush_buf()
            continue

        # Başlık değiştiyse önceki bloğu kapat
        if heading and heading != buf_heading and buf_text:
            flush_buf()
            buf_text = para
            buf_heading = heading
            buf_start = page_no
            buf_end = page_no
            if wc >= min_chunk_words:
                flush_buf()
            continue

        candidate = (buf_text + " " + para).strip()
        if len(_split_words(candidate)) <= chunk_size_words:
            buf_text = candidate
            buf_end = page_no
            if buf_heading is None:
                buf_heading = heading
            if len(_split_words(buf_text)) >= min_chunk_words and page_no != buf_start:
                # Sayfa değişiminde yeterince uzunsa flush tercihi yapma; pencereleme zaten böler
                pass
        else:
            flush_buf()
            buf_text = para
            buf_heading = heading
            buf_start = page_no
            buf_end = page_no
    flush_buf()

    for page_start, page_end, block, heading in merged_units:
        words = _split_words(block)
        for window in _windows_from_words(words, chunk_size_words, overlap, min_chunk_words):
            if not window:
                continue
            # Tek başına çok kısa son parçaları atla (boş sayfa artığı)
            if len(window) < max(40, min_chunk_words // 5) and texts:
                # Önceki chunk'a eklemeyi dene
                prev_words = texts[-1].split()
                if len(prev_words) + len(window) <= int(chunk_size_words * 1.35):
                    texts[-1] = " ".join(prev_words + window)
                    metas[-1].word_count = len(prev_words) + len(window)
                    metas[-1].page_end = page_end
                    continue
            chunk_text = " ".join(window)
            texts.append(chunk_text)
            metas.append(
                ChunkMetadata(
                    source_file=source_file,
                    chunk_id=chunk_id,
                    page_start=page_start,
                    page_end=page_end,
                    word_count=len(window),
                    heading=heading,
                    folder=folder or "",
                    tags=list(tag_list),
                )
            )
            chunk_id += 1

    return texts, metas
