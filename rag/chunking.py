from typing import List, Tuple
from .types import ChunkMetadata


def _split_words(text: str) -> List[str]:
    return [w for w in text.replace("\n", " ").split() if w]


def chunk_pages(
    source_file: str,
    pages: List[str],
    chunk_size_words: int = 700,
    overlap_ratio: float = 0.12,
) -> Tuple[List[str], List[ChunkMetadata]]:
    """Her sayfayı 500–800 kelimelik chunk’lara böler; %10–15 overlap.
    Basitlik için sayfa sınırlarını korur.
    """
    texts: List[str] = []
    metas: List[ChunkMetadata] = []
    chunk_id = 0
    overlap = max(1, int(chunk_size_words * overlap_ratio))

    for page_idx, page_text in enumerate(pages):
        words = _split_words(page_text)
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size_words)
            window = words[start:end]
            # Kısa chunk’lardan kaçın: <200 kelime ise birleştirmeyi dene
            if len(window) < 200 and end == len(words):
                # Son küçük parça ise atla
                break
            chunk_text = " ".join(window)
            texts.append(chunk_text)
            metas.append(
                ChunkMetadata(
                    source_file=source_file,
                    chunk_id=chunk_id,
                    page_start=page_idx + 1,
                    page_end=page_idx + 1,
                    word_count=len(window),
                    heading=None,
                )
            )
            chunk_id += 1
            if end == len(words):
                break
            start = end - overlap
    return texts, metas
