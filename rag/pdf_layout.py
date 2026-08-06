"""Layout-aware PDF çıkarımı: okuma sırası bloklar + tablolar (markdown)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from app.config import LAYOUT_TABLE_MIN_ROWS


def table_to_markdown(rows: Sequence[Sequence[Optional[str]]]) -> str:
    """Basit tabloyu markdown pipe tablosuna çevirir."""
    cleaned: List[List[str]] = []
    for row in rows or []:
        cells = [("" if c is None else str(c).replace("\n", " ").strip()) for c in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < LAYOUT_TABLE_MIN_ROWS:
        return ""

    width = max(len(r) for r in cleaned)
    normalized = [r + [""] * (width - len(r)) for r in cleaned]
    header = normalized[0]
    body = normalized[1:] or [[""] * width]

    def _esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = [
        "| " + " | ".join(_esc(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_esc(c) for c in row) + " |")
    return "[Tablo]\n" + "\n".join(lines)


def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _block_mostly_inside_table(
    block_bbox: Tuple[float, float, float, float],
    table_bboxes: Sequence[Tuple[float, float, float, float]],
) -> bool:
    """Metin bloğu tablo alanında ise (çift sayımı önlemek için) atlanır."""
    bx0, by0, bx1, by1 = block_bbox
    bw = max(bx1 - bx0, 1.0)
    bh = max(by1 - by0, 1.0)
    for tb in table_bboxes:
        tx0, ty0, tx1, ty1 = tb
        ix0, iy0 = max(bx0, tx0), max(by0, ty0)
        ix1, iy1 = min(bx1, tx1), min(by1, ty1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        if inter / (bw * bh) >= 0.55:
            return True
    return False


def extract_tables_markdown_pdfplumber(page) -> List[Tuple[Tuple[float, float, float, float], str]]:
    """pdfplumber sayfasından (bbox, markdown) tablo listesi."""
    out: List[Tuple[Tuple[float, float, float, float], str]] = []
    try:
        tables = page.find_tables() or []
    except Exception:
        tables = []
    for table in tables:
        try:
            bbox = tuple(float(x) for x in table.bbox)  # type: ignore[attr-defined]
            raw = table.extract()
            md = table_to_markdown(raw or [])
            if md:
                out.append((bbox, md))  # type: ignore[arg-type]
        except Exception:
            continue
    if out:
        return out

    # find_tables yoksa / boşsa extract_tables yedeği
    try:
        raw_tables = page.extract_tables() or []
    except Exception:
        raw_tables = []
    for i, raw in enumerate(raw_tables):
        md = table_to_markdown(raw or [])
        if not md:
            continue
        # bbox bilinmiyor; sayfa sonuna yakın sahte y
        h = float(getattr(page, "height", 1000) or 1000)
        bbox = (0.0, h + i, float(getattr(page, "width", 100) or 100), h + i + 1)
        out.append((bbox, md))
    return out


def extract_text_blocks_pymupdf(page) -> List[Tuple[Tuple[float, float, float, float], str]]:
    """PyMuPDF okuma sırası blokları: (bbox, text)."""
    blocks_out: List[Tuple[Tuple[float, float, float, float], str]] = []
    try:
        raw_blocks = page.get_text("blocks") or []
    except Exception:
        return blocks_out
    for b in raw_blocks:
        # x0, y0, x1, y1, text, block_no, block_type
        if len(b) < 5:
            continue
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        block_type = b[6] if len(b) > 6 else 0
        if block_type != 0:
            continue
        content = (text or "").strip()
        if not content:
            continue
        blocks_out.append(((float(x0), float(y0), float(x1), float(y1)), content))
    return blocks_out


def compose_page_layout(
    text_blocks: Sequence[Tuple[Tuple[float, float, float, float], str]],
    tables: Sequence[Tuple[Tuple[float, float, float, float], str]],
) -> str:
    """Blok + tabloları yukarıdan aşağı / soldan sağa birleştirir."""
    table_bboxes = [tb for tb, _ in tables]
    items: List[Tuple[float, float, str, str]] = []

    for bbox, text in text_blocks:
        if _block_mostly_inside_table(bbox, table_bboxes):
            continue
        y0, x0 = bbox[1], bbox[0]
        items.append((y0, x0, "text", text))

    for bbox, md in tables:
        items.append((bbox[1], bbox[0], "table", md))

    items.sort(key=lambda t: (round(t[0], 1), round(t[1], 1)))
    parts = [content for *_coords, _kind, content in items if content.strip()]
    return "\n\n".join(parts).strip()


def extract_page_text_layout(fitz_page, plumber_page=None) -> str:
    """Tek sayfa için layout-aware metin + tablo markdown."""
    text_blocks = extract_text_blocks_pymupdf(fitz_page)
    tables: List[Tuple[Tuple[float, float, float, float], str]] = []
    if plumber_page is not None:
        tables = extract_tables_markdown_pdfplumber(plumber_page)
    composed = compose_page_layout(text_blocks, tables)
    if composed:
        return composed
    # yedek: düz metin
    try:
        return (fitz_page.get_text("text") or "").strip()
    except Exception:
        return ""
