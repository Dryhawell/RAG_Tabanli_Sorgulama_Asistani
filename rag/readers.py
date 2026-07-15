from typing import List, Optional, Tuple
import os

from app.config import (
    ENABLE_LAYOUT_PDF,
    ENABLE_OCR,
    OCR_LANGS,
    OCR_MIN_CHARS,
)

# PDF okuma: layout-aware (blok + tablo) → PyMuPDF düz metin → pdfplumber yedek


def _ocr_available() -> bool:
    """Tesseract binary'sinin bulunup bulunmadığını kontrol eder."""
    from shutil import which

    return which("tesseract") is not None


def _ocr_page_pymupdf(page, langs: str = OCR_LANGS) -> str:
    """PyMuPDF TextPage OCR ile sayfa metni çıkarır."""
    try:
        tp = page.get_textpage_ocr(language=langs, dpi=200)
        return page.get_text("text", textpage=tp) or ""
    except Exception:
        return ""


def _read_pdf_layout(path: str, enable_ocr: bool = ENABLE_OCR) -> List[str]:
    """PyMuPDF blok sırası + pdfplumber tabloları ile sayfa metinleri."""
    import fitz
    import pdfplumber

    from rag.pdf_layout import extract_page_text_layout

    texts: List[str] = []
    use_ocr = enable_ocr and _ocr_available()

    with fitz.open(path) as doc, pdfplumber.open(path) as pdf:
        plumber_pages = list(pdf.pages)
        for i, page in enumerate(doc):
            plumber_page = plumber_pages[i] if i < len(plumber_pages) else None
            text = extract_page_text_layout(page, plumber_page)
            if use_ocr and len(text.strip()) < OCR_MIN_CHARS:
                ocr_text = _ocr_page_pymupdf(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            texts.append(text or "")
    return texts


def _read_pdf_pymupdf(path: str, enable_ocr: bool = ENABLE_OCR) -> List[str]:
    import fitz  # pymupdf

    texts = []
    use_ocr = enable_ocr and _ocr_available()
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            if use_ocr and len(text.strip()) < OCR_MIN_CHARS:
                ocr_text = _ocr_page_pymupdf(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            texts.append(text)
    return texts


def _read_pdf_pdfplumber(path: str) -> List[str]:
    import pdfplumber

    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
    return texts


def read_pdf(
    path: str,
    enable_ocr: bool = ENABLE_OCR,
    enable_layout: Optional[bool] = None,
) -> List[str]:
    """PDF'i sayfa sayfa metin olarak döndürür.

    Layout modunda tablolar markdown'a çevrilir ve bloklar okuma sırasına dizilir.
    Tarama PDF'lerde Tesseract kuruluysa OCR dener.
    """
    use_layout = ENABLE_LAYOUT_PDF if enable_layout is None else enable_layout
    if use_layout:
        try:
            return _read_pdf_layout(path, enable_ocr=enable_ocr)
        except Exception:
            pass
    try:
        return _read_pdf_pymupdf(path, enable_ocr=enable_ocr)
    except Exception:
        return _read_pdf_pdfplumber(path)


def read_txt(path: str) -> List[str]:
    """TXT'i tek bir sayfa olarak döndürür."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [text]


def read_document(
    path: str,
    enable_ocr: bool = ENABLE_OCR,
    enable_layout: Optional[bool] = None,
) -> Tuple[str, List[str]]:
    """Dosya tipini algılar ve sayfa metinlerini döndürür.
    Returns: (source_file_name, [page_texts])
    """
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext == ".pdf":
        pages = read_pdf(path, enable_ocr=enable_ocr, enable_layout=enable_layout)
    elif ext == ".txt":
        pages = read_txt(path)
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {ext}")
    return name, pages
