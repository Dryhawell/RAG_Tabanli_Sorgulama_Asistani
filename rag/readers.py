from typing import List, Tuple
import os

from app.config import ENABLE_OCR, OCR_LANGS, OCR_MIN_CHARS

# PDF okuma: PyMuPDF tercih, pdfplumber yedek; düşük metinli sayfalarda OCR


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


def read_pdf(path: str, enable_ocr: bool = ENABLE_OCR) -> List[str]:
    """PDF'i sayfa sayfa metin olarak döndürür.

    Tarama PDF'lerde (metin katmanı zayıf) Tesseract kuruluysa OCR dener.
    """
    try:
        return _read_pdf_pymupdf(path, enable_ocr=enable_ocr)
    except Exception:
        return _read_pdf_pdfplumber(path)


def read_txt(path: str) -> List[str]:
    """TXT'i tek bir sayfa olarak döndürür."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [text]


def read_document(path: str, enable_ocr: bool = ENABLE_OCR) -> Tuple[str, List[str]]:
    """Dosya tipini algılar ve sayfa metinlerini döndürür.
    Returns: (source_file_name, [page_texts])
    """
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext == ".pdf":
        pages = read_pdf(path, enable_ocr=enable_ocr)
    elif ext == ".txt":
        pages = read_txt(path)
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {ext}")
    return name, pages
