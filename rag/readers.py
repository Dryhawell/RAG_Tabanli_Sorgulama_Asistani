from typing import List, Tuple
import os

# PDF okuma: PyMuPDF tercih, pdfplumber yedek

def _read_pdf_pymupdf(path: str) -> List[str]:
    import fitz  # pymupdf
    texts = []
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
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


def read_pdf(path: str) -> List[str]:
    """PDF'i sayfa sayfa metin olarak döndürür."""
    try:
        return _read_pdf_pymupdf(path)
    except Exception:
        return _read_pdf_pdfplumber(path)


def read_txt(path: str) -> List[str]:
    """TXT'i tek bir sayfa olarak döndürür."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [text]


def read_document(path: str) -> Tuple[str, List[str]]:
    """Dosya tipini algılar ve sayfa metinlerini döndürür.
    Returns: (source_file_name, [page_texts])
    """
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext == ".pdf":
        pages = read_pdf(path)
    elif ext == ".txt":
        pages = read_txt(path)
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {ext}")
    return name, pages
