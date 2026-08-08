from unittest.mock import MagicMock, patch

from rag.readers import _ocr_available, read_document, read_pdf


def test_read_txt_document(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("merhaba dünya", encoding="utf-8")
    name, pages = read_document(str(p), enable_ocr=False)
    assert name == "a.txt"
    assert pages == ["merhaba dünya"]


def test_read_pdf_triggers_ocr_when_text_short(tmp_path):
    # Gerçek PDF olmadan pymupdf yolunu mock'la
    fake_page = MagicMock()
    fake_page.get_text.side_effect = lambda *a, **k: "" if "textpage" not in k else "OCR metni buradadır yeterli uzunlukta"
    # get_text("text") -> ""; get_text("text", textpage=...) -> ocr
    def get_text(mode="text", textpage=None):
        if textpage is not None:
            return "OCR metni buradadır yeterli uzunlukta ve okunabilir"
        return ""

    fake_page.get_text.side_effect = get_text
    fake_page.get_textpage_ocr.return_value = object()

    fake_doc = MagicMock()
    fake_doc.__enter__.return_value = [fake_page]
    fake_doc.__exit__.return_value = False

    with patch("rag.readers._ocr_available", return_value=True), patch(
        "fitz.open", return_value=fake_doc
    ):
        pages = read_pdf("dummy.pdf", enable_ocr=True)

    assert len(pages) == 1
    assert "OCR" in pages[0]
    fake_page.get_textpage_ocr.assert_called()


def test_ocr_available_is_bool():
    assert isinstance(_ocr_available(), bool)
