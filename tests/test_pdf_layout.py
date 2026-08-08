from rag.pdf_layout import (
    compose_page_layout,
    table_to_markdown,
)


def test_table_to_markdown_basic():
    md = table_to_markdown(
        [
            ["Ürün", "Fiyat"],
            ["Elma", "10"],
            ["Armut", "12"],
        ]
    )
    assert md.startswith("[Tablo]")
    assert "| Ürün | Fiyat |" in md
    assert "| Elma | 10 |" in md
    assert "| --- | --- |" in md


def test_table_to_markdown_skips_tiny():
    assert table_to_markdown([["tek"]]) == ""


def test_compose_orders_text_and_tables():
    text_blocks = [
        ((10, 10, 100, 30), "Başlık paragrafı"),
        ((10, 200, 100, 220), "Alt paragraf"),
        # tablo alanındaki metin (atlanmalı)
        ((20, 80, 180, 140), "hücre1 hücre2"),
    ]
    tables = [
        ((15, 70, 190, 160), "[Tablo]\n| A | B |\n| --- | --- |\n| 1 | 2 |"),
    ]
    out = compose_page_layout(text_blocks, tables)
    assert out.index("Başlık") < out.index("[Tablo]")
    assert out.index("[Tablo]") < out.index("Alt paragraf")
    assert "hücre1" not in out
