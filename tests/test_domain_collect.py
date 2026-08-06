"""Domain veri toplama testleri."""

import json

from rag.domain_collect import (
    append_domain_pairs,
    collect_from_chats,
    collect_from_metrics,
    dedupe_pairs,
    merge_pairs,
)


def test_collect_from_chats(tmp_path):
    chat_dir = tmp_path / "chats" / "alice"
    chat_dir.mkdir(parents=True)
    session = {
        "id": "s1",
        "messages": [
            {"role": "user", "content": "İzin kaç gün?"},
            {
                "role": "assistant",
                "content": "14 gün",
                "sources": [
                    {
                        "text": "Çalışanlar yılda 14 gün ücretli izin hakkına sahiptir.",
                        "score": 0.85,
                        "source_file": "politika.txt",
                    }
                ],
            },
        ],
    }
    with open(chat_dir / "s1.json", "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False)

    pairs = collect_from_chats(str(tmp_path / "chats"), min_gate_score=0.5)
    assert len(pairs) == 1
    assert "14 gün" in pairs[0]["positive"]


def test_collect_from_metrics(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    row = {
        "kind": "query",
        "values": {
            "question": "Stok kodu?",
            "top_chunk": "Atlas Not Defteri stok kodu AT-ND-001",
            "top_source": "urun.txt",
            "gate_score": 0.9,
            "no_answer": False,
        },
    }
    metrics.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    pairs = collect_from_metrics(str(metrics), min_gate_score=0.5)
    assert len(pairs) == 1
    assert pairs[0]["anchor"] == "Stok kodu?"


def test_append_and_dedupe(tmp_path):
    path = str(tmp_path / "pairs.jsonl")
    rows = [
        {"anchor": "Soru A", "positive": "Cevap uzun metin örneği"},
        {"anchor": "Soru B", "positive": "Başka uzun cevap metni"},
    ]
    n1 = append_domain_pairs(rows, path)
    assert n1 == 2
    n2 = append_domain_pairs(rows, path)
    assert n2 == 0
    merged = merge_pairs(rows, [{"anchor": "Soru C", "positive": "Yeni uzun cevap"}])
    assert len(merged) == 3
