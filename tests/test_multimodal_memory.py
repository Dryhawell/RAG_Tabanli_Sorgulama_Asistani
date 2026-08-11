from rag.memory import (
    AgentMemory,
    extract_facts_heuristic,
    load_memory_from_session,
    save_memory_to_session,
    update_memory_after_turn,
)
from rag.planner import heuristic_plan, parse_plan, run_planned_agent
from rag.types import ChunkMetadata, RetrievedChunk
from rag.vision import (
    IMAGE_EXTENSIONS,
    chunk_has_table,
    is_table_question,
    merge_image_into_question,
    prioritize_table_chunks,
)


def test_table_question_and_boost():
    assert is_table_question("Bu tablodaki satır sayısı nedir?")
    assert is_table_question("Which column has totals?")
    assert not is_table_question("İzin günleri kaç?")

    def _c(text, src="a.txt", cid=0):
        return RetrievedChunk(
            chunk_id=cid,
            score=0.5,
            text=text,
            metadata=ChunkMetadata(
                source_file=src,
                chunk_id=cid,
                page_start=1,
                page_end=1,
                word_count=10,
            ),
        )

    chunks = [
        _c("Düz metin açıklama", cid=0),
        _c("[Tablo]\n| A | B |\n| --- | --- |\n| 1 | 2 |", cid=1),
    ]
    ordered = prioritize_table_chunks(chunks, question="tablodaki değerler?")
    assert "[Tablo]" in ordered[0].text
    assert chunk_has_table(ordered[0])


def test_merge_image_context():
    q = merge_image_into_question("Bu nedir?", "[Görüntü OCR: x.png]\nMerhaba")
    assert "Merhaba" in q and "Bu nedir?" in q
    assert ".png" in IMAGE_EXTENSIONS


def test_memory_roundtrip_and_facts():
    mem = AgentMemory(summary="özet", notes=["n1"], facts=["f1"])
    session = {"id": "abc"}
    save_memory_to_session(session, mem)
    loaded = load_memory_from_session(session)
    assert loaded.summary == "özet"
    assert loaded.notes == ["n1"]

    msgs = [
        {"role": "user", "content": "Unutma: tercih dil Türkçe"},
        {"role": "assistant", "content": "Tamam"},
        {"role": "user", "content": "14 gün izin"},
    ]
    facts = extract_facts_heuristic(msgs)
    assert any("tercih" in f.lower() or "14" in f for f in facts)
    updated = update_memory_after_turn(
        loaded, messages=msgs, plan=["adım1", "adım2"], note="yeni not"
    )
    assert updated.last_plan == ["adım1", "adım2"]
    assert "yeni not" in updated.notes


def test_parse_and_heuristic_plan():
    text = """PLAN
1. Tarihi öğren
2. Hesapla
END_PLAN"""
    assert parse_plan(text) == ["Tarihi öğren", "Hesapla"]
    plan = heuristic_plan("Bugün tarih nedir ve 2+2 hesapla")
    assert any("Takvim" in s or "tarih" in s.lower() for s in plan)
    assert any("Hesap" in s or "hesap" in s.lower() for s in plan)


def test_run_planned_agent_stub():
    calls = {"n": 0}

    def fake_gen(prompt: str) -> str:
        calls["n"] += 1
        if "END_PLAN" in prompt or "Plan:" in prompt:
            return "PLAN\n1. Takvimi kontrol et\n2. Yanıtla\nEND_PLAN"
        if "Takvim" in prompt or "adım" in prompt.lower() or "Şu anki adım" in prompt:
            return 'TOOL_CALL {"name": "calendar", "args": {"action": "today"}}\nFINAL Bugün hazır'
        return "FINAL bitti"

    result = run_planned_agent("Bugün?", fake_gen, use_llm_plan=True, max_steps=3)
    assert result.plan
    assert result.answer
    assert calls["n"] >= 2
