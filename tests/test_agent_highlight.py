from datetime import date

from rag.agent import run_agent_loop, run_heuristic_tools, tools_context_block
from rag.highlight import find_overlap_spans, highlight_answer_html, highlight_answer_markdown
from rag.tools import (
    execute_tool,
    extract_final,
    heuristic_tool_plan,
    parse_tool_calls,
    tool_calculator,
    tool_calendar,
)


def test_calculator_and_calendar():
    assert "5" in tool_calculator("2+3")
    assert "Hata" in tool_calculator("import os") or "hatası" in tool_calculator("__import__('os')").lower()
    today = tool_calendar("today")
    assert str(date.today().year) in today
    assert "gün" in tool_calendar("add_days", days=3).lower() or "ISO" in tool_calendar("add_days", days=3)


def test_parse_tool_calls_and_final():
    text = 'Önce hesapla\nTOOL_CALL {"name": "calculator", "args": {"expression": "1+1"}}\nFINAL sonuç 2'
    calls = parse_tool_calls(text)
    assert calls == [("calculator", {"expression": "1+1"})]
    assert extract_final(text) == "sonuç 2"
    assert "2" in execute_tool("calculator", {"expression": "1+1"})


def test_heuristic_plans():
    plans = heuristic_tool_plan("Bugün tarih nedir?")
    assert any(p[0] == "calendar" for p in plans)
    plans2 = heuristic_tool_plan("12 * 4 + 3 kaç eder?")
    assert any(p[0] == "calculator" for p in plans2)


def test_agent_heuristic_and_llm_stub():
    r = run_heuristic_tools("Bugün hangi gün?")
    assert r.tool_traces
    assert "calendar" in tools_context_block(r.tool_traces)

    def fake_gen(prompt: str) -> str:
        if "TOOL_RESULT" in prompt or "Ön araç" in prompt:
            return "FINAL Bugün araç sonucuna göre biliniyor."
        return 'TOOL_CALL {"name": "calendar", "args": {"action": "today"}}'

    agent = run_agent_loop("Bugün?", fake_gen, max_steps=2, seed_tools=True)
    assert agent.answer
    assert agent.mode == "llm"


def test_highlight_overlaps():
    answer = "Çalışanlar yılda 14 gün yıllık izin hakkına sahiptir."
    sources = [
        {
            "text": "Şirket politikasına göre çalışanlar yılda 14 gün yıllık izin hakkına sahiptir.",
            "anchor": "src-a-c0",
            "label": "a.txt",
        }
    ]
    spans = find_overlap_spans(answer, sources[0]["text"], min_tokens=3)
    assert spans
    html = highlight_answer_html(answer, sources, min_tokens=3)
    assert "<mark" in html
    md = highlight_answer_markdown(answer, sources, min_tokens=3)
    assert "**" in md
