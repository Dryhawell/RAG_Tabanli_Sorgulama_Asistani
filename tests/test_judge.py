from rag.judge import (
    JudgeCase,
    evaluate_judge_cases,
    heuristic_judge,
    llm_judge,
    load_judge_cases,
    run_judge_file,
    summarize_judge,
)


def test_heuristic_grounded_and_hallucination():
    good = JudgeCase(
        question="İzin?",
        answer="Yıllık izin 14 gündür.",
        contexts=["Çalışanlar yılda 14 gün yıllık izin hakkına sahiptir."],
        expect_grounded=True,
    )
    bad = JudgeCase(
        question="İzin?",
        answer="Mars kolonisinde pizza partisi düzenlenir.",
        contexts=["Çalışanlar yılda 14 gün yıllık izin hakkına sahiptir."],
        expect_grounded=False,
    )
    assert heuristic_judge(good).passed
    assert heuristic_judge(bad).passed


def test_heuristic_no_answer():
    case = JudgeCase(
        question="Köpek?",
        answer="Bu bilgi dokümanda bulunmamaktadır",
        contexts=["İzin 14 gün."],
        expect_no_answer=True,
    )
    r = heuristic_judge(case)
    assert r.passed
    assert r.grounded


def test_llm_judge_with_stub():
    case = JudgeCase(
        id="stub",
        question="İzin?",
        answer="14 gün yıllık izin vardır.",
        contexts=["14 gün yıllık izin"],
        expect_grounded=True,
    )

    def fake_gen(_prompt: str) -> str:
        return '{"grounded": true, "score": 0.9, "reason": "bağlama uyumlu"}'

    r = llm_judge(case, generate_fn=fake_gen)
    assert r.passed
    assert r.mode == "llm"
    assert r.score == 0.9


def test_judge_cases_file_and_cli_helper():
    path = "evals/judge_cases.json"
    cases = load_judge_cases(path)
    assert len(cases) >= 3
    results = evaluate_judge_cases(cases, mode="heuristic")
    summary = summarize_judge(results)
    assert summary["total"] == len(cases)
    assert summary["accuracy"] == 1.0

    report = run_judge_file(path, mode="heuristic", min_accuracy=1.0)
    assert report["summary"]["ok"] is True
