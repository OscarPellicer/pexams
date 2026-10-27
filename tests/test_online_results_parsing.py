"""Shared row building of the Wooclap and Moodle results parsers (offline)."""
import pandas as pd

from pexams.io.online_results import parse_moodle_results, parse_wooclap_results
from pexams.schemas import PexamOption, PexamQuestion


def _q(qid, text, options, correct):
    return PexamQuestion(id=qid, original_id=f"q{qid}", text=text,
                         options=[PexamOption(text=o, is_correct=k == correct) for k, o in enumerate(options)])


QUESTIONS = [
    _q(1, "What is 2 + 2?", ["3", "4", "5", "22"], 1),
    _q(2, "What is the capital of France?", ["Berlin", "Madrid", "Paris", "Rome"], 2),
]


def test_wooclap_fuzzy_answers_unmatched_na_and_page_after_summary(tmp_path, capsys):
    path = tmp_path / "w.csv"
    pd.DataFrame({
        "Alumno": ["Media", "s1", None],
        "Nombre": ["", "Ana", "Bea"],
        "Q1 - What is 2 plus 2? (1 pts)": ["50 %", "X - 2 2", "V - 4"],
        "Q2 - What is the capital of France? (2 points)": ["", "X - zzzzzzzzzzzz", "V - Berlín"],
    }).to_csv(path, index=False)
    df = parse_wooclap_results(str(path), QUESTIONS)
    assert df.to_dict("records") == [
        {"page": 2, "student_id": "s1", "student_name": "Ana", "model_id": "1", "score": 0.0,
         "total_questions": 2, "answer_1": "D", "answer_2": "NA"},
        {"page": 3, "student_id": "3", "student_name": "Bea", "model_id": "1", "score": 1.0,
         "total_questions": 2, "answer_1": "B", "answer_2": "A"},
    ]
    out = capsys.readouterr().out
    assert "[Wooclap] Fuzzy-matched column 'Q1 - What is 2 plus 2? (1 pts)'" in out
    assert "[Wooclap] Fuzzy-matched answer for Q1 (student=s1): '2 2' → option 3 '22'" in out
    assert "[Wooclap] Parsed 2 student response(s)." in out


def test_moodle_content_mapping_and_fuzzy_answers(tmp_path, capsys):
    path = tmp_path / "m.csv"
    pd.DataFrame({
        "Nom d'usuari": ["a", "b"],
        "Cognoms": ["Smith", "Jones"],
        "Resposta 1": ["Paris", "Pariss"],
        "Resposta 2": ["4", None],
    }).to_csv(path, index=False, sep=";")
    df = parse_moodle_results(str(path), QUESTIONS)
    assert list(df.columns) == ["page", "student_id", "student_name", "model_id", "score",
                                "total_questions", "answer_2", "answer_1"]
    assert df[["answer_1", "answer_2", "score"]].values.tolist() == [["B", "C", 2.0], ["NA", "C", 1.0]]
    assert "[Moodle] Fuzzy-matched answer for Q2 (student=b): 'Pariss' → option 2 'Paris'" in capsys.readouterr().out
