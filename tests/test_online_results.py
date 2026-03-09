"""Tests for pexams.io.online_results and pexams.utils.create_solutions_from_questions."""

import pandas as pd
import pytest

from pexams.io.online_results import (
    detect_encoding,
    detect_sep,
    match_answer_to_option,
    parse_moodle_results,
    parse_wooclap_results,
)
from pexams.schemas import PexamOption, PexamQuestion
from pexams.utils import create_solutions_from_questions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def questions():
    q1 = PexamQuestion(
        id=1,
        original_id="q_math_001",
        text="What is 2 + 2?",
        options=[
            PexamOption(text="3", is_correct=False),
            PexamOption(text="4", is_correct=True),
            PexamOption(text="5", is_correct=False),
            PexamOption(text="22", is_correct=False),
        ],
    )
    q2 = PexamQuestion(
        id=2,
        original_id="q_geo_001",
        text="What is the capital of France?",
        options=[
            PexamOption(text="Berlin", is_correct=False),
            PexamOption(text="Madrid", is_correct=False),
            PexamOption(text="Paris", is_correct=True),
            PexamOption(text="Rome", is_correct=False),
        ],
    )
    q3 = PexamQuestion(
        id=3,
        original_id="q_sci_001",
        text="Water boils at what temperature (°C) at sea level?",
        options=[
            PexamOption(text="90", is_correct=False),
            PexamOption(text="100", is_correct=True),
            PexamOption(text="110", is_correct=False),
        ],
    )
    return [q1, q2, q3]


@pytest.fixture
def city_options():
    return [
        PexamOption(text="Berlin", is_correct=False),
        PexamOption(text="Madrid", is_correct=False),
        PexamOption(text="Paris", is_correct=True),
        PexamOption(text="Rome", is_correct=False),
    ]


def _wooclap_df():
    return pd.DataFrame({
        "Alumno": ["1", "2", "3"],
        "Q1 - What is 2 + 2? (1 pts)": ["V - 4", "X - 3", "/"],
        "Q2 - What is the capital of France? (1 pts)": ["V - Paris", "V - Paris", "X - Berlin"],
        "Q3 - Water boils at what temperature (°C) at sea level? (1 pts)": ["V - 100", "X - 90", "V - 100"],
        "Total": ["3 / 3", "1 / 3", "1 / 3"],
    })


def _moodle_df():
    return pd.DataFrame({
        "Cognoms": ["Smith", "Jones", "Doe"],
        "Nom": ["Alice", "Bob", "Carol"],
        "Nom d'usuari": ["asmith", "bjones", "cdoe"],
        "Resposta 1": ["4", "3", None],
        "Resposta 2": ["Paris", "Paris", "Berlin"],
        "Resposta 3": ["100", "90", "100"],
    })


def _write_csv(df, tmp_path, encoding="utf-8", name="data.csv"):
    p = tmp_path / name
    df.to_csv(str(p), index=False, encoding=encoding)
    return str(p)


def _write_xlsx(df, tmp_path, name="data.xlsx"):
    p = tmp_path / name
    df.to_excel(str(p), index=False)
    return str(p)


# ---------------------------------------------------------------------------
# match_answer_to_option
# ---------------------------------------------------------------------------


def test_match_exact(city_options):
    idx, is_exact = match_answer_to_option("Paris", city_options)
    assert idx == 2
    assert is_exact


def test_match_case_insensitive(city_options):
    idx, is_exact = match_answer_to_option("paris", city_options)
    assert idx == 2
    assert is_exact


def test_match_strips_whitespace(city_options):
    idx, is_exact = match_answer_to_option("  Paris  ", city_options)
    assert idx == 2
    assert is_exact


def test_match_fuzzy_typo(city_options):
    idx, is_exact = match_answer_to_option("Pariis", city_options)
    assert idx == 2
    assert not is_exact


def test_match_no_match_returns_minus_one(city_options):
    idx, _ = match_answer_to_option("xyzabc123", city_options)
    assert idx == -1


def test_match_first_option(city_options):
    idx, is_exact = match_answer_to_option("Berlin", city_options)
    assert idx == 0
    assert is_exact


# ---------------------------------------------------------------------------
# detect_encoding / detect_sep
# ---------------------------------------------------------------------------


def test_detect_utf8_encoding(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert detect_encoding(str(p)) in ("utf-8-sig", "utf-8")


def test_detect_semicolon_sep(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    assert detect_sep(str(p), "utf-8") == ";"


def test_detect_comma_sep(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert detect_sep(str(p), "utf-8") == ","


# ---------------------------------------------------------------------------
# parse_wooclap_results
# ---------------------------------------------------------------------------


def test_wooclap_basic_parsing(tmp_path, questions):
    path = _write_csv(_wooclap_df(), tmp_path, encoding="utf-8-sig")
    result = parse_wooclap_results(path, questions)
    assert len(result) == 3
    assert {"answer_1", "answer_2", "answer_3"}.issubset(result.columns)


def test_wooclap_correct_answer_mapping(tmp_path, questions):
    path = _write_csv(_wooclap_df(), tmp_path, encoding="utf-8-sig")
    result = parse_wooclap_results(path, questions)
    assert result.loc[0, "answer_1"] == "B"   # "4" is index 1 → B
    assert result.loc[1, "answer_1"] == "A"   # "3" is index 0 → A
    assert result.loc[2, "answer_1"] == "NA"  # "/" → blank


def test_wooclap_score_column(tmp_path, questions):
    path = _write_csv(_wooclap_df(), tmp_path, encoding="utf-8-sig")
    result = parse_wooclap_results(path, questions)
    assert result.loc[0, "score"] == 3
    assert result.loc[1, "score"] == 1


def test_wooclap_model_id_is_always_one(tmp_path, questions):
    path = _write_csv(_wooclap_df(), tmp_path, encoding="utf-8-sig")
    result = parse_wooclap_results(path, questions)
    assert (result["model_id"] == "1").all()


def test_wooclap_fuzzy_question_matching(tmp_path, questions):
    df = _wooclap_df().rename(columns={
        "Q1 - What is 2 + 2? (1 pts)": "Q1 - What is 2+2? (1 pts)"
    })
    path = _write_csv(df, tmp_path, encoding="utf-8-sig")
    result = parse_wooclap_results(path, questions, fuzzy_threshold=0.70)
    assert "answer_1" in result.columns


def test_wooclap_summary_row_skipped(tmp_path, questions):
    df = _wooclap_df()
    summary = pd.DataFrame({col: ["76.67%"] for col in df.columns})
    df = pd.concat([df, summary], ignore_index=True)
    path = _write_csv(df, tmp_path, encoding="utf-8-sig")
    result = parse_wooclap_results(path, questions)
    assert len(result) == 3


def test_wooclap_xlsx_loading(tmp_path, questions):
    path = _write_xlsx(_wooclap_df(), tmp_path)
    result = parse_wooclap_results(path, questions)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# parse_moodle_results
# ---------------------------------------------------------------------------


def test_moodle_basic_parsing(tmp_path, questions):
    path = _write_csv(_moodle_df(), tmp_path)
    result = parse_moodle_results(path, questions)
    assert len(result) == 3
    assert {"answer_1", "answer_2"}.issubset(result.columns)


def test_moodle_answer_mapping(tmp_path, questions):
    path = _write_csv(_moodle_df(), tmp_path)
    result = parse_moodle_results(path, questions)
    assert result.loc[0, "answer_1"] == "B"   # "4" → index 1 → B
    assert result.loc[1, "answer_1"] == "A"   # "3" → index 0 → A
    assert result.loc[2, "answer_1"] == "NA"  # None → blank
    assert result.loc[0, "answer_2"] == "C"   # "Paris" → index 2 → C
    assert result.loc[2, "answer_2"] == "A"   # "Berlin" → index 0 → A


def test_moodle_score_column(tmp_path, questions):
    path = _write_csv(_moodle_df(), tmp_path)
    result = parse_moodle_results(path, questions)
    assert result.loc[0, "score"] == 3   # Alice: all correct
    assert result.loc[1, "score"] == 1   # Bob: only Q2 correct


def test_moodle_custom_question_order(tmp_path, questions):
    path = _write_csv(_moodle_df(), tmp_path)
    result = parse_moodle_results(path, questions, question_order=[1, 0, 2])
    # Resposta 1 ("4") now maps to Q2 (capital of France) → no match → NA
    assert result.loc[0, "answer_2"] == "NA"


def test_moodle_spanish_column_names(tmp_path, questions):
    df = _moodle_df().rename(columns={
        "Resposta 1": "Respuesta 1",
        "Resposta 2": "Respuesta 2",
        "Resposta 3": "Respuesta 3",
    })
    path = _write_csv(df, tmp_path)
    result = parse_moodle_results(path, questions)
    assert len(result) == 3
    assert "answer_1" in result.columns


def test_moodle_xlsx_loading(tmp_path, questions):
    path = _write_xlsx(_moodle_df(), tmp_path)
    result = parse_moodle_results(path, questions)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# create_solutions_from_questions
# ---------------------------------------------------------------------------


def test_solutions_returns_three_items(questions):
    assert len(create_solutions_from_questions(questions)) == 3


def test_solutions_single_model_key(questions):
    sol_full, sol_simple, _ = create_solutions_from_questions(questions)
    assert "1" in sol_full
    assert "1" in sol_simple


def test_solutions_max_score_equals_question_count(questions):
    _, _, max_score = create_solutions_from_questions(questions)
    assert max_score == 3


def test_solutions_correct_answer_indices(questions):
    _, sol_simple, _ = create_solutions_from_questions(questions)
    assert sol_simple["1"][1] == 1   # Q1: "4" is index 1
    assert sol_simple["1"][2] == 2   # Q2: "Paris" is index 2
    assert sol_simple["1"][3] == 1   # Q3: "100" is index 1


def test_solutions_full_contain_options(questions):
    sol_full, _, _ = create_solutions_from_questions(questions)
    q1_data = sol_full["1"][1]
    assert "options" in q1_data
    assert len(q1_data["options"]) == 4


def test_solutions_original_id_preserved(questions):
    sol_full, _, _ = create_solutions_from_questions(questions)
    assert sol_full["1"][1]["original_id"] == "q_math_001"


def test_solutions_custom_model_id(questions):
    sol_full, sol_simple, _ = create_solutions_from_questions(questions, model_id="custom")
    assert "custom" in sol_full
    assert "custom" in sol_simple
