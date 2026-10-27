"""fill_marks_in_file: reading the roster, exact/fuzzy ID matching, writing marks and final_marks.csv."""
import pandas as pd
import pytest

from pexams.grades import fill_marks_in_file

ROSTER = [
    {"DNI": "ab123", "Nombre": "Ana", "Nota": ""},       # exact (case-insensitive)
    {"DNI": "CD457", "Nombre": "Carlos", "Nota": ""},    # fuzzy with CD456 (80 %)
    {"DNI": "ZZ000", "Nombre": "Zoe", "Nota": ""},       # no exam
    {"DNI": None, "Nombre": "Nadie", "Nota": ""},        # no ID
    {"DNI": "QQ11", "Nombre": "Quique", "Nota": ""},     # fuzzy with QQ111 (88.9 %)
]


def _correction_dir(tmp_path, with_final=True):
    marks = pd.DataFrame([
        {"student_id": "AB123", "score": 7, "max_score": 10, "mark": 7.123},
        {"student_id": "cd456 ", "score": 5, "max_score": 10, "mark": 5.0},
        {"student_id": "XY999", "score": 2, "max_score": 10, "mark": 2.0},
        {"student_id": "QQ111", "score": 9, "max_score": 10, "mark": 9.5},
    ])
    marks.drop(columns=["mark"]).to_csv(tmp_path / "correction_results.csv", index=False)
    if with_final:
        marks.assign(student_name="OCR name").to_csv(tmp_path / "final_marks.csv", index=False)
    return str(tmp_path / "correction_results.csv")


def _marks(path, **kw):
    df = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path, **kw)
    return dict(zip(df["DNI"].fillna(""), df.iloc[:, -1] if "Nota" not in df else df["Nota"]))


def test_exact_match_only(tmp_path):
    results = _correction_dir(tmp_path)
    pd.DataFrame(ROSTER).to_csv(tmp_path / "roster.csv", index=False)
    fill_marks_in_file(str(tmp_path / "roster.csv"), "DNI", "Nota", results)
    marks = _marks(tmp_path / "roster_with_marks.csv")
    assert marks["AB123"] == pytest.approx(7.12)
    assert all(pd.isna(marks[k]) for k in ("CD457", "ZZ000", "QQ11", ""))


def test_fuzzy_match_csv_with_separator_and_decimal_comma(tmp_path):
    results = _correction_dir(tmp_path)
    pd.DataFrame(ROSTER).to_csv(tmp_path / "roster.csv", index=False, sep=";", encoding="latin1")
    fill_marks_in_file(str(tmp_path / "roster.csv"), "DNI", "Nota", results, fuzzy_threshold=70,
                       sep=";", decimal_sep=",", encoding="latin1")
    text = (tmp_path / "roster_with_marks.csv").read_text(encoding="latin1").splitlines()
    assert text == ["DNI;Nombre;Nota", "AB123;Ana;7,12", "CD457;Carlos;5,0", "ZZ000;Zoe;", ";Nadie;", "QQ11;Quique;9,5"]
    # final_marks.csv now carries the roster IDs of the matched exams.
    final = pd.read_csv(tmp_path / "final_marks.csv")
    assert list(final["student_id"]) == ["AB123", "CD457", "XY999", "QQ11"]


def test_fuzzy_match_xlsx_writes_names_to_final_marks(tmp_path):
    results = _correction_dir(tmp_path)
    pd.DataFrame(ROSTER).to_excel(tmp_path / "roster.xlsx", index=False)
    fill_marks_in_file(str(tmp_path / "roster.xlsx"), "DNI", "Nota", results, fuzzy_threshold=70, name_col="Nombre")
    marks = _marks(tmp_path / "roster_with_marks.xlsx")
    assert marks["AB123"] == pytest.approx(7.12) and marks["CD457"] == 5 and marks["QQ11"] == 9.5
    assert pd.isna(marks["ZZ000"])
    final = pd.read_csv(tmp_path / "final_marks.csv")
    assert dict(zip(final["student_id"], final["student_name"].fillna(""))) == {
        "AB123": "Ana", "CD457": "Carlos", "XY999": "OCR name", "QQ11": "Quique"}


def test_simplify_csv_keeps_id_name_mark_and_strips_hash(tmp_path):
    results = _correction_dir(tmp_path)
    pd.DataFrame([{**r, "#Nota": "", "Extra": 1} for r in ROSTER]).to_csv(tmp_path / "roster.csv", index=False)
    fill_marks_in_file(str(tmp_path / "roster.csv"), "DNI", "#Nota", results, fuzzy_threshold=70,
                       name_col="Nombre", simplify_csv=True)
    df = pd.read_csv(tmp_path / "roster_with_marks.csv")
    assert list(df.columns) == ["DNI", "Nombre", "Nota"]
    assert list(df["Nota"].fillna(-1)) == [7.12, 5.0, -1, -1, 9.5]


def test_tsv_without_final_marks_uses_raw_scores(tmp_path):
    results = _correction_dir(tmp_path, with_final=False)
    pd.DataFrame(ROSTER).to_csv(tmp_path / "roster.tsv", sep="\t", index=False)
    fill_marks_in_file(str(tmp_path / "roster.tsv"), "DNI", "Nota", results, fuzzy_threshold=70)
    lines = (tmp_path / "roster_with_marks.tsv").read_text().splitlines()
    assert lines[1] == '"AB123"\t"Ana"\t7' and lines[2] == '"CD457"\t"Carlos"\t5'
    assert not (tmp_path / "final_marks.csv").exists()


def test_missing_id_column_or_unsupported_format_writes_nothing(tmp_path):
    results = _correction_dir(tmp_path)
    pd.DataFrame([{"X": 1}]).to_csv(tmp_path / "roster.csv", index=False)
    fill_marks_in_file(str(tmp_path / "roster.csv"), "DNI", "Nota", results)
    (tmp_path / "roster.txt").write_text("DNI\nAB123\n")
    fill_marks_in_file(str(tmp_path / "roster.txt"), "DNI", "Nota", results)
    assert not list(tmp_path.glob("*_with_marks*"))


def test_roster_names_replace_an_all_empty_name_column(tmp_path):
    results = _correction_dir(tmp_path, with_final=False)
    pd.read_csv(results).assign(student_name="", mark=1.0).to_csv(tmp_path / "final_marks.csv", index=False)
    pd.DataFrame(ROSTER).to_csv(tmp_path / "roster.csv", index=False)
    fill_marks_in_file(str(tmp_path / "roster.csv"), "DNI", "Nota", results, name_col="Nombre")
    final = pd.read_csv(tmp_path / "final_marks.csv")
    assert final.loc[0, "student_id"] == "AB123" and final.loc[0, "student_name"] == "Ana"
