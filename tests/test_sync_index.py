import csv

from pexams.open_responses import sync_index_with_results


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_names_fixed_in_correction_results_reach_the_open_answer_index(tmp_path):
    _write(tmp_path / "correction_results.csv", [
        {"page": "1", "student_id": "AVIDFER", "student_name": "Àlex Vidal Ferrà Ponç"},
        {"page": "9", "student_id": "NSANLLO", "student_name": "Nerea Sanz Lloret"}])
    _write(tmp_path / "open_responses_index.csv", [
        {"student_id": "AVIDFER", "student_name": "Àlex Vidal Ferrà Pon?", "answer_sheet_page": "1", "question_id": "26"},
        {"student_id": "AVIDFER", "student_name": "Àlex Vidal Ferrà Pon?", "answer_sheet_page": "1", "question_id": "27"},
        {"student_id": "NSANLLO", "student_name": "Nerea Sanz Lloret", "answer_sheet_page": "9", "question_id": "26"}])

    assert sync_index_with_results(str(tmp_path)) == 2
    with open(tmp_path / "open_responses_index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["student_name"] for r in rows] == ["Àlex Vidal Ferrà Ponç"] * 2 + ["Nerea Sanz Lloret"]
    assert list(rows[0].keys()) == ["student_id", "student_name", "answer_sheet_page", "question_id"]
    assert sync_index_with_results(str(tmp_path)) == 0
