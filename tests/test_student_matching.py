import csv

from pexams.student_matching import match_scanned_students, name_order_variants


def test_name_order_variants_rotates_surname_first_names():
    variants = name_order_variants("Heras Pardo Carlos")

    assert "heras pardo carlos" in variants
    assert "carlos heras pardo" in variants


def test_match_scanned_students_matches_name_order_and_writes_artifact(tmp_path):
    roster = tmp_path / "roster.csv"
    with roster.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "name"])
        writer.writeheader()
        writer.writerow({"student_id": "abc123", "name": "Carlos Heras Pardo"})

    output_csv = tmp_path / "student_matches.csv"
    matches = match_scanned_students(
        [{"scanned_id": "scan_1", "ocr_student_id": "bad-id", "ocr_student_name": "Heras Pardo Carlos"}],
        roster_csv=str(roster),
        id_column="student_id",
        name_column="name",
        output_csv=str(output_csv),
        fuzzy_threshold=70,
    )

    assert matches[0].matched
    assert matches[0].roster_student_id == "abc123"
    assert matches[0].match_type == "exact"
    text = output_csv.read_text(encoding="utf-8")
    assert "scan_1" in text
    assert "Carlos Heras Pardo" in text


def test_match_scanned_students_leaves_unmatched_for_review(tmp_path):
    roster = tmp_path / "roster.csv"
    roster.write_text("student_id,name\nabc123,Carlos Heras Pardo\n", encoding="utf-8")

    output_csv = tmp_path / "student_matches.csv"
    matches = match_scanned_students(
        [{"scanned_id": "scan_1", "ocr_student_id": "", "ocr_student_name": "Totally Different"}],
        roster_csv=str(roster),
        id_column="student_id",
        name_column="name",
        output_csv=str(output_csv),
        fuzzy_threshold=90,
    )

    assert not matches[0].matched
    assert "unmatched" in output_csv.read_text(encoding="utf-8")


def test_match_scanned_students_fuzzy_matches_username_but_returns_configured_id(tmp_path):
    roster = tmp_path / "roster.csv"
    with roster.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Número ID", "Nom complet", "Nom d'usuari"])
        writer.writeheader()
        writer.writerow({
            "Número ID": "FG03269",
            "Nom complet": "Carvajal Chicaiza, José Andrés",
            "Nom d'usuari": "jocarchi",
        })

    output_csv = tmp_path / "student_matches.csv"
    matches = match_scanned_students(
        [{"scanned_id": "scan_1", "ocr_student_id": "JOCARCH", "ocr_student_name": "JOSE AMELIES GIVAYAL"}],
        roster_csv=str(roster),
        id_column="Número ID",
        name_column="Nom complet",
        output_csv=str(output_csv),
        fuzzy_threshold=70,
    )

    assert matches[0].matched
    assert matches[0].roster_student_id == "FG03269"
    assert matches[0].match_type == "id-fuzzy"


def test_names_module_identical_to_pevaluate_copy():
    from pathlib import Path
    import pytest
    pevaluate_names = pytest.importorskip("autocorrect.names")
    import pexams.names as ours
    assert Path(ours.__file__).read_bytes() == Path(pevaluate_names.__file__).read_bytes()


def test_names_normalize_and_symmetric_similarity():
    from pexams.names import name_similarity, normalize_name
    assert normalize_name("Pérez Gil, Ana") == "ana perez gil"
    assert name_similarity("Arnau Julio", "Julio Candel, Arnau") == name_similarity("Julio Candel, Arnau", "Arnau Julio") >= 75


def test_moodle_columns_module_identical_to_pevaluate_copy():
    from pathlib import Path
    import pytest
    theirs = pytest.importorskip("autocorrect.moodle_columns")
    import pexams.moodle_columns as ours
    assert Path(ours.__file__).read_bytes() == Path(theirs.__file__).read_bytes()


def test_moodle_columns_resolve_localised_headers():
    from pexams.moodle_columns import find_column
    cols = ["Identificador", "Nom complet", "Qualificació", "Comentaris de retroacció."]
    assert find_column(cols, "Calificación", "grade") == "Qualificació"
    assert find_column(cols, None, "feedback") == "Comentaris de retroacció."
    assert find_column(cols, "", "name") is None and find_column(cols, None, "status") is None
