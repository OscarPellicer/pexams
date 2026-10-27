import os
import zipfile

import pandas as pd
from PIL import Image

from pexams.feedback_zip import create_moodle_feedback_csv, create_moodle_feedback_zip


def test_create_moodle_feedback_zip(tmp_path):
    correction_dir = _sample_feedback_inputs(tmp_path)

    output_zip = tmp_path / "feedback.zip"
    result = create_moodle_feedback_zip(
        moodle_csv=str(tmp_path / "moodle.csv"),
        correction_dir=str(correction_dir),
        output_zip=str(output_zip),
    )

    assert result.added == 1
    assert result.unmatched_roster == 1
    assert result.unmatched_marks == 1
    assert os.path.exists(result.manifest_csv)

    with zipfile.ZipFile(output_zip) as zf:
        names = zf.namelist()
        assert len(names) == 1
        assert "12744361_assignfeedback_file_pexams_feedback_AB27864.pdf" in names[0]
        assert names[0].endswith(".pdf")
        assert zf.read(names[0]).startswith(b"%PDF")

    manifest = pd.read_csv(result.manifest_csv)
    assert set(manifest["status"]) == {"added", "unmatched mark", "no mark"}


def test_create_moodle_feedback_csv(tmp_path):
    correction_dir = _sample_feedback_inputs(tmp_path)

    output_csv = tmp_path / "feedback.csv"
    result = create_moodle_feedback_csv(
        moodle_csv=str(tmp_path / "moodle.csv"),
        correction_dir=str(correction_dir),
        output_csv=str(output_csv),
    )

    assert result.added == 1
    assert result.unmatched_roster == 1
    assert result.unmatched_marks == 1

    df = pd.read_csv(output_csv)
    row = df[df["Número ID"] == "AB27864"].iloc[0]
    assert "data:image/png;base64," in row["Comentaris de retroacció."]
    assert row["Qualificació"] == "8,00"


def test_create_moodle_feedback_zip_resolves_localized_columns(tmp_path):
    correction_dir = _sample_feedback_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "Identifier": "Participant12744361",
                "Full name": "Heras Pardo, Carlos",
                "Student ID": "AB27864",
                "Grade": "",
                "Feedback comments": "",
            }
        ]
    ).to_csv(tmp_path / "moodle_en.csv", index=False)

    output_zip = tmp_path / "feedback_en.zip"
    result = create_moodle_feedback_zip(
        moodle_csv=str(tmp_path / "moodle_en.csv"),
        correction_dir=str(correction_dir),
        output_zip=str(output_zip),
    )

    assert result.added == 1
    with zipfile.ZipFile(output_zip) as zf:
        assert zf.namelist()[0].endswith(".pdf")


def _sample_feedback_inputs(tmp_path):
    correction_dir = tmp_path / "correction"
    images_dir = correction_dir / "scanned_pages"
    images_dir.mkdir(parents=True)

    Image.new("RGB", (300, 420), "white").save(images_dir / "AB27864.png")

    pd.DataFrame(
        [
            {
                "Identificador": "Participant12744361",
                "Nom complet": "Heras Pardo, Carlos",
                "Número ID": "AB27864",
                "Qualificació": "",
                "Comentaris de retroacció.": "",
            },
            {
                "Identificador": "Participant12744362",
                "Nom complet": "Unmatched Student",
                "Número ID": "ZZ99999",
                "Qualificació": "",
                "Comentaris de retroacció.": "",
            },
        ]
    ).to_csv(tmp_path / "moodle.csv", index=False)

    pd.DataFrame(
        [
            {
                "student_id": "AB27864",
                "student_name": "Heras Pardo, Carlos",
                "score": 8,
                "max_score": 10,
                "mark": 8.0,
            },
            {
                "student_id": "NO_ROSTER",
                "student_name": "No Roster",
                "score": 5,
                "max_score": 10,
                "mark": 5.0,
            },
        ]
    ).to_csv(correction_dir / "final_marks.csv", index=False)
    return correction_dir


def _add_edge_cases(tmp_path, correction_dir):
    """A roster row without participant and a matched student without a scanned page."""
    moodle = pd.read_csv(tmp_path / "moodle.csv")
    extra = [
        {"Identificador": "", "Nom complet": "No Participant", "Número ID": "NP1"},
        {"Identificador": "Participant3", "Nom complet": "No Image", "Número ID": "NI1"},
    ]
    pd.concat([moodle, pd.DataFrame(extra)]).to_csv(tmp_path / "moodle.csv", index=False)
    marks = pd.read_csv(correction_dir / "final_marks.csv")
    pd.concat([marks, pd.DataFrame([{"student_id": "NI1", "mark": 3.333}])]).to_csv(
        correction_dir / "final_marks.csv", index=False)


def test_feedback_zip_and_csv_report_missing_images_and_participants(tmp_path):
    correction_dir = _sample_feedback_inputs(tmp_path)
    _add_edge_cases(tmp_path, correction_dir)

    zip_result = create_moodle_feedback_zip(
        moodle_csv=str(tmp_path / "moodle.csv"), correction_dir=str(correction_dir),
        output_zip=str(tmp_path / "feedback.zip"), feedback_file_format="png")
    zip_manifest = pd.read_csv(zip_result.manifest_csv, dtype=str).fillna("")
    assert (zip_result.added, zip_result.unmatched_roster, zip_result.unmatched_marks) == (1, 2, 2)
    assert list(zip_manifest["status"]) == [
        "added", "unmatched mark", "missing image", "missing id or participant", "no mark"]
    assert zip_manifest.loc[2, "mark"] == "3.33"
    with zipfile.ZipFile(tmp_path / "feedback.zip") as zf:
        assert zf.namelist() == ["Heras Pardo, Carlos_12744361_assignfeedback_file_pexams_feedback_AB27864.png"]

    csv_result = create_moodle_feedback_csv(
        moodle_csv=str(tmp_path / "moodle.csv"), correction_dir=str(correction_dir),
        output_csv=str(tmp_path / "feedback.csv"))
    csv_manifest = pd.read_csv(csv_result.manifest_csv, dtype=str).fillna("")
    assert (csv_result.added, csv_result.unmatched_roster, csv_result.unmatched_marks) == (1, 3, 2)
    assert list(csv_manifest["status"]) == [
        "added", "no mark", "missing id or participant", "missing image", "unmatched mark",
        "missing id or participant"]
