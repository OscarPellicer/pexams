import logging
import os
import re
import tempfile
import zipfile
import base64
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pexams.moodle_columns import find_column
from PIL import Image, ImageDraw, ImageFont


@dataclass
class FeedbackZipResult:
    output_zip: str
    manifest_csv: str
    added: int
    unmatched_roster: int
    unmatched_marks: int


@dataclass
class FeedbackCsvResult:
    output_csv: str
    manifest_csv: str
    added: int
    unmatched_roster: int
    unmatched_marks: int


def _normalise_id(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip().upper())


def _resolve_column(df: pd.DataFrame, requested: Optional[str], role: str, source: str, required: bool = True) -> Optional[str]:
    columns = list(df.columns)
    resolved = find_column(columns, requested, role)
    if resolved and requested and requested != resolved:
        logging.info("Resolved Moodle column '%s' to '%s' in %s.", requested, resolved, source)
    if resolved is None and required and requested != "":
        raise ValueError(
            f"Column '{requested or role}' not found in {source}. "
            f"Available columns: {', '.join(str(c) for c in columns)}"
        )
    return resolved


def _safe_moodle_name(value: str) -> str:
    safe = re.sub(r"[^\w .,'()-]+", " ", str(value), flags=re.UNICODE).strip()
    safe = re.sub(r"\s+", " ", safe)
    safe = safe.replace("_", " ")
    return safe or "Student"


def _participant_number(value: str) -> str:
    match = re.search(r"(\d+)", str(value))
    return match.group(1) if match else ""


def _read_table(path: str, encoding: str = "utf-8", sep: str = ",") -> pd.DataFrame:
    if sep == "semi":
        sep = ";"
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", encoding=encoding)
    return pd.read_csv(path, sep=sep, encoding=encoding)


def _format_mark(mark) -> str:
    if pd.isna(mark):
        return ""
    try:
        return f"{float(mark):.2f}"
    except (TypeError, ValueError):
        return str(mark)


def _format_mark_for_csv(mark, decimal_sep: str = ",") -> str:
    value = _format_mark(mark)
    if decimal_sep != ".":
        value = value.replace(".", decimal_sep)
    return value


def _find_image(images_dir: str, student_id: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(images_dir, f"{student_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _load_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _write_feedback_image(
    source_image: str,
    output_image: str,
    display_name: str,
    student_id: str,
    mark,
    max_mark: str,
) -> None:
    image = Image.open(source_image).convert("RGB")
    width, height = image.size
    header_height = max(120, int(width * 0.075))
    canvas = Image.new("RGB", (width, height + header_height), "white")
    canvas.paste(image, (0, header_height))

    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(max(24, int(width * 0.018)), bold=True)
    meta_font = _load_font(max(18, int(width * 0.012)))
    mark_font = _load_font(max(42, int(width * 0.035)), bold=True)

    pad = max(24, int(width * 0.02))
    draw.rectangle((0, 0, width, header_height), fill=(248, 250, 252))
    draw.line((0, header_height - 2, width, header_height - 2), fill=(30, 64, 175), width=4)
    draw.text((pad, pad), display_name, fill=(15, 23, 42), font=title_font)
    draw.text((pad, pad + int(header_height * 0.42)), f"ID: {student_id}", fill=(71, 85, 105), font=meta_font)

    mark_text = f"{_format_mark(mark)} / {max_mark}".strip()
    mark_x = width - pad - _text_width(draw, mark_text, mark_font)
    draw.text((mark_x, pad), mark_text, fill=(185, 28, 28), font=mark_font)

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    canvas.save(output_image)


def _write_feedback_pdf(
    source_image: str,
    output_pdf: str,
    display_name: str,
    student_id: str,
    mark,
    max_mark: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="pexams_feedback_pdf_") as work_dir:
        rendered_png = os.path.join(work_dir, "feedback.png")
        _write_feedback_image(source_image, rendered_png, display_name, student_id, mark, max_mark)
        image = Image.open(rendered_png).convert("RGB")
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        image.save(output_pdf, "PDF", resolution=150.0)


def _write_feedback_file(
    source_image: str,
    output_file: str,
    display_name: str,
    student_id: str,
    mark,
    max_mark: str,
    file_format: str,
) -> None:
    if file_format == "pdf":
        _write_feedback_pdf(source_image, output_file, display_name, student_id, mark, max_mark)
    elif file_format == "png":
        _write_feedback_image(source_image, output_file, display_name, student_id, mark, max_mark)
    else:
        raise ValueError(f"Unsupported feedback file format: {file_format}")


def _build_roster_map(
    moodle_df: pd.DataFrame,
    id_column: str,
    participant_column: str,
    name_column: str,
) -> Tuple[Dict[str, dict], List[dict]]:
    roster = {}
    skipped = []
    for _, row in moodle_df.iterrows():
        student_id = _normalise_id(row.get(id_column))
        participant = _participant_number(row.get(participant_column, ""))
        name = str(row.get(name_column, "")).strip()
        if not student_id or not participant:
            skipped.append({"student_id": student_id, "name": name, "reason": "missing id or participant"})
            continue
        roster[student_id] = {
            "student_id": student_id,
            "participant": participant,
            "name": name,
        }
    return roster, skipped


_UNMATCHED_ROSTER_STATUSES = {"no mark", "missing id or participant"}
_UNMATCHED_MARK_STATUSES = {"unmatched mark", "missing image"}


@dataclass
class _FeedbackInputs:
    """The Moodle roster and the pexams marks, read and with their columns resolved."""
    moodle_df: pd.DataFrame
    marks_df: pd.DataFrame
    images_dir: str
    moodle_csv: str
    marks_csv: str
    id_column: str
    participant_column: str
    name_column: str
    image_id_column: str = ""
    mark_column: str = ""

    def resolve_mark_columns(self, image_id_column: str, mark_column: str) -> None:
        self.image_id_column = _resolve_column(self.marks_df, image_id_column, "image_id", self.marks_csv)
        self.mark_column = _resolve_column(self.marks_df, mark_column, "mark", self.marks_csv)


def _load_feedback_inputs(
    moodle_csv: str,
    correction_dir: str,
    output_path: str,
    output_kind: str,
    id_column: str,
    participant_column: str,
    name_column: str,
    images_dir: Optional[str],
    marks_csv: Optional[str],
    moodle_encoding: str,
    moodle_sep: str,
    overwrite: bool,
) -> _FeedbackInputs:
    """Check the inputs exist (and the output does not, unless ``overwrite``), read the Moodle
    roster and the marks, and resolve the roster's (possibly localised) column names. The
    marks columns are resolved afterwards with ``resolve_mark_columns``."""
    if images_dir is None:
        images_dir = os.path.join(correction_dir, "scanned_pages")
    if marks_csv is None:
        marks_csv = os.path.join(correction_dir, "final_marks.csv")

    if not os.path.exists(moodle_csv):
        raise FileNotFoundError(f"Moodle CSV not found: {moodle_csv}")
    if not os.path.exists(marks_csv):
        raise FileNotFoundError(f"Marks CSV not found: {marks_csv}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Annotated images directory not found: {images_dir}")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output {output_kind} already exists: {output_path}. Use overwrite=True to replace it."
        )

    moodle_df = _read_table(moodle_csv, encoding=moodle_encoding, sep=moodle_sep)
    marks_df = pd.read_csv(marks_csv)
    return _FeedbackInputs(
        moodle_df=moodle_df,
        marks_df=marks_df,
        images_dir=images_dir,
        moodle_csv=moodle_csv,
        marks_csv=marks_csv,
        id_column=_resolve_column(moodle_df, id_column, "id", moodle_csv),
        participant_column=_resolve_column(moodle_df, participant_column, "participant", moodle_csv),
        name_column=_resolve_column(moodle_df, name_column, "name", moodle_csv),
    )


def _manifest_row(
    status: str,
    student_id: str,
    participant: str = "",
    name: str = "",
    mark: str = "",
    source_image: str = "",
    zip_path: Optional[str] = None,
) -> dict:
    """One manifest line; ``zip_path`` is a column only in the zip manifest."""
    row = {
        "student_id": student_id,
        "moodle_participant": participant,
        "name": name,
        "mark": mark,
        "source_image": source_image,
    }
    if zip_path is not None:
        row["zip_path"] = zip_path
    row["status"] = status
    return row


def _skipped_roster_rows(skipped_roster: List[dict], zip_path: Optional[str] = None) -> List[dict]:
    return [
        _manifest_row(row["reason"], row["student_id"], name=row["name"], zip_path=zip_path)
        for row in skipped_roster
    ]


def _write_manifest(output_path: str, manifest_rows: List[dict]) -> Tuple[str, int, int]:
    """Write ``<output>_manifest.csv``; return its path and the unmatched roster/mark counts."""
    manifest_csv = os.path.splitext(output_path)[0] + "_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    unmatched_roster = sum(1 for row in manifest_rows if row["status"] in _UNMATCHED_ROSTER_STATUSES)
    unmatched_marks = sum(1 for row in manifest_rows if row["status"] in _UNMATCHED_MARK_STATUSES)
    return manifest_csv, unmatched_roster, unmatched_marks


def _feedback_html(student_id: str, rendered_png: str, mark_text: str, max_mark: str) -> str:
    """Feedback-comment HTML: the mark and the annotated page as a base64 PNG."""
    with open(rendered_png, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return (
        f'<p><strong>Pexams mark:</strong> {mark_text} / {max_mark}</p>'
        f'<p><img alt="Pexams feedback {student_id}" '
        f'src="data:image/png;base64,{encoded}"></p>'
    )


def create_moodle_feedback_zip(
    moodle_csv: str,
    correction_dir: str,
    output_zip: str,
    id_column: str = "Número ID",
    participant_column: str = "Identificador",
    name_column: str = "Nom complet",
    mark_column: str = "mark",
    images_dir: Optional[str] = None,
    marks_csv: Optional[str] = None,
    moodle_encoding: str = "utf-8",
    moodle_sep: str = ",",
    max_mark: str = "10",
    image_id_column: str = "student_id",
    feedback_file_format: str = "pdf",
    overwrite: bool = False,
) -> FeedbackZipResult:
    """Create a Moodle assignment feedback-file zip from pexams correction output."""
    feedback_file_format = feedback_file_format.lower().strip()
    if feedback_file_format not in {"pdf", "png"}:
        raise ValueError("feedback_file_format must be 'pdf' or 'png'")

    inputs = _load_feedback_inputs(
        moodle_csv, correction_dir, output_zip, "zip", id_column, participant_column, name_column,
        images_dir, marks_csv, moodle_encoding, moodle_sep, overwrite,
    )
    inputs.resolve_mark_columns(image_id_column, mark_column)
    roster, skipped_roster = _build_roster_map(
        inputs.moodle_df, inputs.id_column, inputs.participant_column, inputs.name_column
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_zip)) or ".", exist_ok=True)

    manifest_rows: List[dict] = []
    added = 0
    matched_roster_ids = set()

    with tempfile.TemporaryDirectory(prefix="pexams_feedback_") as work_dir:
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for _, mark_row in inputs.marks_df.iterrows():
                student_id = _normalise_id(mark_row.get(inputs.image_id_column))
                mark = mark_row.get(inputs.mark_column)
                roster_row = roster.get(student_id)
                if not roster_row:
                    manifest_rows.append(_manifest_row(
                        "unmatched mark", student_id, mark=_format_mark(mark), zip_path=""))
                    continue

                matched_roster_ids.add(student_id)
                source_image = _find_image(inputs.images_dir, student_id)
                if not source_image:
                    manifest_rows.append(_manifest_row(
                        "missing image", student_id, roster_row["participant"], roster_row["name"],
                        _format_mark(mark), zip_path=""))
                    continue

                moodle_name = _safe_moodle_name(roster_row["name"])
                feedback_basename = f"pexams_feedback_{student_id}.{feedback_file_format}"
                archive_name = f"{moodle_name}_{roster_row['participant']}_assignfeedback_file_{feedback_basename}"
                rendered_path = os.path.join(work_dir, archive_name)
                _write_feedback_file(
                    source_image=source_image,
                    output_file=rendered_path,
                    display_name=roster_row["name"] or student_id,
                    student_id=student_id,
                    mark=mark,
                    max_mark=max_mark,
                    file_format=feedback_file_format,
                )
                zf.write(rendered_path, archive_name)
                added += 1
                manifest_rows.append(_manifest_row(
                    "added", student_id, roster_row["participant"], roster_row["name"],
                    _format_mark(mark), source_image, zip_path=archive_name))

    manifest_rows.extend(_skipped_roster_rows(skipped_roster, zip_path=""))
    for student_id, row in roster.items():
        if student_id not in matched_roster_ids:
            manifest_rows.append(_manifest_row(
                "no mark", student_id, row["participant"], row["name"], zip_path=""))

    manifest_csv, unmatched_roster, unmatched_marks = _write_manifest(output_zip, manifest_rows)

    logging.info("Created Moodle feedback zip: %s (%d file(s))", os.path.abspath(output_zip), added)
    logging.info("Created Moodle feedback manifest: %s", os.path.abspath(manifest_csv))

    return FeedbackZipResult(
        output_zip=output_zip,
        manifest_csv=manifest_csv,
        added=added,
        unmatched_roster=unmatched_roster,
        unmatched_marks=unmatched_marks,
    )


def _ensure_object_column(df: pd.DataFrame, column: Optional[str]) -> None:
    """Add ``column`` (empty) if missing and make it hold strings."""
    if not column:
        return
    if column not in df.columns:
        df[column] = ""
    df[column] = df[column].astype("object")


def create_moodle_feedback_csv(
    moodle_csv: str,
    correction_dir: str,
    output_csv: str,
    id_column: str = "Número ID",
    participant_column: str = "Identificador",
    name_column: str = "Nom complet",
    feedback_column: str = "Comentaris de retroacció.",
    grade_column: Optional[str] = "Qualificació",
    mark_column: str = "mark",
    images_dir: Optional[str] = None,
    marks_csv: Optional[str] = None,
    moodle_encoding: str = "utf-8",
    moodle_sep: str = ",",
    max_mark: str = "10",
    image_id_column: str = "student_id",
    csv_decimal_sep: str = ",",
    overwrite: bool = False,
) -> FeedbackCsvResult:
    """Create a Moodle grading CSV with base64 PNGs embedded in the feedback column."""
    inputs = _load_feedback_inputs(
        moodle_csv, correction_dir, output_csv, "CSV", id_column, participant_column, name_column,
        images_dir, marks_csv, moodle_encoding, moodle_sep, overwrite,
    )
    moodle_df = inputs.moodle_df
    feedback_column = _resolve_column(moodle_df, feedback_column, "feedback", moodle_csv, required=False) or feedback_column
    grade_column = _resolve_column(moodle_df, grade_column, "grade", moodle_csv, required=False) if grade_column else None
    inputs.resolve_mark_columns(image_id_column, mark_column)
    mark_column = inputs.mark_column
    _ensure_object_column(moodle_df, feedback_column)
    _ensure_object_column(moodle_df, grade_column)

    roster, skipped_roster = _build_roster_map(
        moodle_df, inputs.id_column, inputs.participant_column, inputs.name_column
    )
    mark_by_id = {
        _normalise_id(row.get(inputs.image_id_column)): row
        for _, row in inputs.marks_df.iterrows()
        if _normalise_id(row.get(inputs.image_id_column))
    }

    manifest_rows: List[dict] = []
    added = 0

    with tempfile.TemporaryDirectory(prefix="pexams_feedback_csv_") as work_dir:
        for index, moodle_row in moodle_df.iterrows():
            student_id = _normalise_id(moodle_row.get(inputs.id_column))
            roster_row = roster.get(student_id)
            mark_row = mark_by_id.get(student_id)

            if not roster_row:
                manifest_rows.append(_manifest_row(
                    "missing id or participant", student_id,
                    name=str(moodle_row.get(inputs.name_column, ""))))
                continue
            if mark_row is None:
                manifest_rows.append(_manifest_row(
                    "no mark", student_id, roster_row["participant"], roster_row["name"]))
                continue

            mark = mark_row.get(mark_column)
            mark_text = _format_mark(mark)
            source_image = _find_image(inputs.images_dir, student_id)
            if not source_image:
                manifest_rows.append(_manifest_row(
                    "missing image", student_id, roster_row["participant"], roster_row["name"], mark_text))
                continue

            rendered_path = os.path.join(work_dir, f"pexams_feedback_{student_id}.png")
            _write_feedback_image(
                source_image=source_image,
                output_image=rendered_path,
                display_name=roster_row["name"] or student_id,
                student_id=student_id,
                mark=mark,
                max_mark=max_mark,
            )
            moodle_df.at[index, feedback_column] = _feedback_html(student_id, rendered_path, mark_text, max_mark)
            if grade_column:
                moodle_df.at[index, grade_column] = _format_mark_for_csv(mark, csv_decimal_sep)

            added += 1
            manifest_rows.append(_manifest_row(
                "added", student_id, roster_row["participant"], roster_row["name"], mark_text, source_image))

    for student_id, mark_row in mark_by_id.items():
        if student_id not in roster:
            manifest_rows.append(_manifest_row(
                "unmatched mark", student_id, mark=_format_mark(mark_row.get(mark_column))))
    manifest_rows.extend(_skipped_roster_rows(skipped_roster))

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    sep = ";" if moodle_sep == "semi" else moodle_sep
    moodle_df.to_csv(output_csv, index=False, encoding=moodle_encoding, sep=sep)

    manifest_csv, unmatched_roster, unmatched_marks = _write_manifest(output_csv, manifest_rows)

    logging.info("Created Moodle feedback CSV: %s (%d row(s))", os.path.abspath(output_csv), added)
    logging.info("Created Moodle feedback CSV manifest: %s", os.path.abspath(manifest_csv))

    return FeedbackCsvResult(
        output_csv=output_csv,
        manifest_csv=manifest_csv,
        added=added,
        unmatched_roster=unmatched_roster,
        unmatched_marks=unmatched_marks,
    )
