import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pexams.names import (  # noqa: F401  (re-exported for older imports)
    common_prefix_len, levenshtein_distance, levenshtein_ratio, name_order_variants, normalize_name,
    normalize_student_id, token_name_similarity,
)

__all__ = [
    "StudentMatch", "match_scanned_students", "write_student_matches",
    # Re-exported from pexams.names for older imports.
    "common_prefix_len", "levenshtein_distance", "levenshtein_ratio", "name_order_variants", "normalize_name",
    "normalize_student_id", "token_name_similarity",
]


@dataclass
class StudentMatch:
    scanned_id: str
    ocr_student_id: str
    ocr_student_name: str
    roster_student_id: str = ""
    roster_student_name: str = ""
    match_type: str = "unmatched"
    match_score: float = 0.0

    @property
    def matched(self) -> bool:
        return bool(self.roster_student_id and self.roster_student_name)


def _best_id_match(
    ocr_student_id: str,
    candidates: Iterable[dict],
    id_column: str,
    name_column: str,
    threshold: float = 80.0,
) -> Tuple[Optional[dict], str, float]:
    normalized_ocr_id = normalize_student_id(ocr_student_id)
    if len(normalized_ocr_id) < 4:
        return None, "unmatched", 0.0

    best_row = None
    best_score = 0.0
    for candidate in candidates:
        for column, value in candidate.items():
            if column == name_column:
                continue

            normalized_value = normalize_student_id(value)
            if len(normalized_value) < 4:
                continue

            if normalized_value == normalized_ocr_id:
                return candidate, "id-exact", 100.0

            score = levenshtein_ratio(normalized_ocr_id, normalized_value)
            if score > best_score:
                best_row = candidate
                best_score = score

    if best_row is not None and best_score >= threshold:
        return best_row, "id-fuzzy", best_score
    return None, "unmatched", best_score


def _read_roster(path: str, id_column: str, name_column: str, encoding: str = "utf-8", sep: str = ",") -> List[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required to read Excel rosters") from e
        rows = pd.read_excel(file_path).fillna("").astype(str).to_dict(orient="records")
    else:
        delimiter = "\t" if suffix == ".tsv" else sep
        with file_path.open(newline="", encoding=encoding) as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
    if not rows:
        return []
    missing = [col for col in (id_column, name_column) if col not in rows[0]]
    if missing:
        raise ValueError(f"Roster column(s) not found: {', '.join(missing)}")
    return rows


def _best_name_match(student_name: str, candidates: Iterable[dict], name_column: str, threshold: float) -> Tuple[Optional[dict], str, float]:
    keys = name_order_variants(student_name)
    if not keys:
        return None, "unmatched", 0.0

    best_row = None
    best_score = 0.0
    best_type = "fuzzy"
    for row in candidates:
        candidate_name = row.get(name_column, "")
        for candidate_key in name_order_variants(candidate_name):
            for key in keys:
                if key == candidate_key:
                    return row, "exact", 100.0

                score = levenshtein_ratio(key, candidate_key)
                if score > best_score:
                    best_row = row
                    best_score = score
                    best_type = "fuzzy"

                token_score = token_name_similarity(key, candidate_key)
                if token_score > best_score:
                    best_row = row
                    best_score = token_score
                    best_type = "token-fuzzy"

    if best_row is not None and best_score >= threshold:
        return best_row, best_type, best_score
    return None, "unmatched", best_score


def match_scanned_students(
    page_results: List[dict],
    roster_csv: str,
    id_column: str,
    name_column: str,
    output_csv: str,
    fuzzy_threshold: float = 70.0,
    encoding: str = "utf-8",
    sep: str = ",",
) -> List[StudentMatch]:
    roster_rows = _read_roster(roster_csv, id_column, name_column, encoding=encoding, sep=sep)
    unmatched_roster = list(roster_rows)
    matches: List[StudentMatch] = []

    for result in page_results:
        scanned_id = str(result.get("scanned_id", "")).strip()
        ocr_student_id = str(result.get("ocr_student_id", "")).strip()
        ocr_student_name = str(result.get("ocr_student_name", "")).strip()

        # A sheet may contain a Moodle username while the requested output ID
        # is a separate institutional identifier. Match OCR IDs/usernames
        # against any non-name roster column, but keep returning the configured
        # id_column so downstream marks and feedback use Moodle's expected key.
        id_row, id_match_type, id_score = _best_id_match(
            ocr_student_id,
            unmatched_roster,
            id_column=id_column,
            name_column=name_column,
        )
        if id_row is not None:
            unmatched_roster.remove(id_row)
            matches.append(
                StudentMatch(
                    scanned_id=scanned_id,
                    ocr_student_id=ocr_student_id,
                    ocr_student_name=ocr_student_name,
                    roster_student_id=str(id_row.get(id_column, "")).strip(),
                    roster_student_name=str(id_row.get(name_column, "")).strip(),
                    match_type=id_match_type,
                    match_score=id_score,
                )
            )
            continue

        row, match_type, score = _best_name_match(
            ocr_student_name,
            unmatched_roster,
            name_column=name_column,
            threshold=fuzzy_threshold,
        )
        if row is None:
            matches.append(StudentMatch(scanned_id, ocr_student_id, ocr_student_name, match_score=score))
            continue

        unmatched_roster.remove(row)
        matches.append(
            StudentMatch(
                scanned_id=scanned_id,
                ocr_student_id=ocr_student_id,
                ocr_student_name=ocr_student_name,
                roster_student_id=str(row.get(id_column, "")).strip(),
                roster_student_name=str(row.get(name_column, "")).strip(),
                match_type=match_type,
                match_score=score,
            )
        )

    write_student_matches(output_csv, matches)
    unmatched = [m for m in matches if not m.matched]
    if unmatched:
        logging.error(
            "Student matching halted correction: %d/%d scanned page(s) unmatched. Review %s.",
            len(unmatched),
            len(matches),
            output_csv,
        )
    return matches


def write_student_matches(output_csv: str, matches: List[StudentMatch]) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scanned_id",
        "ocr_student_id",
        "ocr_student_name",
        "roster_student_id",
        "roster_student_name",
        "match_type",
        "match_score",
        "status",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            writer.writerow({
                "scanned_id": match.scanned_id,
                "ocr_student_id": match.ocr_student_id,
                "ocr_student_name": match.ocr_student_name,
                "roster_student_id": match.roster_student_id,
                "roster_student_name": match.roster_student_name,
                "match_type": match.match_type,
                "match_score": f"{match.match_score:.1f}",
                "status": "matched" if match.matched else "unmatched",
            })
