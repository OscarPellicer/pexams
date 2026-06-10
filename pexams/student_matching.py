import csv
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9\s]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def name_order_variants(name: str) -> List[str]:
    normalized = normalize_name(name)
    tokens = normalized.split()
    variants = [normalized]

    if len(tokens) >= 2:
        for split in range(1, len(tokens)):
            variants.append(" ".join(tokens[split:] + tokens[:split]))

    seen = set()
    unique_variants = []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            unique_variants.append(variant)
    return unique_variants


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 100.0
    return 100.0 * (1.0 - levenshtein_distance(a, b) / max_len)


def common_prefix_len(a: str, b: str) -> int:
    count = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        count += 1
    return count


def token_name_similarity(query_key: str, candidate_key: str) -> float:
    query_tokens = query_key.split()
    candidate_tokens = candidate_key.split()
    if not query_tokens or not candidate_tokens:
        return 0.0

    token_scores = []
    exact_matches = 0
    for query_token in query_tokens:
        best_token_score = 0.0
        for candidate_token in candidate_tokens:
            if query_token == candidate_token:
                best_token_score = 1.0
                exact_matches += 1
                break

            prefix_len = common_prefix_len(query_token, candidate_token)
            min_len = min(len(query_token), len(candidate_token))
            if min_len >= 5 and prefix_len >= 4:
                best_token_score = max(best_token_score, 0.85)
            elif min_len >= 4 and prefix_len >= 3:
                best_token_score = max(best_token_score, 0.75)

        token_scores.append(best_token_score)

    if exact_matches == 0:
        return 0.0
    return 100.0 * sum(token_scores) / len(token_scores)


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
