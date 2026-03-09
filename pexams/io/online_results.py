"""
Parsers for online quiz platform results (Wooclap, Moodle).

Converts student response files into the standard correction_results.csv
format expected by pexams.analysis.analyze_results().

Matching strategy (for both question text and answer text):
  1. Exact match (case-insensitive, stripped).
  2. Levenshtein fuzzy match as fallback — a message is printed whenever
     fuzzy matching is used so the user can verify mappings.
"""

import csv
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import Levenshtein
import pandas as pd

from pexams.schemas import PexamOption, PexamQuestion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

_ENCODINGS_TO_TRY = [ "utf-8", "utf-8-sig", "latin1", "cp1252"]


def detect_encoding(path: str) -> str:
    """Try common encodings in order and return the first that successfully
    decodes a sample of the file.  Falls back to ``'latin1'``."""
    for enc in _ENCODINGS_TO_TRY:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(4096)
            return enc
        except UnicodeDecodeError:
            continue
    logger.warning("Could not auto-detect encoding for '%s', falling back to 'latin1'.", path)
    return "latin1"


def detect_sep(path: str, encoding: str) -> str:
    """Use csv.Sniffer to detect the delimiter of a CSV file.
    Falls back to ``','`` if detection fails."""
    try:
        with open(path, "r", encoding=encoding) as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except (csv.Error, UnicodeDecodeError):
        return ","


def load_results_file(
    path: str,
    encoding: str = "auto",
    sep: str = "auto",
) -> pd.DataFrame:
    """Load a CSV or XLSX results file into a DataFrame.

    When *encoding* or *sep* are ``'auto'``, they are detected automatically.
    Requires ``openpyxl`` for ``.xlsx`` / ``.xls`` files.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path)
        except ImportError:
            raise ImportError(
                "openpyxl is required to read Excel files. "
                "Install it with: pip install openpyxl"
            )
        logger.info("Loaded %d rows from Excel file: %s", len(df), path)
        return df

    # CSV
    if encoding == "auto":
        encoding = detect_encoding(path)
        logger.info("Auto-detected encoding: %s", encoding)
    if sep == "auto":
        sep = detect_sep(path, encoding)
        logger.info("Auto-detected CSV separator: %r", sep)

    df = pd.read_csv(path, encoding=encoding, sep=sep)
    logger.info("Loaded %d rows from CSV file: %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Answer-to-option matching
# ---------------------------------------------------------------------------

def match_answer_to_option(
    answer_text: str,
    options: List[PexamOption],
) -> Tuple[int, bool]:
    """Match *answer_text* to the best option in *options*.

    Returns ``(index, is_exact)`` where *index* is 0-based (``-1`` when no
    acceptable match was found) and *is_exact* is ``True`` for an exact
    case-insensitive match.

    Strategy:
      1. Exact match (case-insensitive, stripped).
      2. Levenshtein fuzzy match — the best ratio must be ≥ 0.5.
    """
    clean = answer_text.strip()

    # 1. Exact match (case-insensitive)
    for i, opt in enumerate(options):
        if opt.text.strip().lower() == clean.lower():
            return i, True

    # 2. Levenshtein fuzzy match
    best_idx, best_ratio = -1, 0.0
    for i, opt in enumerate(options):
        ratio = Levenshtein.ratio(opt.text.strip(), clean)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio >= 0.5:
        return best_idx, False

    return -1, False


# ---------------------------------------------------------------------------
# Wooclap parser helpers
# ---------------------------------------------------------------------------

# Matches headers like "Q1 - ¿Cual…? (1 pts)" or "Q12 – Some text (2 pts)"
_WOOCLAP_COL_RE = re.compile(
    r"^Q\d+\s*[-–]\s*(.+?)\s*\(\d+\s*pts?\)\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Strips the "V - " or "X - " correctness prefix from answer cells
_WOOCLAP_PREFIX_RE = re.compile(r"^[VX]\s*[-–]\s*", re.IGNORECASE)


def _match_question_by_text(
    col_text: str,
    questions: List[PexamQuestion],
    fuzzy_threshold: float,
    col_name: str,
) -> Optional[PexamQuestion]:
    """Match a column header text to the best :class:`PexamQuestion` by text.

    Tries exact match first, then fuzzy.  Prints a notice (and logs at INFO)
    whenever fuzzy matching is used so the user can verify the mapping.
    Returns ``None`` and logs a warning when no match meets the threshold.
    """
    clean = col_text.strip()

    # 1. Exact match
    for q in questions:
        if q.text.strip() == clean:
            return q

    # 2. Fuzzy match
    best_q, best_ratio = None, 0.0
    for q in questions:
        ratio = Levenshtein.ratio(q.text.strip(), clean)
        if ratio > best_ratio:
            best_ratio = ratio
            best_q = q

    if best_ratio >= fuzzy_threshold and best_q is not None:
        msg = (
            f"[Wooclap] Fuzzy-matched column '{col_name}' "
            f"(ratio={best_ratio:.3f}) to question: "
            f"'{best_q.text[:70]}...'"
        )
        print(msg)
        logger.info(msg)
        return best_q

    logger.warning(
        "Could not match Wooclap column '%s' to any question "
        "(best ratio=%.3f, threshold=%.2f). Column will be skipped.",
        col_name, best_ratio, fuzzy_threshold,
    )
    return None


def _detect_student_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Heuristically identify student-ID and student-name columns.

    Searches column names for common patterns from Wooclap and Moodle exports
    in several languages.  Falls back to the first two columns.
    """
    id_keywords   = ["id", "usuari", "usuario", "username", "email", "alumno", "#"]
    name_keywords = ["cognoms", "apellidos", "last", "nom", "nombre", "name"]

    id_col = name_col = None
    for col in df.columns:
        col_l = str(col).lower().strip()
        if id_col is None and any(k in col_l for k in id_keywords):
            id_col = col
        if name_col is None and any(k in col_l for k in name_keywords):
            name_col = col

    if id_col is None and len(df.columns) >= 1:
        id_col = df.columns[0]
    if name_col is None and len(df.columns) >= 2:
        name_col = df.columns[1]

    return id_col, name_col


def _cell_is_blank(cell) -> bool:
    """Return True for NaN / empty-string / '/' cells."""
    if pd.isna(cell):
        return True
    s = str(cell).strip()
    return s in ("", "/", "nan")


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

def parse_wooclap_results(
    results_path: str,
    questions: List[PexamQuestion],
    fuzzy_threshold: float = 0.80,
    encoding: str = "auto",
    sep: str = "auto",
) -> pd.DataFrame:
    """Parse a Wooclap results file (CSV or XLSX) into the standard
    ``correction_results.csv`` format used by
    :func:`pexams.analysis.analyze_results`.

    Expected column header format::

        Q1 - <question text> (N pts)

    Expected answer cell format::

        V - <option text>   (correct answer chosen)
        X - <option text>   (wrong answer chosen)
        /  or blank         (no answer)

    Args:
        results_path: Path to the Wooclap results CSV or XLSX file.
        questions: List of :class:`~pexams.schemas.PexamQuestion` objects.
            Must have sequential integer IDs (1, 2, …).
        fuzzy_threshold: Minimum Levenshtein ratio (0–1) for question-text
            matching.  Default 0.80.
        encoding: File encoding, or ``'auto'`` to detect automatically.
        sep: CSV separator, or ``'auto'`` to detect automatically.

    Returns:
        DataFrame with columns:
        ``page, student_id, student_name, model_id, score,
        total_questions, answer_1, answer_2, …``
    """
    df = load_results_file(results_path, encoding=encoding, sep=sep)

    # --- Match answer columns to questions ---
    col_to_question: Dict[str, PexamQuestion] = {}
    for col in df.columns:
        m = _WOOCLAP_COL_RE.match(str(col))
        if m:
            col_text = m.group(1).strip()
            q = _match_question_by_text(col_text, questions, fuzzy_threshold, str(col))
            if q is not None:
                col_to_question[col] = q

    if not col_to_question:
        raise ValueError(
            "No answer columns could be matched to questions. "
            "Verify that the results file has columns in the format "
            "'Q1 - <question text> (N pts)' and that the question texts "
            "match those in the questions file."
        )

    matched_questions = list(col_to_question.values())
    print(f"[Wooclap] Matched {len(col_to_question)} answer column(s).")

    id_col, name_col = _detect_student_cols(df)

    # --- Build correction_results rows ---
    rows = []
    for page_idx, (_, srow) in enumerate(df.iterrows(), start=1):
        # Skip summary/percentage rows that Wooclap sometimes appends
        first_cell = str(srow.iloc[0]).strip() if len(srow) > 0 else ""
        if re.match(r"^\d+\.?\d*\s*%$", first_cell):
            continue

        student_id = (
            str(srow[id_col]).strip()
            if id_col and not pd.isna(srow[id_col])
            else str(page_idx)
        )
        student_name = (
            str(srow[name_col]).strip()
            if name_col and name_col in srow.index and not pd.isna(srow[name_col])
            else ""
        )

        answers: Dict[int, str] = {}
        for col, question in col_to_question.items():
            cell = srow[col]

            if _cell_is_blank(cell):
                answers[question.id] = "NA"
                continue

            # Strip correctness prefix ("V - " or "X - ")
            cell_text = _WOOCLAP_PREFIX_RE.sub("", str(cell).strip())

            opt_idx, is_exact = match_answer_to_option(cell_text, question.options)

            if not is_exact and opt_idx >= 0:
                msg = (
                    f"[Wooclap] Fuzzy-matched answer for Q{question.id} "
                    f"(student={student_id}): "
                    f"'{cell_text[:50]}' → "
                    f"option {opt_idx} '{question.options[opt_idx].text[:50]}'"
                )
                print(msg)
                logger.info(msg)

            if opt_idx >= 0:
                answers[question.id] = chr(ord("A") + opt_idx)
            else:
                logger.warning(
                    "Could not match answer '%s' to any option for Q%s "
                    "(student=%s). Marking as NA.",
                    cell_text[:80], question.id, student_id,
                )
                answers[question.id] = "NA"

        score = sum(
            1
            for q in matched_questions
            if (
                q.correct_answer_index is not None
                and answers.get(q.id) == chr(ord("A") + q.correct_answer_index)
            )
        )

        row: Dict = {
            "page": page_idx,
            "student_id": student_id,
            "student_name": student_name,
            "model_id": "1",
            "score": score,
            "total_questions": len(col_to_question),
        }
        for q in matched_questions:
            row[f"answer_{q.id}"] = answers.get(q.id, "NA")
        rows.append(row)

    if not rows:
        raise ValueError("No student rows found in the Wooclap results file.")

    result_df = pd.DataFrame(rows)
    print(f"[Wooclap] Parsed {len(result_df)} student response(s).")
    return result_df


# ---------------------------------------------------------------------------

# Matches Moodle answer column names in several locales:
# "Resposta 1", "Respuesta 2", "Response 3", "Réponse 4", …
_MOODLE_ANSWER_COL_RE = re.compile(
    r"^R(?:esposta|espuesta|esponse|[eé]ponse)\s*(\d+)",
    re.IGNORECASE,
)


def parse_moodle_results(
    results_path: str,
    questions: List[PexamQuestion],
    question_order: Optional[List[int]] = None,
    encoding: str = "auto",
    sep: str = "auto",
) -> pd.DataFrame:
    """Parse a Moodle results file (CSV or XLSX) into the standard
    ``correction_results.csv`` format.

    Answer columns are detected by locale-flexible regex
    (``'Resposta N'``, ``'Respuesta N'``, ``'Response N'``, …).

    Questions are mapped **positionally** by default:
    ``Resposta 1`` → ``questions[0]``, ``Resposta 2`` → ``questions[1]``, …
    Override with *question_order* (a list of 0-based indices into *questions*).

    The student's chosen answer text is matched to question options via
    exact match first, then Levenshtein closest.

    Args:
        results_path: Path to the Moodle results CSV or XLSX file.
        questions: List of :class:`~pexams.schemas.PexamQuestion` objects.
            Must have sequential integer IDs (1, 2, …).
        question_order: Optional list of 0-based indices into *questions*
            that maps ``Resposta N`` → ``questions[question_order[N-1]]``.
            Defaults to sequential order.
        encoding: File encoding, or ``'auto'`` to detect automatically.
        sep: CSV separator, or ``'auto'`` to detect automatically.

    Returns:
        DataFrame with columns:
        ``page, student_id, student_name, model_id, score,
        total_questions, answer_1, answer_2, …``
    """
    df = load_results_file(results_path, encoding=encoding, sep=sep)

    # --- Identify answer columns ---
    answer_cols: List[Tuple[int, str]] = []  # (answer_number, col_name)
    for col in df.columns:
        m = _MOODLE_ANSWER_COL_RE.match(str(col))
        if m:
            answer_cols.append((int(m.group(1)), col))
    answer_cols.sort()

    if not answer_cols:
        raise ValueError(
            "No answer columns detected. "
            "Expected columns like 'Resposta N', 'Respuesta N', or 'Response N'."
        )

    # --- Map answer columns to questions ---
    if question_order is not None:
        if len(question_order) != len(answer_cols):
            raise ValueError(
                f"question_order length ({len(question_order)}) must match "
                f"number of answer columns ({len(answer_cols)})."
            )
        ordered_questions = [questions[i] for i in question_order]
    else:
        n = min(len(answer_cols), len(questions))
        if len(questions) < len(answer_cols):
            logger.warning(
                "Moodle results has %d answer columns but only %d questions "
                "were loaded. The last %d column(s) will be ignored.",
                len(answer_cols), len(questions), len(answer_cols) - len(questions),
            )
        ordered_questions = questions[:n]
        answer_cols = answer_cols[:n]

    print(
        f"[Moodle] Mapping {len(answer_cols)} answer column(s) "
        f"to {len(ordered_questions)} question(s)."
    )

    id_col, name_col = _detect_student_cols(df)

    # --- Build correction_results rows ---
    rows = []
    for page_idx, (_, srow) in enumerate(df.iterrows(), start=1):
        student_id = (
            str(srow[id_col]).strip()
            if id_col and not pd.isna(srow[id_col])
            else str(page_idx)
        )
        student_name = (
            str(srow[name_col]).strip()
            if name_col and name_col in srow.index and not pd.isna(srow[name_col])
            else ""
        )

        answers: Dict[int, str] = {}
        for (_, col), question in zip(answer_cols, ordered_questions):
            cell = srow[col]

            if _cell_is_blank(cell):
                answers[question.id] = "NA"
                continue

            cell_text = str(cell).strip()
            opt_idx, is_exact = match_answer_to_option(cell_text, question.options)

            if not is_exact and opt_idx >= 0:
                msg = (
                    f"[Moodle] Fuzzy-matched answer for Q{question.id} "
                    f"(student={student_id}): "
                    f"'{cell_text[:50]}' → "
                    f"option {opt_idx} '{question.options[opt_idx].text[:50]}'"
                )
                print(msg)
                logger.info(msg)

            if opt_idx >= 0:
                answers[question.id] = chr(ord("A") + opt_idx)
            else:
                logger.warning(
                    "Could not match Moodle answer '%s' to any option for Q%s "
                    "(student=%s). Marking as NA.",
                    cell_text[:80], question.id, student_id,
                )
                answers[question.id] = "NA"

        score = sum(
            1
            for q in ordered_questions
            if (
                q.correct_answer_index is not None
                and answers.get(q.id) == chr(ord("A") + q.correct_answer_index)
            )
        )

        row: Dict = {
            "page": page_idx,
            "student_id": student_id,
            "student_name": student_name,
            "model_id": "1",
            "score": score,
            "total_questions": len(answer_cols),
        }
        for q in ordered_questions:
            row[f"answer_{q.id}"] = answers.get(q.id, "NA")
        rows.append(row)

    if not rows:
        raise ValueError("No student rows found in the Moodle results file.")

    result_df = pd.DataFrame(rows)
    print(f"[Moodle] Parsed {len(result_df)} student response(s).")
    return result_df
