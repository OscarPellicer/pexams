"""Moodle column names, shared by pexams and pevaluate.

Moodle names the columns of its exports after the site language (Catalan, Spanish,
English), sometimes with mangled accents. This file is kept byte-identical in both
packages (``pexams/moodle_columns.py`` and ``autocorrect/moodle_columns.py``); a test in each
checks it.
"""
import re
import unicodedata
from typing import Iterable, Optional

COLUMN_ALIASES = {
    # Assignment grading worksheet
    "participant": [
        "Identificador", "Identifier", "Participant", "Participant ID",
        "Participant identifier", "Identificador del participant",
    ],
    "id": [
        "Numero ID", "Número ID", "Nº ID", "N. ID", "ID number", "Student ID",
        "Identificador de estudiante", "Identificador de l'estudiant", "NÃºmero ID",
    ],
    "name": [
        "Nom complet", "Nombre completo", "Full name", "Name", "Cognoms i nom",
        "Apellidos y nombre", "Cognoms", "Nombre",
    ],
    "status": ["Estat", "Estado", "Status"],
    "grade": [
        "Qualificació", "Qualificació.", "Calificación", "Calificación.",
        "Grade", "Nota", "Puntuación", "QualificaciÃ³",
    ],
    "feedback": [
        "Comentaris de retroacció.", "Comentaris de retroacció",
        "Comentarios de retroacción.", "Comentarios de retroacción",
        "Comentarios de retroalimentación.", "Comentarios de retroalimentación",
        "Feedback comments", "Feedback", "Comentaris de retroacciÃ³.",
    ],
    "submission_modified": [
        "Darrera modificació (tramesa)", "Última modificación (entrega)",
        "Last modified (submission)", "Darrera modificaciÃ³ (tramesa)",
    ],
    "grade_modified": [
        "Darrera modificació (qualificació)", "Última modificación (calificación)",
        "Last modified (grade)", "Darrera modificaciÃ³ (qualificaciÃ³)",
    ],
    # pexams marks files
    "mark": ["mark", "Mark", "Nota", "Grade", "Qualificació", "Calificación"],
    "image_id": ["student_id", "Student ID", "Numero ID", "Número ID", "NÃºmero ID"],
}


def normalize_column_name(value) -> str:
    """'Comentaris de retroacció.' -> 'comentarisderetroaccio': no accents, case or punctuation."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def find_column(columns: Iterable, requested: Optional[str], role: str) -> Optional[str]:
    """The column for ``role``: ``requested`` if present, else the first match (ignoring accents,
    case and punctuation) of ``requested`` or the known Moodle spellings. ``""`` disables it."""
    columns = list(columns)
    if requested == "":
        return None
    if requested and requested in columns:
        return requested
    lookup = {normalize_column_name(column): column for column in columns}
    for candidate in ([requested] if requested else []) + COLUMN_ALIASES.get(role, []):
        resolved = lookup.get(normalize_column_name(candidate))
        if resolved:
            return resolved
    return None
