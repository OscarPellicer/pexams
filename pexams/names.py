"""Student name and ID matching, shared by pexams and pevaluate.

This file is kept byte-identical in both packages (``pexams/names.py`` and
``autocorrect/names.py``); a test in each checks it.
"""
import re
import unicodedata
from typing import List


def normalize_name(value) -> str:
    """'Pérez Gil, Ana' -> 'ana perez gil': lower case, no accents or punctuation, and the
    'Surname, Name' order of class lists turned into 'Name Surname'."""
    text = str(value or "").strip()
    if "," in text:
        surname, name = (part.strip() for part in text.split(",", 1))
        text = f"{name} {surname}"
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_student_id(value) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def name_order_variants(name) -> List[str]:
    """The normalized name and its rotations ('ana perez gil', 'perez gil ana', 'gil ana perez')."""
    tokens = normalize_name(name).split()
    variants = [" ".join(tokens[split:] + tokens[:split]) for split in range(max(len(tokens), 1))]
    return list(dict.fromkeys(v for v in variants if v))


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
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
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
    """0-100: how well the tokens of ``query_key`` are found in ``candidate_key`` (exact, or
    sharing a long prefix, e.g. a truncated surname). Zero unless one token matches exactly."""
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


def name_similarity(a, b) -> float:
    """0-100, symmetric: the best of edit-distance and token similarity over the name orders,
    so 'Arnau Julio' matches 'Julio Candel, Arnau' whichever is given first."""
    best = 0.0
    for x in name_order_variants(a):
        for y in name_order_variants(b):
            best = max(best, levenshtein_ratio(x, y), token_name_similarity(x, y), token_name_similarity(y, x))
    return best
