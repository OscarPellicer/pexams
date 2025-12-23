import glob
import os
import re
import logging
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from pexams.schemas import PexamExam, PexamQuestion
import Levenshtein

# Global RNGs initialized with default seed 42
_rng_questions = random.Random(42)
_rng_answers = random.Random(42)

def set_seeds(seed_questions: Optional[int] = None, seed_answers: int = 42):
    """
    Sets the global random seeds for question and answer shuffling.
    
    Args:
        seed_questions: If None, questions will NOT be shuffled by shuffle_questions().
                        If int, questions will be shuffled deterministically.
        seed_answers: Must be an int (default 42). Used for shuffling answers.
    """
    global _rng_questions, _rng_answers
    
    if seed_questions is not None:
        _rng_questions = random.Random(seed_questions)
    else:
        # If None, we don't really use this RNG for shuffling order, 
        # but we keep a valid object just in case.
        _rng_questions = None 

    _rng_answers = random.Random(seed_answers)
    logging.debug(f"Seeds set: Questions={seed_questions}, Answers={seed_answers}")

def shuffle_questions_list(questions: List[PexamQuestion]) -> List[PexamQuestion]:
    """
    Shuffles the list of questions in-place (if a seed was provided) AND 
    renumbers them sequentially (1..N), preserving the original ID in `original_id`.
    
    If seed_questions was None in set_seeds(), the order is preserved, 
    but renumbering still happens.
    """
    if _rng_questions is not None:
        _rng_questions.shuffle(questions)
    
    # Renumber
    for i, q in enumerate(questions, 1):
        if q.original_id is None:
            q.original_id = q.id
        q.id = i
    
    return questions

def shuffle_options_for_question(question: PexamQuestion) -> None:
    """
    Shuffles the options of a single question in-place using the global answer seed.
    """
    if question.options:
        _rng_answers.shuffle(question.options)

def fuzzy_match_id(target_id: str, candidates: list[str], threshold: int = 80) -> str | None:
    """
    Finds the best match for target_id in candidates using Levenshtein distance.
    Threshold is 0-100 similarity score (100 is exact match).
    Returns the best matching candidate or None if no match meets the threshold.
    """
    best_match = None
    best_score = -1
    
    target_id_str = str(target_id)
    target_len = len(target_id_str)
    if target_len == 0:
        return None

    for candidate in candidates:
        if not candidate: continue
        candidate_str = str(candidate)
        
        # Calculate ratio (0-1) and convert to percentage (0-100)
        # Levenshtein.ratio returns similarity ratio
        score = Levenshtein.ratio(target_id_str, candidate_str) * 100.0
        
        if score >= threshold and score > best_score:
            best_score = score
            best_match = candidate
            
    return best_match

def load_solutions(exam_dir: str) -> Tuple[Dict[str, Dict[int, Any]], Dict[str, Dict[int, int]], int]:
    """
    Loads solutions from exam_model_*_questions.json files in the given directory.
    Returns:
        - solutions_per_model (full data for analysis)
        - solutions_per_model_simple (just correct indices for correction)
        - max_score (max possible score)
    """
    solutions_per_model = {}
    solutions_per_model_simple = {}
    max_score = 0
    
    solution_files = glob.glob(os.path.join(exam_dir, "exam_model_*_questions.json"))
    if not solution_files:
        logging.warning(f"No 'exam_model_..._questions.json' files found in {exam_dir}")
        return {}, {}, 0

    for sol_file in solution_files:
        try:
            model_id_match = re.search(r"exam_model_(\w+)_questions.json", os.path.basename(sol_file))
            if model_id_match:
                model_id = model_id_match.group(1)
                exam = PexamExam.model_validate_json(Path(sol_file).read_text(encoding="utf-8"))
                
                # Store full question data for analysis
                solutions_per_model[model_id] = {q.id: q.model_dump() for q in exam.questions}
                
                # Store only indices for the correction module
                solutions_simple = {q.id: q.correct_answer_index for q in exam.questions if q.correct_answer_index is not None}
                solutions_per_model_simple[model_id] = solutions_simple

                if len(solutions_simple) > max_score:
                    max_score = len(solutions_simple)
        except Exception as e:
            logging.error(f"Failed to load solution file {sol_file}: {e}")
            
    return solutions_per_model, solutions_per_model_simple, max_score
