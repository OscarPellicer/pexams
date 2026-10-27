import logging
import json
from pathlib import Path
from typing import Optional, List
from pexams.schemas import PexamQuestion, PexamExam
from pexams.io import md_converter
from pydantic import ValidationError
import pexams

def load_and_prepare_questions(questions_path_str: str) -> Optional[List[PexamQuestion]]:
    """
    Loads questions from a file (JSON or MD), also from the bundled assets.
    """
    questions_path = Path(questions_path_str)

    # Check if the file exists at the given path. If not, try to find it in the package assets.
    if not questions_path.exists():
        try:
            package_dir = Path(pexams.__file__).parent
            asset_path = package_dir / "assets" / questions_path_str
            if asset_path.exists():
                questions_path = asset_path
            else:
                raise FileNotFoundError
        except (FileNotFoundError, AttributeError):
            logging.error(f"Questions file not found at '{questions_path_str}' or as a built-in asset.")
            return None

    questions = None
    
    # Determine format by extension
    ext = questions_path.suffix.lower()
    if ext == '.md':
        logging.info(f"Loading questions from Markdown file: {questions_path}")
        questions = md_converter.load_questions_from_md(str(questions_path))
    elif ext == '.json':
        logging.info(f"Loading questions from JSON file: {questions_path}")
        try:
            exam = PexamExam.model_validate_json(questions_path.read_text(encoding="utf-8"))
            questions = exam.questions
        except ValidationError as e:
            logging.error(f"Failed to validate questions JSON file: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse questions JSON file: {e}")
            return None
    else:
        # Try Markdown first as default
        logging.info(f"No known extension '{ext}'. Trying to load as Markdown...")
        questions = md_converter.load_questions_from_md(str(questions_path))
        if not questions:
             pass

    if not questions:
        logging.error("No questions loaded.")
        return None
        
    return questions

