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
    Loads questions from a file (JSON or MD), resolving bundled assets and image paths.
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
        
    # Resolve paths for images, making them absolute before passing them to the generator.
    file_dir = questions_path.parent
    for q in questions:
        if q.image_source and not Path(q.image_source).is_absolute():
            # First, try to resolve the path relative to the input file's directory.
            image_path_rel_file = (file_dir / q.image_source).resolve()
            
            # If that path doesn't exist, try resolving relative to the current working directory.
            image_path_rel_cwd = Path(q.image_source).resolve()
            
            # Also check relative to package assets if loading from sample
            try:
                package_dir = Path(pexams.__file__).parent
                image_path_rel_assets = (package_dir / "assets" / q.image_source).resolve()
            except:
                image_path_rel_assets = Path("nonexistent")

            if image_path_rel_file.exists():
                q.image_source = str(image_path_rel_file)
            elif image_path_rel_cwd.exists():
                q.image_source = str(image_path_rel_cwd)
            elif image_path_rel_assets.exists():
                q.image_source = str(image_path_rel_assets)
            else:
                logging.warning(
                    f"Could not find image for question {q.id} at '{q.image_source}'. "
                    f"Checked relative to input file, current directory, and assets."
                )
    return questions

