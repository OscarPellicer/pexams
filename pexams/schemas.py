from typing import List, Optional
from pydantic import BaseModel, Field, validator

class PexamOption(BaseModel):
    """Data model for a single answer option in a question."""
    text: str
    is_correct: bool = Field(False, description="True if this is a correct answer.")

class PexamQuestion(BaseModel):
    """
    Data model for a single exam question.
    This schema is portable and can be used as the base for other systems.
    """
    id: int
    text: str
    options: List[PexamOption]
    image_source: Optional[str] = Field(None, description="Source for an image, can be a local path, a URL, or a base64 encoded string.")
    
    @validator('options')
    def check_one_correct_answer(cls, v):
        """Ensures that exactly one option is marked as correct."""
        correct_answers = sum(1 for option in v if option.is_correct)
        if correct_answers != 1:
            raise ValueError('Each question must have exactly one correct answer.')
        return v
    
    @property
    def correct_answer_index(self) -> Optional[int]:
        """Returns the index of the first correct answer, or None if no correct answer is set."""
        for i, option in enumerate(self.options):
            if option.is_correct:
                return i
        return None

class PexamExam(BaseModel):
    """Data model for a full exam, containing a list of questions."""
    questions: List[PexamQuestion]
