from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator, computed_field

class PexamOption(BaseModel):
    """Data model for a single answer option in a question."""
    text: str
    is_correct: bool = Field(False, description="True if this is a correct answer.")

class PexamQuestion(BaseModel):
    """
    Data model for a single exam question.
    This schema is portable and can be used as the base for other systems.
    """
    id: Union[int, str]
    original_id: Optional[Union[int, str]] = Field(None, description="The original ID from the source file, preserved across shuffling.")
    text: str
    options: List[PexamOption]
    image_source: Optional[str] = Field(None, description="Source for an image, can be a local path, a URL, or a base64 encoded string.")
    max_image_width: Optional[str] = Field(None, description="Maximum width for the image (e.g., '100px', '50%').")
    max_image_height: Optional[str] = Field(None, description="Maximum height for the image (e.g., '100px', '50%').")
    explanation: Optional[str] = Field(None, description="Explanation for the correct answer.")
    
    @validator('options')
    def check_one_correct_answer(cls, v):
        """Ensures that exactly one option is marked as correct."""
        correct_answers = sum(1 for option in v if option.is_correct)
        if correct_answers != 1:
            raise ValueError('Each question must have exactly one correct answer.')
        return v
    
    @computed_field
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
