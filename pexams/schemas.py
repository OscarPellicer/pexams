import os
from pathlib import Path
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, computed_field, model_validator

class PexamOption(BaseModel):
    """Data model for a single answer option in a question."""
    text: str
    is_correct: bool = Field(False, description="True if this is a correct answer.")


class PexamAnswerArea(BaseModel):
    """Layout hints for an open-answer response area."""
    lines: float = Field(8, ge=1, description="Approximate number of answer lines to reserve (may be fractional, e.g. 8.5).")
    height_mm: Optional[float] = Field(None, gt=0, description="Exact answer box height in millimeters.")
    show_lines: bool = Field(False, description="Draw horizontal writing guide lines inside the answer box.")

class PexamQuestion(BaseModel):
    """
    Data model for a single exam question.
    This schema is portable and can be used as the base for other systems.
    """
    id: Union[int, str]
    original_id: Optional[Union[int, str]] = Field(None, description="The original ID from the source file, preserved across shuffling.")
    question_type: Literal["multiple_choice", "open_answer"] = Field(
        "multiple_choice",
        description="Question family. Legacy questions without this field are treated as multiple choice.",
    )
    text: str
    points: float = Field(1.0, gt=0, description="Maximum points assigned to this question.")
    options: List[PexamOption] = Field(default_factory=list)
    image_source: Optional[str] = Field(None, description="Legacy: an image shown below the text. It is moved into the text as an ordinary Markdown image when the question is created; put images in the text instead.")
    max_image_width: Optional[str] = Field(None, description="Maximum width of every image in the question (e.g., '100px', '50%').")
    max_image_height: Optional[str] = Field(None, description="Maximum height of every image in the question (e.g., '100px', '50%').")
    font_size: Optional[str] = Field(None, description="Optional per-question font size, e.g. '9pt' or '13px'.")
    explanation: Optional[str] = Field(None, description="Explanation for the correct answer.")
    expected_answer: Optional[str] = Field(None, description="Reference answer for open-answer questions.")
    rubric: Optional[str] = Field(None, description="Correction rubric for open-answer questions.")
    answer_area: PexamAnswerArea = Field(default_factory=PexamAnswerArea)
    
    @model_validator(mode="after")
    def move_image_into_text(self):
        """Images are part of the text: a legacy ``image_source`` becomes a Markdown image below it.

        A relative path that exists from the working directory is made absolute; otherwise it is
        kept as written and resolved against the questions file's folder when rendering.
        """
        if self.image_source:
            src = self.image_source.replace("\\", "/")
            if "://" not in src and not src.startswith("data:") and (os.path.isabs(src) or os.path.isfile(src)):
                src = Path(os.path.abspath(src)).as_uri()  # a file URI: no spaces to break the Markdown
            self.text = f"{self.text.rstrip()}\n\n![]({src})" if self.text.strip() else f"![]({src})"
            self.image_source = None
        return self

    @model_validator(mode="after")
    def validate_question_shape(self):
        """Validate the fields required by each question family."""
        if self.question_type == "multiple_choice":
            correct_answers = sum(1 for option in self.options if option.is_correct)
            if correct_answers != 1:
                raise ValueError('Each multiple-choice question must have exactly one correct answer.')
        elif self.question_type == "open_answer":
            if any(option.is_correct for option in self.options):
                raise ValueError('Open-answer questions cannot define correct multiple-choice options.')
        return self
    
    @computed_field
    @property
    def correct_answer_index(self) -> Optional[int]:
        """Returns the index of the first correct answer, or None if no correct answer is set."""
        if self.question_type != "multiple_choice":
            return None
        for i, option in enumerate(self.options):
            if option.is_correct:
                return i
        return None

    @property
    def is_multiple_choice(self) -> bool:
        return self.question_type == "multiple_choice"

    @property
    def is_open_answer(self) -> bool:
        return self.question_type == "open_answer"

class PexamExam(BaseModel):
    """Data model for a full exam, containing a list of questions."""
    questions: List[PexamQuestion]
