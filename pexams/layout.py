# All dimensions are in millimeters (mm)
from typing import Dict, Tuple, NamedTuple, List
from pexams.schemas import PexamQuestion

class BoxCoordinates(NamedTuple):
    """Stores the coordinates for a rectangular box."""
    top_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    center: Tuple[float, float]

class AnswerSheetLayout(NamedTuple):
    """Stores all calculated coordinates for an answer sheet."""
    exam_title: Tuple[float, float]
    model_id_box: BoxCoordinates
    exam_info: Tuple[float, float]
    student_name_label: Tuple[float, float]
    student_name_box: BoxCoordinates
    student_id_label: Tuple[float, float]
    student_id_box: BoxCoordinates
    student_signature_label: Tuple[float, float]
    student_signature_box: BoxCoordinates
    instructions: Tuple[float, float]
    question_numbers: Dict[int, Tuple[float, float]]
    header_labels: Dict[int, Dict[int, Tuple[float, float]]] # group_index -> label_index -> (x, y)
    answer_boxes: Dict[int, Dict[int, BoxCoordinates]] # question_id -> option_index -> BoxCoordinates

# --- Page and Printable Area ---
PRINTABLE_WIDTH = 180
PRINTABLE_HEIGHT = 267

# --- Header Element Positions (Top-left corner) ---
EXAM_TITLE_POS = (5, 5)
EXAM_INFO_POS = (5, 20)
MODEL_ID_LABEL_POS = (5, 32)
MODEL_ID_BOX_TL = (5, 32)
MODEL_ID_BOX_WIDTH = 4.5
MODEL_ID_BOX_HEIGHT = 6.0


STUDENT_ID_LABEL_POS = (5, 40)
STUDENT_ID_BOX_START = (5, 47)

ID_BOX_WIDTH = 52
ID_BOX_HEIGHT = 9
ID_BOX_SPACING = 1.5

STUDENT_NAME_LABEL_POS = (5, 60)
STUDENT_NAME_BOX_TL = (5, 67)
STUDENT_NAME_BOX_WIDTH = 75
STUDENT_NAME_BOX_HEIGHT = 12

STUDENT_SIGNATURE_LABEL_POS = (90, 60)
STUDENT_SIGNATURE_BOX_TL = (90, 67)
STUDENT_SIGNATURE_BOX_WIDTH = 75
STUDENT_SIGNATURE_BOX_HEIGHT = 12
INSTRUCTIONS_POS = (90, 20)

# --- Answer Block Layout ---
ANSWERS_START_X = 5
ANSWERS_START_Y = 85
NUM_COLUMNS = 3
QUESTIONS_PER_COLUMN = 25
QUESTIONS_PER_GROUP = 5

# Spacing
GROUP_VERTICAL_SPACING = 4
COLUMN_HORIZONTAL_SPACING = 15

# --- Individual Box Dimensions and Spacing ---
OPTION_BOX_WIDTH = 5.0
OPTION_BOX_HEIGHT = 2.0
OPTION_BOX_HORIZONTAL_SPACING = 2.0
QUESTION_NUMBER_WIDTH = 10.0
HEADER_OPTION_LABEL_WIDTH = OPTION_BOX_WIDTH
ANSWER_ROW_HEIGHT = OPTION_BOX_HEIGHT
ANSWER_ROW_VERTICAL_SPACING = 3
HEADER_ROW_HEIGHT = 4.0
HEADER_ROW_BOTTOM_MARGIN = 2.0

# --- Calculated dimensions ---
BUBBLE_STEP_X = OPTION_BOX_WIDTH + OPTION_BOX_HORIZONTAL_SPACING
BUBBLE_STEP_Y = ANSWER_ROW_HEIGHT + ANSWER_ROW_VERTICAL_SPACING


def get_answer_sheet_layout(questions: List[PexamQuestion]) -> AnswerSheetLayout:
    """Calculates the absolute mm coordinates for every element on the answer sheet."""
    
    num_questions = len(questions)
    max_options = max(len(q.options) for q in questions) if questions else 0

    group_content_width = (QUESTION_NUMBER_WIDTH + (max_options * OPTION_BOX_WIDTH) + ((max_options - 1) * OPTION_BOX_HORIZONTAL_SPACING))
    group_content_height = (HEADER_ROW_HEIGHT + HEADER_ROW_BOTTOM_MARGIN + (QUESTIONS_PER_GROUP * ANSWER_ROW_HEIGHT) + ((QUESTIONS_PER_GROUP - 1) * ANSWER_ROW_VERTICAL_SPACING))
    first_bubble_offset_x = QUESTION_NUMBER_WIDTH
    first_bubble_offset_y = HEADER_ROW_HEIGHT + HEADER_ROW_BOTTOM_MARGIN

    # --- Header and Student Info ---
    model_id_box = BoxCoordinates(
        top_left=MODEL_ID_BOX_TL,
        bottom_right=(MODEL_ID_BOX_TL[0] + MODEL_ID_BOX_WIDTH, MODEL_ID_BOX_TL[1] + MODEL_ID_BOX_HEIGHT),
        center=(MODEL_ID_BOX_TL[0] + MODEL_ID_BOX_WIDTH / 2, MODEL_ID_BOX_TL[1] + MODEL_ID_BOX_HEIGHT / 2)
    )

    student_name_box = BoxCoordinates(
        top_left=STUDENT_NAME_BOX_TL,
        bottom_right=(STUDENT_NAME_BOX_TL[0] + STUDENT_NAME_BOX_WIDTH, STUDENT_NAME_BOX_TL[1] + STUDENT_NAME_BOX_HEIGHT),
        center=(STUDENT_NAME_BOX_TL[0] + STUDENT_NAME_BOX_WIDTH / 2, STUDENT_NAME_BOX_TL[1] + STUDENT_NAME_BOX_HEIGHT / 2)
    )
    
    student_id_box = BoxCoordinates(
        top_left=STUDENT_ID_BOX_START,
        bottom_right=(STUDENT_ID_BOX_START[0] + ID_BOX_WIDTH, STUDENT_ID_BOX_START[1] + ID_BOX_HEIGHT),
        center=(STUDENT_ID_BOX_START[0] + ID_BOX_WIDTH / 2, STUDENT_ID_BOX_START[1] + ID_BOX_HEIGHT / 2)
    )

    student_signature_box = BoxCoordinates(
        top_left=STUDENT_SIGNATURE_BOX_TL,
        bottom_right=(STUDENT_SIGNATURE_BOX_TL[0] + STUDENT_SIGNATURE_BOX_WIDTH, STUDENT_SIGNATURE_BOX_TL[1] + STUDENT_SIGNATURE_BOX_HEIGHT),
        center=(STUDENT_SIGNATURE_BOX_TL[0] + STUDENT_SIGNATURE_BOX_WIDTH / 2, STUDENT_SIGNATURE_BOX_TL[1] + STUDENT_SIGNATURE_BOX_HEIGHT / 2)
    )

    # --- Answer Boxes and Labels ---
    question_numbers, header_labels, answer_boxes = {}, {}, {}
    groups_per_column = QUESTIONS_PER_COLUMN // QUESTIONS_PER_GROUP
    num_groups = (num_questions + QUESTIONS_PER_GROUP - 1) // QUESTIONS_PER_GROUP
    
    for group_index in range(num_groups):
        if group_index >= (NUM_COLUMNS * groups_per_column):
            break

        col = group_index // groups_per_column
        row_in_col = group_index % groups_per_column

        group_left_mm = ANSWERS_START_X + col * (group_content_width + COLUMN_HORIZONTAL_SPACING)
        group_top_mm = ANSWERS_START_Y + row_in_col * (group_content_height + GROUP_VERTICAL_SPACING)

        header_labels[group_index] = {}
        for i in range(max_options):
            label_x = group_left_mm + QUESTION_NUMBER_WIDTH + i * BUBBLE_STEP_X
            label_y = group_top_mm
            header_labels[group_index][i] = (label_x, label_y)

        for index_in_group in range(QUESTIONS_PER_GROUP):
            q_num_zero_based = group_index * QUESTIONS_PER_GROUP + index_in_group
            if q_num_zero_based >= num_questions:
                break
            
            question_id = q_num_zero_based + 1
            answer_boxes[question_id] = {}

            q_num_y = group_top_mm + first_bubble_offset_y + index_in_group * BUBBLE_STEP_Y
            question_numbers[question_id] = (group_left_mm, q_num_y)
            
            for opt_idx in range(max_options):
                tl_x = group_left_mm + first_bubble_offset_x + opt_idx * BUBBLE_STEP_X
                tl_y = q_num_y
                answer_boxes[question_id][opt_idx] = BoxCoordinates(
                    top_left=(tl_x, tl_y),
                    bottom_right=(tl_x + OPTION_BOX_WIDTH, tl_y + OPTION_BOX_HEIGHT),
                    center=(tl_x + OPTION_BOX_WIDTH / 2, tl_y + OPTION_BOX_HEIGHT / 2)
                )

    return AnswerSheetLayout(
        exam_title=EXAM_TITLE_POS,
        model_id_box=model_id_box,
        exam_info=EXAM_INFO_POS,
        student_name_label=STUDENT_NAME_LABEL_POS,
        student_name_box=student_name_box,
        student_id_label=STUDENT_ID_LABEL_POS,
        student_id_box=student_id_box,
        student_signature_label=STUDENT_SIGNATURE_LABEL_POS,
        student_signature_box=student_signature_box,
        instructions=INSTRUCTIONS_POS,
        question_numbers=question_numbers,
        header_labels=header_labels,
        answer_boxes=answer_boxes
    )
