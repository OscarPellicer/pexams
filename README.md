# Pexams: Python Exam Generation and Correction

Pexams is a library for generating beautiful multiple-choice exam sheets and automatically correcting them from scans using computer vision. It is similar to R/exams, but written in Python and using [Playwright](https://playwright.dev/python/) for high-fidelity PDF generation instead of LaTeX. It has the following advantages: it has more features, is faster, is easier to install, easier to customize, and it is much less prone to compilation errors than R/exams.

NOTE: This library is still in development and is not yet ready for production use. Although everything should work, there may be some bugs, missing features, or breaking changes in future versions.

## Visual examples

You can view an example of a fully generated exam PDF [here](https://github.com/OscarPellicer/pexams/blob/main/media/example_model_1.pdf).

Below is an example of a simulated answer sheet and the annotated, corrected version that the library produces.

| Simulated Scan | Corrected Scan |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/OscarPellicer/pexams/main/media/simulated.png" width="400"> | <img src="https://raw.githubusercontent.com/OscarPellicer/pexams/main/media/corrected.png" width="400"> |

The analysis module also generates a plot showing the distribution of answers for each question, which helps in identifying problematic questions, as well as a plot showing the distribution of marks, which helps in assessing the fairness of the exam.

| Answer distribution | Marks distribution |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/OscarPellicer/pexams/main/media/answer_distribution.png" width="400"> | <img src="https://raw.githubusercontent.com/OscarPellicer/pexams/main/media/mark_distribution.png" width="400"> |

## Features

### Exam generation
- **Multiple exam models**: Generate multiple unique exam models from a single JSON source file, with automatic shuffling of questions and answers.
- **Rich content support**: Write questions in Markdown and include:
  - **LaTeX equations**: Seamlessly render math formulas using MathJax (`$...$`).
  - **Images**: Embed images in your questions from local files.
  - **Code snippets**: Include code snippets (\`...\`).
- **Customizable layout**:
  - Arrange questions in one, two, or three columns.
  - Adjust the base font size to fit your needs.
- **Customizable answer sheet**:
  - Set the length of the student ID field.
  - Internationalization support for labels (more than 20 languages supported).
- **High-fidelity PDFs**: Uses Playwright to produce clean, modern, and reliable PDF documents from HTML/CSS.

### Correction & analysis
- **Automated correction**: Correct exams from a single PDF containing all scans or from a folder of individual images.
- **Robust image processing**: Uses `OpenCV` with fiducial markers for reliable, automatic perspective correction and alignment, the `TrOCR` vision transformer model for OCR of the student ID, name, and model ID, and custom position detection for the answers.
- **Detailed reports**: Generates a `correction_results.csv` file with detailed scores and answers for each student.
- **Insightful visualizations**: Automatically produces plots for:
  - **Mark distribution**: A histogram to assess overall student performance.
  - **Answer distribution**: A bar plot to analyze performance on each question and identify potential issues.
- **Flexible scoring**: Easily void specific questions during the analysis if needed, either by removing it from the score calculation completely or by voiding it "nicely" (can only increase the score if the question is correct, otherwise the question is removed from the score calculation).

### Development & testing
- **Simulated scans**: Automatically generate a set of fake, filled-in answer sheets to test the full correction and analysis pipeline.
- **End-to-end testing**: A simple `pexams test` command runs a full generate-correct-analyze cycle using bundled sample data.
- **Easy debugging**: Keep the intermediate HTML files to inspect the exam content and layout before PDF conversion, by setting the `--log-level DEBUG` flag.

## Installation

The library has been tested on Python 3.11.

### 1. Install the library

You can install the library from PyPI:
```bash
pip install pexams
```

Alternatively, you can clone the repository and install it in editable mode, which is useful for development:

```bash
git clone https://github.com/OscarPellicer/pexams.git
cd pexams
pip install -e .
```

### 2. Install Playwright browsers

`pexams` uses Playwright to convert HTML to PDF. You need to download the necessary browser binaries by running:
```bash
playwright install chromium
```
This command only needs to be run once.

### 3. Install Poppler

You may also need to install Poppler, which is needed for `pdf2image` to convert PDFs to images during correction, and also for generating simulated scans:

  - **Windows**: `conda install -c conda-forge poppler`
  - **macOS**: `brew install poppler`
  - **Debian/Ubuntu**: `sudo apt-get install poppler-utils`

## Quick start

The `pexams test` command provides a simple way to run a full cycle and see the library in action. It uses a bundled sample `json` file and media to generate, correct, and analyze a sample exam.

```bash
pexams test --output-dir ./my_test_output
```

This will create a `my_test_output` directory containing the generated exams, simulated scans, correction results, and analysis plots.

## Usage

### 1. The questions JSON file

The `generate` command expects a JSON file containing the exam questions.

- The root object should have a single key, `questions`, which is an array of question objects.
- Each question object has the following keys:
  - `id` (integer, required): A unique identifier for the question.
  - `text` (string, required): The question text. You can use Markdown, code blocks, and LaTeX (`$...$`).
  - `options` (array, required): A list of option objects.
    - Each option object has `text` (string) and `is_correct` (boolean). Exactly one option must be correct.
  - `image_source` (string, optional): A path to an image file. The path can be relative to the JSON file's location or to the current working directory.

**Example `questions.json`:**
```json
{
  "questions": [
    {
      "id": 1,
      "text": "What is the value of the integral $\\int_0^\\infty e^{-x^2} dx$?",
      "options": [
        { "text": "$\\sqrt{\\pi}$", "is_correct": false },
        { "text": "$\\frac{\\sqrt{\\pi}}{2}$", "is_correct": true },
        { "text": "$\\pi$", "is_correct": false }
      ]
    }
  ]
}
```

### 2. CLI commands

#### `pexams generate`
Generates exam PDFs and solution files from a questions JSON file.

```bash
pexams generate --questions-json <path> --output-dir <path> [OPTIONS]
```

**Arguments:**
- `--questions-json <path>`: (Required) Path to the JSON file containing the exam questions.
- `--output-dir <path>`: (Required) Directory to save the generated exam PDFs and solution files.
- `--num-models <int>`: Number of different exam models to generate (default: 4).
- `--exam-title <str>`: Title of the exam (default: "Final Exam").
- `--exam-course <str>`: Course name for the exam (optional).
- `--exam-date <str>`: Date of the exam (optional).
- `--columns <int>`: Number of columns for the questions (1, 2, or 3; default: 1).
- `--font-size <str>`: Base font size for the exam (e.g., '10pt', '12px'; default: '11pt').
- `--id-length <int>`: Number of boxes for the student ID (default: 10).
- `--lang <str>`: Language for the answer sheet labels (e.g., 'en', 'es'; default: 'en').
- `--keep-html`: If set, keeps the intermediate HTML files used for PDF generation.
- `--generate-fakes <int>`: Generates a number of simulated scans with fake answers for testing the correction process (default: 0).
- `--generate-references`: If set, generates a reference scan with the correct answers marked for each model.
- `--log-level <level>`: Set the logging level (DEBUG, INFO, WARNING, ERROR; default: INFO).

#### `pexams correct`
Corrects scanned exams and runs an analysis.

```bash
pexams correct --input-path <path> --exam-dir <path> --output-dir <path> [OPTIONS]
```
- The `--input-path` can be a single PDF file or a folder of images (PNG, JPG).
- The `--exam-dir` must contain the `exam_model_*_questions.json` files generated alongside the exam PDFs.

**Arguments:**
- `--input-path <path>`: (Required) Path to the single PDF file or a folder containing scanned answer sheets.
- `--exam-dir <path>`: (Required) Path to the directory containing the `exam_model_*_questions.json` solution files.
- `--output-dir <path>`: (Required) Directory to save the correction results CSV and any debug images.
- `--void-questions <str>`: Comma-separated list of question IDs to exclude from scoring for all students (e.g., '3,4').
- `--void-questions-nicely <str>`: Comma-separated list of question IDs to void "nicely". If a student answered it correctly, the question counts. Otherwise, it is removed from the maximum possible score for that student.
- `--log-level <level>`: Set the logging level (DEBUG, INFO, WARNING, ERROR; default: INFO).

**Important Note on Voiding Questions:** The question IDs to be used with the `--void-questions` and `--void-questions-nicely` options refer to the question numbers as they appear on the final generated exam PDFs (e.g., 1, 2, 3...). These may be different from the original `id` fields in your source JSON file, as `pexams` shuffles and re-numbers the questions for each exam generation session.

#### `pexams test`
Runs a full generate/correct/analyze cycle using bundled sample data.

```bash
pexams test [OPTIONS]
```

**Arguments:**
- `--output-dir <path>`: Directory to save the test output (default: `./pexams_test_output`).

## Python API Usage

In addition to the CLI, you can use `pexams` as a Python library.

### 1. Generating Exams

To generate exams, use the `pexams.generate_exams.generate_exams` function.

```python
from pexams import generate_exams
from pexams.schemas import PexamQuestion, PexamExam
from pexams.main import _load_and_prepare_questions

# You can load questions from a JSON file
# questions = _load_and_prepare_questions("path/to/your/questions.json")

# Or define them manually
questions = [
    PexamQuestion(
        id=1,
        text="What is 2+2?",
        options=[
            {"text": "3", "is_correct": False},
            {"text": "4", "is_correct": True},
        ]
    )
]

if questions:
    generate_exams.generate_exams(
        questions=questions,
        output_dir="./my_exams",
        num_models=2,
        exam_title="Quiz 1"
    )
```

### 2. Correcting Exams

To correct exams, you first need to load the solutions that were generated, then call `pexams.correct_exams.correct_exams`.

```python
import os
import glob
import re
from pathlib import Path
from pexams import correct_exams
from pexams.schemas import PexamExam

exam_dir = "./my_exams" # The output from generate_exams
solutions_per_model = {}
solution_files = glob.glob(os.path.join(exam_dir, "exam_model_*_questions.json"))
for sol_file in solution_files:
    model_id_match = re.search(r"exam_model_(\w+)_questions.json", os.path.basename(sol_file))
    if model_id_match:
        model_id = model_id_match.group(1)
        exam = PexamExam.model_validate_json(Path(sol_file).read_text(encoding="utf-8"))
        solutions_for_correction = {q.id: q.correct_answer_index for q in exam.questions if q.correct_answer_index is not None}
        solutions_per_model[model_id] = solutions_for_correction

correct_exams.correct_exams(
    input_path="./my_exams/simulated_scans", # Path to PDF or folder of images
    solutions_per_model=solutions_per_model,
    output_dir="./correction_output"
)
```

### 3. Analyzing Results

After correction, you can run the analysis using `pexams.analysis.analyze_results`.

```python
from pexams import analysis

# You need the full solution details for the analysis plots.
# See pexams/main.py for a complete example of how to load this.
solutions_per_model_for_analysis = {}
max_score = 0
exam_dir = "./my_exams" 
try:
    solution_files = glob.glob(os.path.join(exam_dir, "exam_model_*_questions.json"))
    for sol_file in solution_files:
        model_id_match = re.search(r"exam_model_(\w+)_questions.json", os.path.basename(sol_file))
        if model_id_match:
            model_id = model_id_match.group(1)
            exam = PexamExam.model_validate_json(Path(sol_file).read_text(encoding="utf-8"))
            
            # Store full question data for analysis plots
            solutions_per_model_for_analysis[model_id] = {q.id: q.model_dump() for q in exam.questions}
            
            # Determine the max score from the solutions
            solutions_for_correction = {q.id: q.correct_answer_index for q in exam.questions if q.correct_answer_index is not None}
            if len(solutions_for_correction) > max_score:
                max_score = len(solutions_for_correction)
except Exception as e:
    print(f"Failed to load solutions for analysis: {e}")


analysis.analyze_results(
    csv_filepath="./correction_output/correction_results.csv",
    max_score=max_score,
    solutions_per_model=solutions_per_model_for_analysis,
    output_dir="./correction_output",
    void_questions_str="3", # Optional: void question 3 for all students
    void_questions_nicely_str="4" # Optional: void question 4 "nicely"
)
```


## Contributing

Pull requests are welcome! Please feel free to submit an issue or pull request.

## Contact

`oscar.pellicer at uv dot es`

## TODO

- Test that the correction resulst are actually correct when running the test command.
- Improve OCR speed by either concatenating the ID images into a single image, or batching them.
- Allow to pass a list of student IDs to the correction command, so that wrongly OCRd IDs can be matched to the correct student ID by using Levenshtein distance.
