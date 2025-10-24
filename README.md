# Pexams: Python exam generation and correction

Pexams is a library for generating beautiful multiple-choice exam sheets from simple data structures and automatically correcting them from scans using computer vision. It is similar to R/exams, but written in Python and using [Playwright](https://playwright.dev/python/) for high-fidelity PDF generation instead of LaTeX. It has the following advantages:
- It is much easier to install, as it requires only Python (no R, LaTeX, or any other external dependencies).
- It is easier to customize (it's Python + HTML/CSS!).
- It is less prone to compilation errors (again, no LaTeX!).

NOTE: This library is still in development and is not yet ready for production use. Although everything should work, there may be some bugs, missing features, or breaking changes in future versions.

## Installation

The library has been tested on Python 3.11.

<!-- may not be needed
### Prerequisites

- **Poppler**: Needed for `pdf2image` to convert PDFs to images during correction.
  - **Windows**: `conda install -c conda-forge poppler`
  - **macOS**: `brew install poppler`
  - **Debian/Ubuntu**: `sudo apt-get install poppler-utils` 
  - -->

### 1. Install the library

You can install the library from PyPI:
```bash
pip install pexams
```

If the previous command failed, you can install the library from GitHub:
```bash
pip install git+https://github.com/OscarPellicer/pexams.git
```

Alternatively, you can clone the repository and install it in editable mode, which is useful for development, testing and pushing changes to the repository:

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

## Quick start

This example will guide you through generating an exam with 2 different models, creating 4 simulated student scans, and then correcting them.

After installation, the `pexams` command is available globally in your environment. For the following example, we assume you are running the commands from the root of the project where `sample_test.json` is located.

### 1. Generate the exams

Run the following command:

```bash
pexams generate --questions-json sample_test.json --output-dir ./exam_output --num-models 2 --generate-fakes 4 --columns 2 --exam-title "Sample Exam" --exam-course "Everything 101" --exam-date "2025-10-26"
```

This will create an `exam_output` directory containing:
- PDF files for 2 exam models.
- JSON files for each model containing the question data and correct answers.
- A `simulated_scans` subdirectory with 4 sample PNGs of filled answer sheets.

### 2. Correct the exams

Now, let's correct the simulated scans:

```bash
pexams correct --input-path ./exam_output/simulated_scans/ --exam-dir ./exam_output/ --output-dir ./correction_results
```

This will create a `correction_results` directory with a CSV report and annotated images of each corrected scan.

## Visual examples

You can view an example of a fully generated exam PDF [here](media/example_model_1.pdf).

Below is an example of a simulated answer sheet and the annotated, corrected version that the library produces.

| Simulated Scan | Corrected Scan |
| :---: | :---: |
| <img src="media/simulated.png" width="400"> | <img src="media/corrected.png" width="400"> |

The analysis module also generates a plot showing the distribution of answers for each question, which helps in identifying problematic questions, as well as a plot showing the distribution of marks, which helps in assessing the fairness of the exam.

| Answer distribution | Marks distribution |
| :---: | :---: |
| <img src="media/answer_distribution.png" width="400"> | <img src="media/mark_distribution.png" width="400"> |

## Usage

### Input format (JSON)

The `generate` command expects a JSON file containing the exam questions. The JSON file must conform to the following schema:

- The root object should have a single key, `questions`, which is an array of question objects.
- Each question object has the following keys:
  - `id` (integer, required): A unique identifier for the question.
  - `text` (string, required): The question text. You can use Markdown here. For LaTeX, enclose it in `$...$`.
  - `options` (array, required): A list of option objects.
    - Each option object has:
      - `text` (string, required): The option text.
      - `is_correct` (boolean, required): Must be `true` for exactly one option per question.
  - `image_source` (string, optional): A path to a local image or a URL.

**Example `questions.json`:**
```json
{
  "questions": [
    {
      "id": 1,
      "text": "What is the capital of France?",
      "options": [
        { "text": "Berlin", "is_correct": false },
        { "text": "Madrid", "is_correct": false },
        { "text": "Paris", "is_correct": true },
        { "text": "Rome", "is_correct": false }
      ]
    }
  ]
}
```

### Command line

#### Generating exams

```bash
pexams generate --questions-json <path_to_questions.json> --output-dir <results_directory> [OPTIONS]
```

**Common options:**
- `--num-models <int>`: Number of exam variations to generate (default: 4).
- `--exam-title <str>`: Title for the exam (default: "Final Exam").
- `--exam-course <str>`: Course name for the exam (optional).
- `--exam-date <str>`: Date of the exam (optional).
- `--font-size <str>`: Base font size, e.g., '10pt' (default: '11pt').
- `--columns <int>`: Number of question columns (1, 2, or 3; default: 1).
- `--generate-fakes <int>`: Number of simulated scans to generate for testing.
- `--generate-references`: Generates a reference scan with correct answers for each model.

#### Correcting exams

```bash
pexams correct --input-path <path_to_scans> --exam-dir <path_to_exam_models> --output-dir <results_directory>
```
- The `--input-path` can be a single PDF file or a folder of images (PNG, JPG).
- The `--exam-dir` must contain the `exam_model_*_questions.json` files generated alongside the exam PDFs.

### Python API

You can also programatically use the library from Python to generate and correct exams.

#### Generating exams

```python
from pexams import generate_exams
from pexams.schemas import PexamQuestion, PexamOption

# 1. Create your list of questions
questions = [
    PexamQuestion(
        id=1,
        text="What is the capital of France?",
        options=[
            PexamOption(text="Berlin", is_correct=False),
            PexamOption(text="Madrid", is_correct=False),
            PexamOption(text="Paris", is_correct=True),
            PexamOption(text="Rome", is_correct=False),
        ]
    ),
    # ... more questions
]

# 2. Generate the exam PDFs
generate_exams(
    questions=questions,
    output_dir="my_exams",
    num_models=4,
    exam_title="Geography Quiz",
    exam_course="GEO101",
    lang="en"
)
```

#### Correcting exams

```python
from pexams import correct_exams

# In a real scenario, you would load the solutions that were 
# generated by the `generate_exams` function.
solutions_per_model = {
    "1": { # model_id
        1: 2,  # Question 1, correct option is index 2 ('C')
        2: 0,
        # ... more solutions for model 1
    },
    "2": {
        1: 0,
        2: 3,
        # ... more solutions for model 2
    }
}

# Correct the scanned PDF or image folder
correct_exams(
    input_path="scans/all_scans.pdf",
    solutions_per_model=solutions_per_model,
    output_dir="results"
)
```
