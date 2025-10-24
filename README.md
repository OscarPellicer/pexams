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

**Common options:**
- `--num-models <int>`: Number of exam variations to generate (default: 4).
- `--exam-title <str>`: Title for the exam (default: "Final Exam").
- `--exam-course <str>`: Course name for the exam (optional).
- `--exam-date <str>`: Date of the exam (optional).
- `--columns <int>`: Number of question columns (1, 2, or 3; default: 1).
- `--font-size <str>`: Base font size, e.g., '10pt' (default: '11pt').
- `--generate-fakes <int>`: Number of simulated scans to generate for testing.
- `--log-level DEBUG`: Keeps the intermediate HTML files for debugging.

#### `pexams correct`
Corrects scanned exams and runs an analysis.

```bash
pexams correct --input-path <path> --exam-dir <path> --output-dir <path> [OPTIONS]
```
- The `--input-path` can be a single PDF file or a folder of images (PNG, JPG).
- The `--exam-dir` must contain the `exam_model_*_questions.json` files generated alongside the exam PDFs.

**Common options:**
- `--void-questions <str>`: Comma-separated list of question IDs to exclude from scoring (e.g., '3,4').
