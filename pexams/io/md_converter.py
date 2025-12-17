import re
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

from pexams.schemas import PexamQuestion, PexamOption

def load_questions_from_md(path: str) -> List[PexamQuestion]:
    """
    Parses a Markdown file containing questions into a list of PexamQuestion objects.
    
    Format specification:
    ## question_id
    > ![Image for question](image.png)
    Question text...
    * Correct answer
    * Wrong answer 1
    * Wrong answer 2
    
    **Explanation:**
    Explanation text...
    """
    if not os.path.exists(path):
        logging.error(f"Questions markdown file not found: {path}")
        return []
        
    questions: List[PexamQuestion] = []
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content by the new question header '## ' that appears at the start of a line
    question_blocks = re.split(r'(?=^## )', content, flags=re.MULTILINE)

    md_dir = os.path.dirname(os.path.abspath(path))

    for block in question_blocks:
        block = block.strip()
        if not block.startswith("##"):
            continue

        lines = block.split('\n')
        
        # Parse ID from header: "## question_id"
        header_line = lines[0]
        match = re.match(r'^##\s+(.+)', header_line)
        if not match:
            continue
        
        question_id_str = match.group(1).strip()
        
        content_lines = lines[1:]
        
        # Check for and extract a quoted image line
        image_path = None
        if content_lines and content_lines[0].strip().startswith('>'):
            img_match = re.search(r'!\[.*\]\((.*)\)', content_lines[0])
            if img_match:
                raw_image_path = img_match.group(1).strip()
                # Resolve image path relative to MD file
                # If it's just a filename, it's in the same dir.
                # If it's a relative path, it's relative to MD dir.
                
                # Check if it exists
                abs_image_path = os.path.normpath(os.path.join(md_dir, raw_image_path))
                if os.path.exists(abs_image_path):
                    image_path = abs_image_path
                else:
                    logging.warning(f"Image not found at {abs_image_path} for question {question_id_str}")
                    # Keep the raw path if not found, maybe it works later or is a URL?
                    image_path = raw_image_path

                # Remove the image line from the content
                content_lines = content_lines[1:]

        first_answer_idx = -1
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            if stripped.startswith('* ') or stripped.startswith('- ') or re.match(r'^\d+\.\s', stripped):
                first_answer_idx = i
                break
        
        if first_answer_idx == -1:
            logging.warning(f"Could not find any answers for question ID '{question_id_str}'. Skipping.")
            continue
            
        question_text = "\n".join(content_lines[:first_answer_idx]).strip()
        
        answer_and_exp_lines = content_lines[first_answer_idx:]
        
        answers_text = []
        explanation_lines = []
        is_parsing_exp = False
        
        for line in answer_and_exp_lines:
            stripped = line.strip()
            if stripped.lower().startswith("**explanation:**"):
                is_parsing_exp = True
                continue
            
            if is_parsing_exp:
                explanation_lines.append(line)
                continue
            
            if stripped.startswith(('* ', '- ')):
                answers_text.append(re.sub(r'^[\*\-]\s*', '', stripped))
            elif re.match(r'^\d+\.\s', stripped):
                 answers_text.append(re.sub(r'^\d+\.\s*', '', stripped))

        if not answers_text:
            continue

        options = []
        # First answer is correct
        options.append(PexamOption(text=answers_text[0], is_correct=True))
        # Rest are distractors
        for dist in answers_text[1:]:
             options.append(PexamOption(text=dist, is_correct=False))

        explanation = "\n".join(explanation_lines).strip() or None
        
        questions.append(PexamQuestion(
            id=question_id_str,
            text=question_text,
            options=options,
            image_source=image_path,
            explanation=explanation
        ))
        
    return questions

def save_questions_to_md(questions: List[PexamQuestion], output_file: str):
    """
    Saves a list of PexamQuestion objects to a Markdown file following the specification.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in questions:
            # Header
            f.write(f"## {q.id}\n")
            
            # Image (in blockquote)
            if q.image_source:
                # Try to make path relative to output file if possible
                image_path = q.image_source
                try:
                    rel_path = os.path.relpath(q.image_source, os.path.dirname(os.path.abspath(output_file)))
                    if not rel_path.startswith(".."):
                         image_path = rel_path
                except ValueError:
                    pass # Keep absolute if on different drive
                
                f.write(f"> ![Image for question]({image_path})\n")
            
            # Question text
            f.write(f"{q.text}\n")
            
            # Options (First one is correct)
            # Find correct option
            correct_opt = next((o for o in q.options if o.is_correct), None)
            other_opts = [o for o in q.options if not o.is_correct]
            
            if correct_opt:
                f.write(f" * {correct_opt.text}\n")
            
            for opt in other_opts:
                f.write(f" * {opt.text}\n")
                
            # Explanation
            if q.explanation:
                f.write(f"\n**Explanation:**\n{q.explanation}\n")
            
            f.write("\n")
