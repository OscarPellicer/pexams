import os
import logging
import pandas as pd
import Levenshtein
import csv
from tabulate import tabulate
from pexams import utils

def fill_marks_in_file(input_file: str, id_col: str, mark_col: str, correction_results_csv: str, 
                       fuzzy_threshold: int = 100, encoding: str = 'utf-8', sep: str = ',', 
                       decimal_sep: str = '.', name_col: str = None, simplify_csv: bool = False):
    """Fills marks into the input file (CSV/XLSX) based on correction results."""
    if not pd:
        logging.error("Pandas is required to fill marks in input files. Please install pandas.")
        return

    if not os.path.exists(correction_results_csv):
        logging.error(f"Correction results file not found: {correction_results_csv}")
        return

    try:
        # Load input file
        file_ext = os.path.splitext(input_file)[1].lower()
        used_encoding = encoding
        used_sep = sep
        
        if file_ext == '.csv':
            df_input = pd.read_csv(input_file, encoding=encoding, sep=sep)
        elif file_ext in ['.xlsx', '.xls']:
            df_input = pd.read_excel(input_file)
        elif file_ext == '.tsv':
            df_input = pd.read_csv(input_file, sep='\t', encoding=encoding)
        else:
            logging.error(f"Unsupported input file format: {file_ext}")
            return
            
        # Load correction results (final_marks.csv or correction_results.csv)
        # We prefer final_marks.csv which has the scaled mark
        final_marks_path = os.path.join(os.path.dirname(correction_results_csv), "final_marks.csv")
        if os.path.exists(final_marks_path):
             df_marks = pd.read_csv(final_marks_path)
             mark_source_col = 'mark'
        else:
             logging.warning(f"final_marks.csv not found, using raw scores from {correction_results_csv}")
             df_marks = pd.read_csv(correction_results_csv)
             mark_source_col = 'score' # Or whatever column holds the value we want
        
        # Ensure ID columns are string
        if id_col not in df_input.columns:
            logging.error(f"ID column '{id_col}' not found in input file.")
            return

        # Ensure mark column in marks df is numeric
        df_marks[mark_source_col] = pd.to_numeric(df_marks[mark_source_col], errors='coerce')
            
        df_input[id_col] = df_input[id_col].astype(str).str.strip().str.upper()
        df_marks['student_id'] = df_marks['student_id'].astype(str).str.strip().str.upper()
        
        # Create a mapping from OCR ID to Mark
        ocr_id_to_mark = dict(zip(df_marks['student_id'], df_marks[mark_source_col]))
        ocr_ids = list(ocr_id_to_mark.keys())
        
        # Prepare the mark column in input df
        # Always initialize/clear the mark column to ensure fresh results
        df_input[mark_col] = None
            
        matched_count = 0
        matched_ocr_ids = set()
        matched_rows_indices = set()
        
        # Store mapping to update final_marks.csv later
        ocr_to_real_mapping = {}

        # 1. Exact Matching
        
        exact_matches = []
        for idx, row in df_input.iterrows():
            target_id = row[id_col]
            if pd.isna(target_id) or not target_id:
                continue
                
            if target_id in ocr_id_to_mark:
                mark_val = ocr_id_to_mark[target_id]
                # Round to 2 decimal places
                if isinstance(mark_val, (int, float)):
                    mark_val = round(mark_val, 2)
                    
                df_input.at[idx, mark_col] = mark_val
                matched_ocr_ids.add(target_id)
                matched_rows_indices.add(idx)
                exact_matches.append(target_id)
                matched_count += 1
                
                # Record mapping for exact match (OCR ID matches Target ID)
                real_name = row[name_col] if name_col and name_col in df_input.columns else None
                ocr_to_real_mapping[target_id] = {'id': target_id, 'name': real_name}
        
        if exact_matches:
            logging.info(f"Exact matched {len(exact_matches)} students directly.")

        # 2. Fuzzy Matching (Global Greedy)
        if fuzzy_threshold < 100:
            # Identify candidates for fuzzy matching
            unmatched_targets = [] # List of (index, id)
            for idx, row in df_input.iterrows():
                if idx not in matched_rows_indices:
                    target_id = row[id_col]
                    if not pd.isna(target_id) and target_id:
                        unmatched_targets.append((idx, target_id))
            
            unmatched_ocr = [oid for oid in ocr_ids if oid not in matched_ocr_ids]
            
            # Calculate all pairs scores
            match_candidates = []
            for idx, t_id in unmatched_targets:
                for o_id in unmatched_ocr:
                    score = Levenshtein.ratio(t_id, str(o_id)) * 100.0
                    match_candidates.append((score, idx, t_id, o_id))
            
            # Sort by score descending
            match_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Greedy assignment
            used_target_indices = set()
            used_ocr_ids = set()
            
            # First pass: Apply matches >= threshold
            for score, idx, t_id, o_id in match_candidates:
                if score < fuzzy_threshold:
                    break # Since sorted, we can stop here for applied matches
                    
                if idx in used_target_indices or o_id in used_ocr_ids:
                    continue
                    
                # Apply match
                mark_val = ocr_id_to_mark[o_id]
                # Format mark for CSV (string with comma for decimal if needed)
                if isinstance(mark_val, (int, float)):
                    mark_val = round(mark_val, 2)
                    
                df_input.at[idx, mark_col] = mark_val
                logging.info(f"Fuzzy matched '{t_id}' with OCR ID '{o_id}' (Score: {score:.1f}%) -> Mark: {mark_val}")
                
                matched_count += 1
                matched_ocr_ids.add(o_id) # Update global set for reporting
                used_target_indices.add(idx)
                used_ocr_ids.add(o_id)

                # Record mapping for fuzzy match
                real_name = df_input.at[idx, name_col] if name_col and name_col in df_input.columns else None
                ocr_to_real_mapping[o_id] = {'id': t_id, 'name': real_name}

            # Second pass: Show skipped matches below threshold
            logging.info("-" * 20 + f" Current Matching Threshold: {fuzzy_threshold} " + "-" * 20)
            
            skipped_matches_count = 0
            for score, idx, t_id, o_id in match_candidates:
                if score >= fuzzy_threshold:
                    continue # Already processed
                
                # Only show if neither side has been matched yet (true misses)
                if idx in used_target_indices or o_id in used_ocr_ids:
                    continue
                
                if score > 40: 
                    logging.info(f"Skipped match '{t_id}' with OCR ID '{o_id}' (Score: {score:.1f}%)")
                    skipped_matches_count += 1
            
            if skipped_matches_count == 0:
                logging.info("No significant matches found below threshold.")

        # Filter columns if requested
        if simplify_csv:
            if name_col:
                if name_col in df_input.columns:
                     logging.info(f"Simplifying output CSV to columns: {id_col}, {name_col}, {mark_col}")
                     # Ensure we don't duplicate columns if id_col == name_col (unlikely but possible)
                     cols_to_keep = [id_col, name_col, mark_col]
                     # Remove duplicates while preserving order
                     cols_to_keep = list(dict.fromkeys(cols_to_keep))
                     df_input = df_input[cols_to_keep]

                     # If mark_col starts with '#', remove it to make it importable (Moodle convention)
                     if mark_col.startswith('#'):
                        clean_col = mark_col.lstrip('#').strip()
                        df_input = df_input.rename(columns={mark_col: clean_col})
                        logging.info(f"Renamed column '{mark_col}' to '{clean_col}' to allow import.")
                        mark_col = clean_col
                else:
                     logging.warning(f"Name column '{name_col}' not found in input file. Skipping simplification.")
            else:
                logging.warning("Simplify CSV requested but --name-column not provided. Skipping simplification.")

        # Save to new file: original_with_marks.ext
        base_name, ext = os.path.splitext(input_file)
        output_file = f"{base_name}_with_marks{ext}"
        
        if file_ext == '.csv':
            # Ensure the mark column is numeric so to_csv respects the decimal separator
            # Only if column exists (it should, as we added it or it existed)
            if mark_col in df_input.columns:
                df_input[mark_col] = pd.to_numeric(df_input[mark_col], errors='coerce')
                
            # Pass decimal separator to to_csv to handle float formatting
            # QUOTE_NONNUMERIC will quote strings (ID, Name) but NOT floats (Mark), 
            # unless they were converted to strings.
            df_input.to_csv(output_file, index=False, encoding=used_encoding, sep=used_sep, decimal=decimal_sep, quoting=csv.QUOTE_MINIMAL)
        elif file_ext in ['.xlsx', '.xls']:
            df_input.to_excel(output_file, index=False)
        elif file_ext == '.tsv':
            df_input.to_csv(output_file, sep='\t', index=False, encoding=used_encoding, quoting=csv.QUOTE_NONNUMERIC)
            
        logging.info(f"Saved marks to {output_file}. Matched {matched_count}/{len(df_input)} students.")

        # Report unmatched
        unmatched_students = []
        for idx, row in df_input.iterrows():
            target_id = row[id_col]
            if not pd.isna(target_id) and target_id and pd.isna(df_input.at[idx, mark_col]):
                 unmatched_students.append(target_id)

        if unmatched_students:
             logging.warning(f"Unmatched students from CSV ({len(unmatched_students)}): {unmatched_students}")
        
        unmatched_ocr_final = [oid for oid in ocr_ids if oid not in matched_ocr_ids]
        if unmatched_ocr_final:
             logging.warning(f"Unmatched exams from OCR ({len(unmatched_ocr_final)}): {unmatched_ocr_final}")
             logging.warning(f"Tip: Try increasing fuzzy match threshold (lower value) or correct the scanned IDs manually in {correction_results_csv}")

        # Update final_marks.csv with matched real IDs and names
        if ocr_to_real_mapping and os.path.exists(final_marks_path):
            try:
                updated_count = 0
                for ocr_id, info in ocr_to_real_mapping.items():
                    # Find row(s) with this OCR ID
                    mask = df_marks['student_id'] == ocr_id
                    if mask.any():
                        df_marks.loc[mask, 'student_id'] = info['id']
                        if info['name']:
                             df_marks.loc[mask, 'student_name'] = info['name']
                        updated_count += 1
                
                if updated_count > 0:
                    df_marks.to_csv(final_marks_path, index=False)
                    logging.info(f"Updated final_marks.csv with {updated_count} matched student IDs/Names.")
                    
                    # Print updated student marks
                    print("\n--- Student Marks (Updated from Roster) ---")
                    # We might need to select specific columns or just print what we have
                    # We assume structure similar to analysis.py: student_id, student_name, score, max_score, mark
                    if 'score' in df_marks.columns and 'max_score' in df_marks.columns:
                        cols_to_print = ['student_id', 'student_name', 'score', 'max_score', 'mark']
                        # Ensure columns exist before filtering
                        cols_to_print = [c for c in cols_to_print if c in df_marks.columns]
                        results_to_print = df_marks[cols_to_print].copy()
                        results_to_print.index = range(1, len(results_to_print) + 1)
                        print(tabulate(results_to_print, headers='keys', tablefmt='psql', floatfmt=".2f"))
                    else:
                        # Fallback if structure is different
                        print(tabulate(df_marks, headers='keys', tablefmt='psql', floatfmt=".2f"))
            
            except Exception as e:
                logging.error(f"Failed to update final_marks.csv: {e}")

    except Exception as e:
        logging.error(f"Failed to fill marks in input file: {e}", exc_info=True)
