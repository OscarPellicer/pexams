import os
import logging
import pandas as pd
import Levenshtein
import csv
from tabulate import tabulate


def _read_roster(input_file: str, file_ext: str, encoding: str, sep: str):
    """The roster as a DataFrame, or None (logged) for an unsupported extension."""
    if file_ext == '.csv':
        return pd.read_csv(input_file, encoding=encoding, sep=sep)
    if file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(input_file)
    if file_ext == '.tsv':
        return pd.read_csv(input_file, sep='\t', encoding=encoding)
    logging.error(f"Unsupported input file format: {file_ext}")
    return None


def _read_marks(correction_results_csv: str, final_marks_path: str):
    """The marks table and the column holding the mark: final_marks.csv (scaled mark) when it
    exists, otherwise the raw scores of the correction results."""
    if os.path.exists(final_marks_path):
        return pd.read_csv(final_marks_path), 'mark'
    logging.warning(f"final_marks.csv not found, using raw scores from {correction_results_csv}")
    return pd.read_csv(correction_results_csv), 'score'


def _rounded(mark_val):
    return round(mark_val, 2) if isinstance(mark_val, (int, float)) else mark_val


def _roster_name(df_input, idx, name_col):
    return df_input.at[idx, name_col] if name_col and name_col in df_input.columns else None


class _MarkMatcher:
    """Assigns the exam marks (keyed by scanned ID) to the roster rows, exactly and then fuzzily,
    tracking which rows/exams are matched and the scanned-ID -> roster mapping."""

    def __init__(self, df_input, id_col, mark_col, name_col, ocr_id_to_mark):
        self.df_input = df_input
        self.id_col = id_col
        self.mark_col = mark_col
        self.name_col = name_col
        self.ocr_id_to_mark = ocr_id_to_mark
        self.ocr_ids = list(ocr_id_to_mark.keys())
        self.matched_count = 0
        self.matched_ocr_ids = set()
        self.matched_rows_indices = set()
        self.ocr_to_real_mapping = {}

    def _valid_ids(self):
        for idx, row in self.df_input.iterrows():
            target_id = row[self.id_col]
            if not pd.isna(target_id) and target_id:
                yield idx, target_id

    def match_exact(self):
        exact_matches = []
        for idx, target_id in self._valid_ids():
            if target_id in self.ocr_id_to_mark:
                self.df_input.at[idx, self.mark_col] = _rounded(self.ocr_id_to_mark[target_id])
                self.matched_ocr_ids.add(target_id)
                self.matched_rows_indices.add(idx)
                exact_matches.append(target_id)
                self.matched_count += 1
                self.ocr_to_real_mapping[target_id] = {'id': target_id, 'name': _roster_name(self.df_input, idx, self.name_col)}
        if exact_matches:
            logging.info(f"Exact matched {len(exact_matches)} students directly.")

    def _fuzzy_candidates(self):
        """All (score, row index, roster ID, scanned ID) pairs still unmatched, best first."""
        unmatched_targets = [(idx, t_id) for idx, t_id in self._valid_ids() if idx not in self.matched_rows_indices]
        unmatched_ocr = [oid for oid in self.ocr_ids if oid not in self.matched_ocr_ids]
        match_candidates = [
            (Levenshtein.ratio(t_id, str(o_id)) * 100.0, idx, t_id, o_id)
            for idx, t_id in unmatched_targets
            for o_id in unmatched_ocr
        ]
        match_candidates.sort(key=lambda x: x[0], reverse=True)
        return match_candidates

    def match_fuzzy(self, fuzzy_threshold):
        """Global greedy assignment of the pairs above the threshold; logs the near misses below it."""
        match_candidates = self._fuzzy_candidates()
        used_target_indices = set()
        used_ocr_ids = set()

        for score, idx, t_id, o_id in match_candidates:
            if score < fuzzy_threshold:
                break
            if idx in used_target_indices or o_id in used_ocr_ids:
                continue
            mark_val = _rounded(self.ocr_id_to_mark[o_id])
            self.df_input.at[idx, self.mark_col] = mark_val
            logging.info(f"Fuzzy matched '{t_id}' with OCR ID '{o_id}' (Score: {score:.1f}%) -> Mark: {mark_val}")
            self.matched_count += 1
            self.matched_ocr_ids.add(o_id)
            used_target_indices.add(idx)
            used_ocr_ids.add(o_id)
            self.ocr_to_real_mapping[o_id] = {'id': t_id, 'name': _roster_name(self.df_input, idx, self.name_col)}

        logging.info("-" * 20 + f" Current Matching Threshold: {fuzzy_threshold} " + "-" * 20)
        skipped_matches_count = 0
        for score, idx, t_id, o_id in match_candidates:
            if score >= fuzzy_threshold:
                continue
            # Only show if neither side has been matched yet (true misses)
            if idx in used_target_indices or o_id in used_ocr_ids:
                continue
            if score > 40:
                logging.info(f"Skipped match '{t_id}' with OCR ID '{o_id}' (Score: {score:.1f}%)")
                skipped_matches_count += 1
        if skipped_matches_count == 0:
            logging.info("No significant matches found below threshold.")


def _simplify_columns(df_input, id_col, name_col, mark_col):
    """Keep only ID, name and mark (Moodle import); a leading '#' is stripped from the mark column."""
    if not name_col:
        logging.warning("Simplify CSV requested but --name-column not provided. Skipping simplification.")
        return df_input, mark_col
    if name_col not in df_input.columns:
        logging.warning(f"Name column '{name_col}' not found in input file. Skipping simplification.")
        return df_input, mark_col
    logging.info(f"Simplifying output CSV to columns: {id_col}, {name_col}, {mark_col}")
    cols_to_keep = list(dict.fromkeys([id_col, name_col, mark_col]))
    df_input = df_input[cols_to_keep]
    if mark_col.startswith('#'):
        clean_col = mark_col.lstrip('#').strip()
        df_input = df_input.rename(columns={mark_col: clean_col})
        logging.info(f"Renamed column '{mark_col}' to '{clean_col}' to allow import.")
        mark_col = clean_col
    return df_input, mark_col


def _write_roster(df_input, input_file, file_ext, mark_col, encoding, sep, decimal_sep) -> str:
    """Save to <original>_with_marks.<ext> and return its path."""
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_with_marks{ext}"
    if file_ext == '.csv':
        # A numeric mark column lets to_csv apply the decimal separator.
        if mark_col in df_input.columns:
            df_input[mark_col] = pd.to_numeric(df_input[mark_col], errors='coerce')
        df_input.to_csv(output_file, index=False, encoding=encoding, sep=sep, decimal=decimal_sep, quoting=csv.QUOTE_MINIMAL)
    elif file_ext in ['.xlsx', '.xls']:
        df_input.to_excel(output_file, index=False)
    elif file_ext == '.tsv':
        df_input.to_csv(output_file, sep='\t', index=False, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC)
    return output_file


def _report_unmatched(df_input, id_col, mark_col, ocr_ids, matched_ocr_ids, correction_results_csv):
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


def _print_marks(df_marks):
    print("\n--- Student Marks (Updated from Roster) ---")
    if 'score' in df_marks.columns and 'max_score' in df_marks.columns:
        cols_to_print = ['student_id', 'student_name', 'score', 'max_score', 'correct', 'incorrect', 'NA', 'mark']
        cols_to_print = [c for c in cols_to_print if c in df_marks.columns]
        results_to_print = df_marks[cols_to_print].copy()
        results_to_print.index = range(1, len(results_to_print) + 1)
        print(tabulate(results_to_print, headers='keys', tablefmt='psql', floatfmt=".2f"))
    else:
        print(tabulate(df_marks, headers='keys', tablefmt='psql', floatfmt=".2f"))


def _update_final_marks(df_marks, final_marks_path, ocr_to_real_mapping):
    """Replace the scanned IDs in final_marks.csv by the matched roster IDs (and names)."""
    try:
        updated_count = 0
        for ocr_id, info in ocr_to_real_mapping.items():
            mask = df_marks['student_id'] == ocr_id
            if mask.any():
                df_marks.loc[mask, 'student_id'] = info['id']
                if info['name']:
                    # An all-empty name column is read as float; names need an object column.
                    df_marks['student_name'] = df_marks['student_name'].astype(object)
                    df_marks.loc[mask, 'student_name'] = info['name']
                updated_count += 1

        if updated_count > 0:
            df_marks.to_csv(final_marks_path, index=False)
            logging.info(f"Updated final_marks.csv with {updated_count} matched student IDs/Names.")
            _print_marks(df_marks)
    except Exception as e:
        logging.error(f"Failed to update final_marks.csv: {e}")


def fill_marks_in_file(input_file: str, id_col: str, mark_col: str, correction_results_csv: str, 
                       fuzzy_threshold: int = 100, encoding: str = 'utf-8', sep: str = ',', 
                       decimal_sep: str = '.', name_col: str = None, simplify_csv: bool = False):
    """Fills marks into the input file (CSV/XLSX) based on correction results."""
    if not os.path.exists(correction_results_csv):
        logging.error(f"Correction results file not found: {correction_results_csv}")
        return

    try:
        file_ext = os.path.splitext(input_file)[1].lower()
        df_input = _read_roster(input_file, file_ext, encoding, sep)
        if df_input is None:
            return

        final_marks_path = os.path.join(os.path.dirname(correction_results_csv), "final_marks.csv")
        df_marks, mark_source_col = _read_marks(correction_results_csv, final_marks_path)

        if id_col not in df_input.columns:
            logging.error(f"ID column '{id_col}' not found in input file.")
            return

        df_marks[mark_source_col] = pd.to_numeric(df_marks[mark_source_col], errors='coerce')
        df_input[id_col] = df_input[id_col].astype(str).str.strip().str.upper()
        df_marks['student_id'] = df_marks['student_id'].astype(str).str.strip().str.upper()
        # Always clear the mark column to ensure fresh results
        df_input[mark_col] = None

        matcher = _MarkMatcher(df_input, id_col, mark_col, name_col,
                               dict(zip(df_marks['student_id'], df_marks[mark_source_col])))
        matcher.match_exact()
        if fuzzy_threshold < 100:
            matcher.match_fuzzy(fuzzy_threshold)

        if simplify_csv:
            df_input, mark_col = _simplify_columns(df_input, id_col, name_col, mark_col)

        output_file = _write_roster(df_input, input_file, file_ext, mark_col, encoding, sep, decimal_sep)
        logging.info(f"Saved marks to {output_file}. Matched {matcher.matched_count}/{len(df_input)} students.")

        _report_unmatched(df_input, id_col, mark_col, matcher.ocr_ids, matcher.matched_ocr_ids, correction_results_csv)

        if matcher.ocr_to_real_mapping and os.path.exists(final_marks_path):
            _update_final_marks(df_marks, final_marks_path, matcher.ocr_to_real_mapping)

    except Exception as e:
        logging.error(f"Failed to fill marks in input file: {e}", exc_info=True)
