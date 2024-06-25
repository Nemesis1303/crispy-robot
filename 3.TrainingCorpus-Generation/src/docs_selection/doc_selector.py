"""
This module contains the DocSelector class to filter documents by Named Entities Recognition (NER) labels. It provides two methods:

- filter_docs_by_ners: It filters documents by NER labels. It can run in two modes: with_freq = False (default) and with_freq = True. In the first mode, it filters out all the words in the document that are not NERs, keeping the original order and frequency. In the second mode, it filters out all the words in the document that are not NERs and returns a dictionary with the words as keys and the frequency of the word in the document as values.
- filter_docs_if_ner: It filters documents if they contain a minimum number of NERs of a target label. It returns a DataFrame with the documents that contain at least min_count NERs of the target label.
"""

from collections import Counter
import logging  
import time
from typing import Any, Dict, List, Union
import pandas as pd


class DocSelector:
    def __init__(self, logger: logging.Logger = None) -> None:
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level=logging.INFO)
            self._logger = logging.getLogger(__name__)

    def filter_docs_by_ners(
        self,
        df: pd.DataFrame,
        target_label: str,
        lemmas_col: str = 'raw_text_LEMMAS',
        ner_col: str = 'raw_text_SPEC_NERS',
        with_freq: bool = False,
        remove_empty: bool = False
    ) -> pd.DataFrame:
        """
        Filter documents by Named Entities Recognition (NER) labels. It runs in two modes:
        - with_freq = False: It filters out all the words in the document that are not NERs, keeping the original order and frequency.
        - with_freq = True: It filters out all the words in the document that are not NERs and returns a dictionary with the words as keys and the frequency of the word in the document as values.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the documents to filter.
        target_label: str
            Target NER label to filter the documents.
        lemmas_col: str
            Column name with the lemmas of the documents.
        ner_col: str
            Column name with the NERs of the documents.
        with_freq: bool
            If True, it returns a dictionary with the words and their frequency. If False, it returns a string with the words in the same order as the original document.
        remove_empty: bool
            If True, it removes the rows where the filtered words are empty.

        Returns
        -------
        pd.DataFrame
            DataFrame with a new column with the filtered words. The name of the new column is the target_label with the suffix '_NER_IN_ROW'.
        """

        if lemmas_col not in df.columns or ner_col not in df.columns:
            self._logger.error(
                f"Columns {lemmas_col} and {ner_col} must be in the DataFrame. Exiting..."
            )
            return df

        # Filter the DataFrame to keep only rows that contain the target_label in the ner_col
        df_filtered = df[df[ner_col].apply(lambda ners: any(
            label == target_label for _, label in ners))]

        # Check if the target label is present
        if df_filtered.empty:
            self._logger.error(
                f"No documents contain the target label '{target_label}'. Exiting..."
            )
            return df
        
        def filter_ners_in_row(row: pd.Series, target_label: str, lemmas_col: str, ner_col: str, with_freq: bool) -> Union[str, Dict[str, int]]:
            """
            Filter the NERs in a row of the DataFrame.

            Parameters
            ----------
            row: pd.Series
                Row of the DataFrame.
            target_label: str
                Target NER label to filter the words.
            lemmas_col: str
                Column name with the lemmas of the documents.
            ner_col: str
                Column name with the NERs of the documents.
            with_freq: bool
                If True, return a dictionary with the words and their frequency. If False, return a string with the words in the same order as the original document.

            Returns
            -------
            str or dict
                If with_freq is False, return a string with the filtered words. If with_freq is True, return a dictionary with the words and their frequency.
            """
            words = row[lemmas_col].split()
            words_lower = [word.lower() for word in words]

            # Ensure ner_col is a list of tuples and normalize case
            ner_words = [ner[0].replace(' ', '_').lower() for ner in row[ner_col] if ner[1] == target_label]

            # Merge words that appear as separated in ner_words
            def merge_words(words: List[str], modified_list: List[str]) -> List[str]:
                """
                Merge consecutive words in the list if their combined form with an underscore is present in the modified list.
                """
                merged = []
                skip_next = False
                for i in range(len(words)):
                    if skip_next:
                        skip_next = False
                        continue
                    if i < len(words) - 1 and f"{words[i]}_{words[i+1]}" in modified_list:
                        merged.append(f"{words[i]}_{words[i+1]}")
                        skip_next = True
                    else:
                        merged.append(words[i])
                return merged
            merged_words = merge_words(words_lower, ner_words)

            # Filter merged words to keep only those in ner_words
            filtered_words = [word for word in merged_words if word in ner_words]
            
            if with_freq:
                word_freq = Counter(filtered_words)
                return dict(word_freq)
            else:
                return ' '.join(filtered_words)


        self._logger.info(
            f"Filtering NERs in the documents with{'out' if not with_freq else ''} frequency..."
        )

        start_time = time.time()

        df[f'{target_label}_NER_IN_ROW'] = df.apply(lambda row: filter_ners_in_row(
            row, target_label, lemmas_col, ner_col, with_freq), axis=1)

        self._logger.info(
            f"Filtering NERs in the documents completed in {time.time() - start_time:.2f} seconds."
        )

        if remove_empty:
            self._logger.info(
                f"Removing rows with empty {target_label}_NER_IN_ROW..."
            )
            df = df[(df[f'{target_label}_NER_IN_ROW'] != '') & (
                df[f'{target_label}_NER_IN_ROW'].apply(lambda x: x != {}))]
            self._logger.info(
                f"Rows with empty {target_label}_NER_IN_ROW removed."
            )

        return df

    def filter_docs_if_ner(
        self,
        df: pd.DataFrame,
        target_label: str,
        ner_col: str = 'raw_text_SPEC_NERS',
        min_count: int = 1
    ) -> pd.DataFrame:
        """
        Filter documents if they contain a minimum number of NERs of a target label.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the documents to filter.
        target_label: str
            Target NER label to filter the documents.
        ner_col: str
            Column name with the NERs of the documents.
        min_count: int
            Minimum number of NERs of the target label to keep the document.

        Returns
        -------
        pd.DataFrame
            DataFrame with the documents that contain at least min_count NERs of the target label.
        """

        if ner_col not in df.columns:
            self._logger.error(
                f"Column {ner_col} must be in the DataFrame. Exiting..."
            )
            return df
        
        def count_ners(ners: Any, target_label: str) -> int:
            return sum(1 for _, label in ners if label == target_label)

        self._logger.info(
            f"Filtering documents with at least {min_count} NERs of the target label '{target_label}'..."
        )

        start_time = time.time()

        df['NER_COUNT'] = df[ner_col].apply(
            lambda x: count_ners(x, target_label))

        # Check if all rows have NER_COUNT == 0
        if (df['NER_COUNT'] == 0).all():
            self._logger.error(
                f"No documents contain the target label '{target_label}'. Exiting..."
            )
            return df

        filtered_df = df[df['NER_COUNT'] >= min_count].copy()
        filtered_df.drop(columns=['NER_COUNT'], inplace=True)

        self._logger.info(
            f"Filtering documents completed in {time.time() - start_time:.2f} seconds."
        )

        self._logger.info(
            f"{len(filtered_df) / len(df):.2%} documents contain at least {min_count} NERs of the target label '{target_label}'."
        )

        return filtered_df
