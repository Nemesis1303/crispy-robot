"""
This module contains the Preprocessor class, which is used to carry out some simple text preprocessing tasks needed by topic modeling (4.TM-Training): stopword removal, replace equivalent terms, calculate BoW, and generate the training files required by the supported TM technologies.
"""

import json
import logging
import pathlib
import time
from typing import Dict, List, Union

import pandas as pd
from gensim import corpora


class Preprocessor:
    def __init__(
        self,
        stw_files: List[str] = [],
        eq_files: List[str] = [],
        min_lemas: int = 15,
        no_below: int = 10,
        no_above: float = 0.6,
        keep_n: int = 100000,
        logger: logging.Logger = None
    ) -> None:
        """
        Initialize the Preprocessor.

        Parameters
        ----------
        stw_files : List[str]
            List of paths to stopwords files.
        eq_files : List[str]
            List of paths to equivalent terms files.
        min_lemas : int
            Minimum number of lemmas for document filtering.
        no_below : int
            Minimum number of documents to keep a term in the vocabulary.
        no_above : float
            Maximum proportion of documents to keep a term in the vocabulary.
        keep_n : int
            Maximum vocabulary size.
        logger : logging.Logger
            Logger for logging object activity.
        """
        self._stopwords = self._load_stw(stw_files)
        self._equivalences = self._load_eq(eq_files)
        self._min_lemas = min_lemas
        self._no_below = no_below
        self._no_above = no_above
        self._keep_n = keep_n

        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level=logging.INFO)
            self._logger = logging.getLogger(__name__)

    def _load_stw(self, stw_files: List[str]) -> List[str]:
        """
        Loads all stopwords from all files provided in the argument.

        Parameters
        ----------
        stw_files : List[str]
            List of paths to stopwords files.

        Returns
        -------
        List[str]
            List of stopwords.
        """
        stop_words = []
        for stw_file in stw_files:
            with pathlib.Path(stw_file).open('r', encoding='utf8') as fin:
                stop_words += json.load(fin)['wordlist']
        return list(set(stop_words))

    def _load_eq(self, eq_files: List[str]) -> Dict[str, str]:
        """
        Loads all equivalent terms from all files provided in the argument.

        Parameters
        ----------
        eq_files : List[str]
            List of paths to equivalent terms files.

        Returns
        -------
        Dict[str, str]
            Dictionary of term_to_replace -> new_term.
        """
        equivalents = {}
        for eq_file in eq_files:
            with pathlib.Path(eq_file).open('r', encoding='utf8') as fin:
                new_eq = json.load(fin)['wordlist']
            new_eq = [x.split(':') for x in new_eq]
            new_eq = [x for x in new_eq if len(x) == 2]
            new_eq_dict = dict(new_eq)
            equivalents = {**equivalents, **new_eq_dict}
        return equivalents

    def preproc(
        self,
        df: pd.DataFrame,
        colname: str
    ) -> pd.DataFrame:
        """
        Preprocesses the documents in the dataframe to carry out the following tasks:
        - Filter out short documents (below min_lemas).
        - Cleaning of stopwords.
        - Equivalent terms application.
        - BoW calculation.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the documents to preprocess.
        colname : str
            The name of the column in the dataframe that contains the text data.

        Returns
        -------
        pd.DataFrame
            A new dataframe with a new column 'tr_tokens' containing the preprocessed text data.
        """
        def tkz_clean_str(rawtext: str) -> List[str]:
            """
            Tokenizes and cleans the text.

            Parameters
            ----------
            rawtext : str
                String with the text to lemmatize.

            Returns
            -------
            List[str]
                Cleaned text tokens.
            """
            if not rawtext:
                return []
            else:
                # Lowercase and tokenization
                cleantext = rawtext.lower().split()
                # Remove stopwords
                cleantext = [el for el in cleantext if el not in self._stopwords]
                # Replace equivalent words
                cleantext = [
                    self._equivalences[el] if el in self._equivalences else el for el in cleantext
                ]
                # Remove stopwords again in case equivalences introduced new stopwords
                cleantext = [el for el in cleantext if el not in self._stopwords]
            return cleantext

        self._logger.info("Preprocessing documents...")
        start_time = time.time()
        
        # Compute tokens, clean them, and filter out documents with less than the minimum number of lemmas
        df['tr_tokens'] = df[colname].apply(tkz_clean_str)
        df = df[df['tr_tokens'].apply(len) >= self._min_lemas]

        # Gensim dictionary creation
        tr_tokens = df['tr_tokens'].values.tolist()
        gensim_dict = corpora.Dictionary(tr_tokens)

        # Remove words that appear in less than no_below documents, in more than no_above, and keep at most keep_n most frequent terms
        gensim_dict.filter_extremes(
            no_below=self._no_below,
            no_above=self._no_above,
            keep_n=self._keep_n
        )
        
        df_new = df.drop(columns=[colname])
        
        self._logger.info(
            f"Preprocessing done in {time.time() - start_time:.2f} seconds"
        )

        return df_new
