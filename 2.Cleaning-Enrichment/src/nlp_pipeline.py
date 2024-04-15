import time
import pandas as pd
import logging
import pathlib
import re
import contractions
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from spacy_download import load_spacy
from gensim.models.phrases import Phrases
import os
import torch
from langdetect import detect

from src.acronym_extractor.acronym_extractor import AcronymExtractor
from src.ner_specific_extractor.ner_specific_extractor import NERSpecificExtractor


class NLPpipeline(object):
    def __init__(
        self,
        spaCy_model: str,
        stw_files_path: List[pathlib.Path] = None,
        lang: str = "en",
        valid_POS: List[str] = ['VERB', 'NOUN', 'ADJ', 'PROPN'],
        gensim_phraser_min_count: int = 2,
        gensim_phraser_threshold: int = 20,
        logger: logging.Logger = None
    ):

        if logger is None:
            self._logger = logging.getLogger(__name__)
        else:
            self._logger = logger

        # Load stopwords
        if not stw_files_path:
            stw_files_path = pathlib.Path(os.getcwd()) / "src" / "stw_lists"
            stw_lsts = []
            for entry in stw_files_path.iterdir():
                # check if it is a file
                if entry.as_posix().endswith("txt"):
                    stw_lsts.append(entry)
        self._loadSTW(stw_lsts)

        self._spaCy_model = spaCy_model
        self._lang = lang
        self._valid_POS = set(valid_POS)
        self._gensim_phraser_min_count = gensim_phraser_min_count
        self._gensim_phraser_threshold = gensim_phraser_threshold
        self._lemmas_cols = []

    def _loadSTW(
        self,
        stw_files: List[pathlib.Path]
    ) -> None:
        """
        Loads stopwords as list from files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        """
        
        stw_list = \
            [pd.read_csv(stw_file, names=['stopwords'], header=None,
                         skiprows=3) for stw_file in stw_files]
        stw_list = \
            [stopword for stw_df in stw_list for stopword in stw_df['stopwords']]
        self._stw_list = list(dict.fromkeys(stw_list))  # remove duplicates
        self._logger.info(
            f"Stopwords list created with {len(stw_list)} items.")

        return

    def _replace(
        self,
        text: str,
        patterns: List[Tuple[str, str]]
    ) -> str:
        """
        Replaces patterns in strings.

        Parameters
        ----------
        text: str
            Text in which the patterns are going to be replaced
        patterns: List of tuples
            Replacement to be carried out

        Returns
        -------
        text: str
            Replaced text
        """

        for (raw, rep) in patterns:
            regex = re.compile(raw, flags=re.IGNORECASE)
            text = regex.sub(rep, text)
        return text

    def get_acronyms(
        self,
        df: pd.DataFrame,
        col_calculate_on: str
    ) -> None:
        """
        Extracts acronyms from a DataFrame column and stores them in a list using the AcronymExtractor class.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the text column
        col_calculate_on: str
            Column name to calculate acronyms on

        Returns
        -------
        df: pd.DataFrame
            DataFrame with the acronyms column added to it as "col_calculate_on_ACR"
        """

        def create_acronym_list(acronyms_data):
            acronyms_list = []
            for acronym, full_form in acronyms_data:
                acronym_pattern = r'\b{}\b'.format(acronym)
                acronyms_list.append((acronym_pattern, full_form))

            # Remove duplicates if any
            acronyms_list_dict = dict(acronyms_list)
            acronyms_list = list(acronyms_list_dict.items())

            return acronyms_list

        self._logger.info(f"-- -- Extracting acronyms...")
        start_time = time.time()
        AE = AcronymExtractor(lang =self._lang, logger=self._logger)
        col_save = f"{col_calculate_on}_ACR"
        df[col_save] = df[col_calculate_on].apply(AE.extract)
        acronyms = df[col_save].explode()
        self._acr_list = create_acronym_list(acronyms)
        
        self._logger.info(
            f'Acronyms identification finished in {(time.time() - start_time)}')

        return df
    
    def get_lang(
        self,
        df: pd.DataFrame,
        col_calculate_on: str
    ) -> str:
        
        def det(x: str) -> str:
            """
            Detects the language of a given text

            Parameters
            ----------
            x : str
                Text whose language is to be detected

            Returns
            -------
            lang : str
                Language of the text
            """

            try:
                lang = detect(x)
            except:
                lang = 'Other'
            return lang
        
        self._logger.info(f"-- Detecting language...")
        start_time = time.time()
        
        df['lang'] = df[col_calculate_on].apply(det)
    
        self._logger.info(
            f'-- -- Language detect finished in {(time.time() - start_time)}')
        
        return df

    def get_lemmas(
        self,
        df: pd.DataFrame,
        col_calculate_on: str,
        replace_acronyms: bool = False
    ) -> str:
        """
        Get lemmas from a text column in a DataFrame using spaCy. More specifically, the following steps are carried out:
        - Lemmatization according to POS
        - Removal of non-alphanumerical tokens
        - Removal of basic English stopwords and additional ones provided       
          within stw_files
        - Acronyms replacement, if replace_acronyms is True
        - Expansion of English contractions
        - Word tokenization
        - Lowercase conversion

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the text column

        Returns
        -------
        df : pd.DataFrame
            DataFrame with the lemmas column added to it as "col_calculate_on_LEMMAS"
        """

        # Download spaCy model if not already downloaded and load
        # Disable parser and NER to speed up processing
        nlp = load_spacy(self._spaCy_model, exclude=['parser', 'ner'])
        
        def do_lemmas(text):
            # 1. Change acronyms by their meaning
            if replace_acronyms:
                if not hasattr(self, '_acr_list'):
                    self._logger.info(
                        f"-- -- Acronyms have not been extracted yet. Proceeding with the extraction...")
                    self.get_acronyms(df, col_calculate_on)
                # Change acronyms by their meaning
                text = self._replace(text, self._acr_list)
            # 2. Expand contractions
            try:
                text = contractions.fix(text)
            except:
                text = text
            # 3. Lemmatize text
            try:
                doc = nlp(text)
            except Exception as e:
                self._logger.error(
                    f"-- -- Error ocurred while applying Spacy pipe: {e}")
                try:
                    # If error, try to apply the pipe with a larger max_length
                    self.nlp_max_length = nlp.max_length
                    nlp.max_length = len(text)
                    doc = nlp(text)
                    nlp.max_length = self.nlp_max_length
                except Exception as e:
                    self._logger.error(
                        f"-- -- Error ocurred while applying Spacy pipe: {e} after increasing max_length...")
                    nlp.max_length = self.nlp_max_length
                    return []

            lemmas = [token.lemma_ for token in doc
                      if token.is_alpha
                      and token.pos_ in self._valid_POS
                      and not token.is_stop
                      and token.lemma_ not in self._stw_list]

            # Convert to lowercase
            final_tokenized = " ".join([token.lower() for token in lemmas])

            return final_tokenized

        # Apply lemmatization to the text
        self._logger.info(f"-- -- Extracting acronyms...")
        start_time = time.time()
        col_save = f"{col_calculate_on}_LEMMAS"
        df[col_save] = df[col_calculate_on].apply(do_lemmas)
        
        # Save the column name for future reference
        self._lemmas_cols.append(col_save)
        self._logger.info(
            f'-- -- Lemmatization finished in {(time.time() - start_time)}')
        return df
    
    def get_ngrams(
        self,
        df: pd.DataFrame,
        col_calculate_on: str
    ):
        """
        Get n-grams from a text column in a DataFrame using the Phrases model from Gensim.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the text column

        Returns
        -------
        df : pd.DataFrame
            DataFrame with the n-grams column added to it as "col_calculate_on_NGRAMS"
        """

        def get_ngram(doc):
            return " ".join(phrase_model[doc])
        
        col_calculate_on_lemmas = f"{col_calculate_on}_LEMMAS"
        if col_calculate_on_lemmas not in self._lemmas_cols:
            self._logger.error(
                f"-- -- Lemmas for column {col_calculate_on} have not been calculated yet. Please calculate lemmas first.")
            return df

        # Create corpus from tokenized lemmas
        self._logger.info(f"-- -- Extracting n-grams...")
        start_time = time.time()
        df[col_calculate_on_lemmas] = df[col_calculate_on_lemmas].apply(lambda x: x.split())
        lemmas = df[col_calculate_on_lemmas]
        # Create Phrase model for n-grams detection
        phrase_model = Phrases(
            lemmas,
            min_count=self._gensim_phraser_min_count, 
            threshold=self._gensim_phraser_threshold)
        
        # Carry out n-grams substitution
        # N-grams are stored in the same column as lemmas (directly substituting them)        
        df[col_calculate_on_lemmas] = df[col_calculate_on_lemmas].apply(get_ngram) 
        self._logger.info(
            f"-- -- N-grams extraction finished in {(time.time() - start_time)}")

        return df

    def get_context_embeddings(
        self,
        df: pd.DataFrame,
        col_calculate_on: str,
        batch_size: int = 128,
        sbert_model: str = "paraphrase-distilroberta-base-v2",
        use_gpu: bool = True  # Add a parameter to specify GPU usage
    ):
        """Calculate embeddings for text columns in a dataframe using SentenceTransformer.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing text columns.
        col_calculate_on : str
            Column name to calculate embeddings on.
        batch_size : int, optional
            Batch size for SentenceTransformer, by default 128.
        sbert_model : str, optional
            SentenceTransformer model to use, by default "paraphrase-distilroberta-base-v2".
        use_gpu : bool, optional
            Whether to use GPU for inference, by default True.

        Returns
        -------
        pd.DataFrame
            Dataframe with embeddings added.
        """

        self._logger.info(f"-- -- Extracting embeddings...")
        start_time = time.time()
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(sbert_model, device=device)

        def encode_text(text):
            embedding = model.encode(
                text,
                show_progress_bar=True,
                batch_size=batch_size
            )
            # Convert to string to save space
            embedding = ' '.join(str(x) for x in embedding)
            return embedding

        col_save = f"{col_calculate_on}_EMBEDDINGS"
        df[col_save] = df[col_calculate_on].apply(encode_text)
        
        self._logger.info(
            f"-- -- Embeddings extraction finished in {(time.time() - start_time)}")

        return df
    
    def get_ner_generic(
        self,
        df: pd.DataFrame,
        col_calculate_on: str
    ):  
        
        col_calculate_on_lemmas = f"{col_calculate_on}_LEMMAS"
        if col_calculate_on_lemmas not in self._lemmas_cols:
            self._logger.error(
                f"-- -- Lemmas for column {col_calculate_on} have not been calculated yet. Please calculate lemmas first.")
            return df
        
        # Download spaCy model if not already downloaded and load
        # Disable parser and NER to speed up processing
        nlp = load_spacy(self._spaCy_model, exclude=['lemmatization', 'parser'])

        def get_ners(text):
            try:
                doc = nlp(text)
            except Exception as e:
                self._logger.error(
                    f"-- -- Error ocurred while applying Spacy pipe: {e}")
                try:
                    # If error, try to apply the pipe with a larger max_length
                    self.nlp_max_length = nlp.max_length
                    nlp.max_length = len(text)
                    doc = nlp(text)
                    nlp.max_length = self.nlp_max_length
                except Exception as e:
                    self._logger.error(
                        f"-- -- Error ocurred while applying Spacy pipe: {e} after increasing max_length...")
                    nlp.max_length = self.nlp_max_length
                    return []

            ners = [(ent.text, ent.label_) for ent in doc.ents]
            return ners

        # Apply NER extraction to the text
        self._logger.info(f"-- -- Extracting generic NERS...")
        start_time = time.time()
        col_save = f"{col_calculate_on}_GEN_NERS"
        df[col_save] = df[col_calculate_on_lemmas].apply(get_ners)
        
        # Save the column name for future reference
        self._lemmas_cols.append(col_save)
        self._logger.info(
            f'-- -- Generic NER identification finished in {(time.time() - start_time)}')
        
        return df
    
    def get_ner_specific(
        self,
        df: pd.DataFrame,
        col_calculate_on: str
    ):  
        
        self._logger.info(f"-- -- Extracting specific NER...")
        start_time = time.time()
        NSE = NERSpecificExtractor(lang =self._lang, logger=self._logger)
        col_save = f"{col_calculate_on}_SPEC_NERS"
        df[col_save] = df[col_calculate_on].apply(NSE.extract)
       
        
        self._logger.info(
            f'-- -- Specific NER identification finished in {(time.time() - start_time)}')
        
        return df
