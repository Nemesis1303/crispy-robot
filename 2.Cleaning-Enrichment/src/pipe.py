import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class Pipe(object):
    def __init__(
        self,
        logger: logging.Logger = None
    ):

        if logger is None:
            self._logger = logging.getLogger(__name__)
        else:
            self._logger = logger

        pass
    
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
    
    def _loadACR(self, lang: str) -> None:
        """
        Loads list of acronyms
        """

        self._acr_list = acronyms.en_acronyms_list if lang == 'en' else acronyms.es_acronyms_list

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
    
    def do_pipeline(self, rawtext) -> str:
        """
        Implements the preprocessing pipeline, by carrying out:
        - Lemmatization according to POS
        - Removal of non-alphanumerical tokens
        - Removal of basic English stopwords and additional ones provided       
          within stw_files
        - Acronyms replacement
        - Expansion of English contractions
        - Word tokenization
        - Lowercase conversion

        Parameters
        ----------
        rawtext: str
            Text to preprocess

        Returns
        -------
        final_tokenized: List[str]
            List of tokens (strings) with the preprocessed text
        """

        # Change acronyms by their meaning
        text = self._replace(rawtext, self._acr_list)

        # Expand contractions
        try:
            text = contractions.fix(text)
        except:
            text = text  # this is only for SS

        valid_POS = set(['VERB', 'NOUN', 'ADJ', 'PROPN'])

        doc = self._nlp(text)
        lemmatized = [token.lemma_ for token in doc
                      if token.is_alpha
                      and token.pos_ in valid_POS
                      and not token.is_stop
                      and token.lemma_ not in self._stw_list]

        # Convert to lowercase
        final_tokenized = [token.lower() for token in lemmatized]

        return final_tokenized

    def preproc(self,
                corpus_df: Union[dd.DataFrame, pd.DataFrame],
                use_dask: bool = False,
                nw: int = 0,
                no_ngrams: bool = False) -> Union[dd.DataFrame, pd.DataFrame]:
        """
        Invokes NLP pipeline and carries out, in addition, n-gram detection.

        Parameters
        ----------
        corpus_df: Union[dd.DataFrame, pd.DataFrame]
            Dataframe representation of the corpus to be preprocessed. 
            It needs to contain (at least) the following columns:
            - raw_text
        nw: int
            Number of workers for Dask computations
        no_grams: Bool
            If True, calculation of ngrams will be skipped

        Returns
        -------
        corpus_df: Union[dd.DataFrame, pd.DataFrame]
            Preprocessed DataFrame
            It needs to contain (at least) the following columns:
            - raw_text
            - lemmas
        """

        # Lemmatize text
        self._logger.info("-- Lemmatizing text")
        if use_dask:
            corpus_df["lemmas"] = corpus_df["raw_text"].apply(self.do_pipeline,
                                                              meta=('lemmas', 'str'))
        else:
            corpus_df["lemmas"] = corpus_df["raw_text"].apply(self.do_pipeline)

        # If no_ngrams is False, carry out n-grams detection
        if not no_ngrams:

            def get_ngram(doc):
                return " ".join(phrase_model[doc])

            # Create corpus from tokenized lemmas
            self._logger.info(
                "-- Creating corpus from lemmas for n-grams detection")
            if use_dask:
                with ProgressBar():
                    if nw > 0:
                        lemmas = corpus_df["lemmas"].compute(
                            scheduler='processes', num_workers=nw)
                    else:
                        # Use Dask default number of workers (i.e., number of cores)
                        lemmas = corpus_df["lemmas"].compute(
                            scheduler='processes')
            else:
                lemmas = corpus_df["lemmas"]

            # Create Phrase model for n-grams detection
            self._logger.info("-- Creating Phrase model")
            phrase_model = Phrases(lemmas, min_count=2, threshold=20)

            # Carry out n-grams substitution
            self._logger.info("-- Carrying out n-grams substitution")

            if use_dask:
                corpus_df["lemmas"] = \
                    corpus_df["lemmas"].apply(
                        get_ngram, meta=('lemmas', 'str'))
            else:
                corpus_df["lemmas"] = corpus_df["lemmas"].apply(get_ngram)

        else:
            if use_dask:
                corpus_df["lemmas"] = \
                    corpus_df["lemmas"].apply(
                        lambda x: " ".join(x), meta=('lemmas', 'str'))
            else:
                corpus_df["lemmas"] = corpus_df["lemmas"].apply(
                    lambda x: " ".join(x))

        return corpus_df

    def get_context_embeddings(
        self,
        df: pd.DataFrame,
        calculate_on: str,
        batch_size: int = 128,
        sbert_model: str = "paraphrase-distilroberta-base-v2",
        use_gpu: bool = True  # Add a parameter to specify GPU usage
    ):
        """Calculate embeddings for text columns in a dataframe using SentenceTransformer.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing text columns.
        calculate_on : str
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

        df["embeddings"] = df[calculate_on].apply(encode_text)

        return df
