"""
This script offers an Embedder class that provides methods for generating embeddings for a given list of sentences or words using a SBERT model. It also provides a method for creating SBERT embeddings for each row in a dask or pandas dataframe and saving the embeddings in a new column.

Author: Lorena Calvo-Bartolomé
Date: 15/06/2024
"""

import logging
from typing import List, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class Embedder(object):
    def __init__(
        self,
        logger: logging.Logger = None,
    ) -> None:

        # Set logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def _get_sentence_embedding(
        self,
        sent: list[str],
        sbert_model_to_load: str ="paraphrase-distilroberta-base-v2",
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get the vector representation of a sentence using a BERT or Word2Vec model. If Word2Vec is used, the input sentence is tokenized and the vector is the average of the vectors of the tokens in the sentence. If BERT is used, the sentence is embedded using the SBERT model specified.

        Parameters
        ----------
        sent: list[str]
            The sentence to get the vector representation.
        sbert_model_to_load: str
            Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
        batch_size: int (default=32)
            The batch size used for the computation
        max_seq_length: int
            Context of the transformer model used for the embeddings generation

        Returns
        -------
        np.ndarray
            The vector representation of the sentence.
        """

        model = SentenceTransformer(sbert_model_to_load)

        embeddings = model.encode(
            sent, show_progress_bar=True, batch_size=batch_size)

        return embeddings

    def infer_embeddings(
        self,
        embed_from: list[list[str]],
        sentence_transformer_model: str = "paraphrase-distilroberta-base-v2"
    ) -> list:
        """
        Infer embeddings for a given list of sentences or words using the specified method.

        Parameters
        ----------
        embed_from: list[list[str]]
            A list of sentences or words for which embeddings need to be inferred.

        Returns
        -------
            list: A list of embeddings for each sentence or word in embed_from.

        Raises
        ------
            FileNotFoundError
                If the model_path is not provided and do_train_w2vec is False.
        """

        if len(embed_from) == 1:
            return self._get_sentence_embedding(
                sent=embed_from[0], sbert_model_to_load=sentence_transformer_model)
        else:
            return [
                self._get_sentence_embedding(
                    sent=sent, sbert_model_to_load=sentence_transformer_model)
                for sent in embed_from]

    def bert_embeddings_from_df(
        self,
        df: Union[dd.DataFrame, pd.DataFrame],
        text_columns: List[str],
        sbert_model_to_load: str,
        batch_size: int = 32,
        use_dask=False
    ) -> Union[dd.DataFrame, pd.DataFrame]:
        """
        Creates SBERT Embeddings for each row in a dask or pandas dataframe and saves the embeddings in a new column.

        Parameters
        ----------
        df : Union[dd.DataFrame, pd.DataFrame]
            The dataframe containing the sentences to embed
        text_column : str
            The name of the column containing the text to embed
        sbert_model_to_load : str
            Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
        batch_size : int (default=32)
            The batch size used for the computation

        Returns
        -------
        df: Union[dd.DataFrame, pd.DataFrame]
            The dataframe with the original data and the generated embeddings
        """

        model = SentenceTransformer(sbert_model_to_load)
        
        def encode_text(text):
            embedding = model.encode(text,
                                     show_progress_bar=True, batch_size=batch_size)
            # Convert to string
            embedding = ' '.join(str(x) for x in embedding)
            return embedding

        for col in text_columns:
            col_emb = col.split(
                "_")[0]+"_embeddings" if len(text_columns) > 1 else "embeddings"

            if use_dask:
                df[col_emb] = df[col].apply(
                    encode_text, meta=('x', 'str'))
            else:
                df[col_emb] = df[col].apply(encode_text)

        return df
