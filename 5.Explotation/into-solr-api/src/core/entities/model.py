"""
This module is a class implementation to manage and hold all the information associated with a model trained with the module 4.TM-Training. It provides methods to retrieve information about the model such as topic distribution over documents, topic-word probabilities, and betas. 

Author: Lorena Calvo-Bartolomé
Date: 28/03/2024
"""


import configparser
import json
import os
import pathlib
from typing import List

from scipy import sparse
import yaml

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from src.core.entities.utils import calculate_beta_ds, process_line, sum_up_to
# from utils import sum_up_to


class Model(object):
    """
    A class to manage and hold all the information associated with a topic model so it can be indexed in Solr.
    """

    def __init__(self,
                 path_to_model: pathlib.Path,
                 logger=None,
                 config_file: str = "/config/config.cf") -> None:
        """Init method.

        Parameters
        ----------
        path_to_model: pathlib.Path
            Path to the model folder.
        logger : logging.Logger
            The logger object to log messages and errors.
        config_file: str
            Path to the configuration file.
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Entity Model')

        if not os.path.isdir(path_to_model):
            self._logger.error(
                f'-- -- The provided model path {path_to_model} does not exist.')
        self.path_to_model = path_to_model

        # Get model and corpus names
        self.name = path_to_model.stem.lower()
        self.corpus_name = None

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.thetas_max_sum = int(cf.get('restapi', 'thetas_max_sum'))
        self.betas_max_sum = int(cf.get('restapi', 'betas_max_sum'))

        # Get model information
        self.alphas = np.load(self.path_to_model.joinpath("alphas.npy"))
        self.betas = np.load(self.path_to_model.joinpath("betas.npy"))
        self.betas_ds = calculate_beta_ds(self.betas)
        self.thetas = sparse.load_npz(self.path_to_model.joinpath("thetas.npz"))
        self.ndocs_active = np.array((self.thetas != 0).sum(0).tolist()[0])
        with self.path_to_model.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
            self.vocab = [el.strip() for el in fin.readlines()]
        self.vocab_w2id = {}
        self.vocab_id2w = {}
        with self.path_to_model.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin):
                wd = line.strip()
                self.vocab_w2id[wd] = i
                self.vocab_id2w[str(i)] = wd
        self.sims = sparse.load_npz(self.path_to_model.joinpath('distances.npz'))
        with self.path_to_model.joinpath('tpc_coords.txt').open('r', encoding='utf8') as fin:
            # read the data from the file and convert it back to a list of tuples
            self.coords = [tuple(map(float, line.strip()[1:-1].split(', '))) for line in fin]
        with self.path_to_model.joinpath('tpc_descriptions.txt').open('r', encoding='utf8') as fin:
                self.tpc_descriptions = [el.strip() for el in fin.readlines()]
        with self.path_to_model.joinpath('tpc_labels.txt').open('r', encoding='utf8') as fin:
                self.tpc_labels = [el.strip() for el in fin.readlines()]
        data = {
            "betas": [self.betas],
            "betas_ds": [self.betas_ds],
            "alphas": [self.alphas],
            #"topic_entropy": [self._topic_entropy],
            #"topic_coherence": [self._topic_coherence],
            "ndocs_active": [self.ndocs_active],
            "tpc_descriptions": [self.tpc_descriptions],
            "tpc_labels": [self.tpc_labels],
        }
        self.df = pd.DataFrame(data)
        
        self._logger.info(self.df)

        return

    def get_model_info(self) -> List[dict]:
        """It retrieves the information about a topic model as a list of dictionaries.

        Returns:
        --------
        json_lst: list[dict]
            A list of dictionaries containing information about the topic model.
        """

        # Get model information as dataframe, where each row is a topic
        df, vocab_id2w, vocab = self.df, self.vocab_id2w, self.vocab
        df = df.apply(pd.Series.explode)
        df.reset_index(drop=True)
        df["id"] = [f"t{i}" for i in range(len(df))]

        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        # Get betas scale to self.max_sum
        def get_betas_scale(vector: np.array,
                            max_sum: int):
            vector = sum_up_to(vector, max_sum)
            return vector

        df["betas_scale"] = df["betas"].apply(
            lambda x: get_betas_scale(x, self.betas_max_sum))

        # Get words in each topic
        def get_tp_words(vector: np.array,
                         vocab_id2w: dict) -> str:
            return ", ".join([vocab_id2w[str(idx)] for idx, val in enumerate(vector) if val != 0])

        df["vocab"] = df["betas"].apply(
            lambda x: get_tp_words(x, vocab_id2w))

        # Get betas string representation
        def get_tp_str_rpr(vector: np.array,
                           vocab_id2w: dict) -> str:
            rpr = " ".join([f"{vocab_id2w[str(idx)]}|{val}" for idx,
                           val in enumerate(vector) if val != 0]).rstrip()
            return rpr

        df["betas"] = df["betas_scale"].apply(
            lambda x: get_tp_str_rpr(x, vocab_id2w))

        def get_top_words_betas(row, vocab, n_words=15):
            # a row is a topic
            word_tfidf_dict = {vocab[idx2]: row["betas_ds"][idx2]
                               for idx2 in np.argsort(row["betas_ds"])[::-1][:n_words]}

            # Transform such as value0 = 100 % and following values are relative to value0
            value0 = word_tfidf_dict[list(word_tfidf_dict.keys())[0]]
            word_tfidf_dict = {word: round((val/value0)*100, 3)
                               for word, val in word_tfidf_dict.items()}
            rpr = " ".join(
                [f"{word}|{word_tfidf_dict[word]}" for word in word_tfidf_dict.keys()]).rstrip()

            return rpr

        df['top_words_betas'] = df.apply(
            lambda row: get_top_words_betas(row, vocab), axis=1)

        # Drop betas scale because it is not needed
        df = df.drop(columns=["betas_scale", "betas_ds"])

        # Get topic coordinates in cluster space
        df["coords"] = self.coords

        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)

        return json_lst

    def get_model_info_update(self, action: str) -> List[dict]:
        """
        Retrieves the information from the model that goes to a corpus collection (document-topic proportions) and save it as an update in the format required by Solr.

        Parameters
        ----------
        action: str
            Action to be performed ('set', 'remove')

        Returns:[]
        --------
        json_lst: list[dict]
            A list of dictionaries with thr document-topic proportions update.
        """
        # Get corpus path and name of the collection
        path_model_config = self.path_to_model.joinpath("config.yaml")
        with open(path_model_config, 'r') as file:
            model_config = yaml.safe_load(file)
        self.corpus = model_config.get('path_to_data')
        self.corpus_name = pathlib.Path(self.corpus).name
        if self.corpus_name.endswith(".parquet") or self.corpus_name.endswith(".json"):
            self.corpus_name = self.corpus_name.split(".")[0].lower()

        # Keys for dodument-topic proportions and similarity that will be used within the corpus collection
        model_key = 'doctpc_' + self.name
        sim_model_key = 'sim_' + self.name

        # Get ids of documents kept in the tr corpus

        try:
            with open(self.path_to_model.joinpath("modelFiles/corpus.txt"), encoding="utf-8") as file:
                ids_corpus = [process_line(line) for line in file]
        except FileNotFoundError:
            self._logger.info(
                '-- -- Corpus file not found.')
        

        # Actual topic model's information only needs to be retrieved if action is "set"
        if action == "set":
            # Get doc-topic representation
            def get_doc_str_rpr(vector, max_sum):
                """Calculates the string representation of a document's topic proportions in the format 't0|100 t1|200 ...', so that the sum of the topic proportions is at most max_sum.

                Parameters
                ----------
                vector: numpy.array
                    Array with the topic proportions of a document.
                max_sum: int
                    Maximum sum of the topic proportions.

                Returns 
                -------
                rpr: str
                    String representation of the document's topic proportions.
                """
                vector = sum_up_to(vector, max_sum)
                rpr = ""
                for idx, val in enumerate(vector):
                    if val != 0:
                        rpr += "t" + str(idx) + "|" + str(val) + " "
                rpr = rpr.rstrip()
                return rpr

            self._logger.info("Attaining thetas rpr...")
            thetas_dense = self.thetas.todense()
            doc_tpc_rpr = [get_doc_str_rpr(thetas_dense[row, :], self.thetas_max_sum)
                           for row in range(len(thetas_dense))]

            # Get similarities string representation
            self._logger.info("Attaining sims rpr...")

            with open(self.path_to_model.joinpath('distances.txt'), 'r') as f:
                sim_rpr = [line.strip() for line in f]
            self._logger.info(
                "Thetas and sims attained. Creating dataframe...")

            # Save the information in a dataframe
            df = pd.DataFrame(list(zip(ids_corpus, doc_tpc_rpr, sim_rpr)),
                              columns=['id', model_key, sim_model_key])
            self._logger.info(
                f"Dataframe created. Printing columns:{df.columns.tolist()}")

        elif action == "remove":
            doc_tpc_rpr = ["" for _ in range(len(ids_corpus))]
            sim_rpr = doc_tpc_rpr
            # Save the information in a dataframe
            df = pd.DataFrame(list(zip(ids_corpus, doc_tpc_rpr, sim_rpr)),
                              columns=['id', model_key, sim_model_key])

        # Create json from dataframe
        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)

        # Updating json in the format required by Solr
        new_list = []
        if action == 'set':
            for d in json_lst:
                tpc_dict = {'set': d[model_key]}
                d[model_key] = tpc_dict
                sim_dict = {'set': d[sim_model_key]}
                d[sim_model_key] = sim_dict
                new_list.append(d)
        elif action == 'remove':
            for d in json_lst:
                tpc_dict = {'set': []}
                d[model_key] = tpc_dict
                sim_dict = {'set': []}
                d[sim_model_key] = sim_dict
                new_list.append(d)

        return new_list, self.corpus_name

    def get_corpora_model_update(
        self,
        id: int,
        action: str
    ) -> List[dict]:
        """Generates an update for the CORPUS_COL collection.
        
        Parameters
        ----------
        id: int
            Identifier of the corpus collection in CORPUS_COL
        action: str
            Action to be performed ('add', 'remove')

        Returns:
        --------
        json_lst: list[dict]
            A list of dictionaries with the update.
        """

        json_lst = [{"id": id,
                    "fields": {action: ['doctpc_' + self.name,
                                        'sim_' + self.name]},
                     "models": {action: self.name}
                     }]

        return json_lst


# if __name__ == '__main__':
#    model = Model(pathlib.Path(
#       "/export/data_ml4ds/IntelComp/EWB/data/source/HFRI-30"))
    # json_lst = model.get_model_info_update(action='set')
    # pos = model.get_topic_pos()
    # print(json_lst[0])
#    df = model.get_model_info()
    # print(df[0].keys())
    # upt = model.get_corpora_model_update()
    # print(upt)
