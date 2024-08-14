"""
This module is a class implementation to manage and hold all the information associated with a logical corpus.

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Proyect))
"""

import configparser
import json
from typing import List

import pandas as pd
from gensim.corpora import Dictionary
import pathlib
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from src.core.entities.utils import (convert_datetime_to_strftime,
                                     parseTimeINSTANT)
from datetime import datetime

class Corpus(object):
    """
    A class to manage and hold all the information associated with a logical corpus.
    """

    def __init__(self,
                 path_to_raw: pathlib.Path,
                 logger=None,
                 config_file: str = "/config/config.cf") -> None:
        """Init method.

        Parameters
        ----------
        path_to_raw: pathlib.Path
            Path the raw corpus file.
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
            self._logger = logging.getLogger('Entity Corpus')

        if not path_to_raw.exists():
            self._logger.error(
                f"Path to raw data {path_to_raw} does not exist."
            )
        self.path_to_raw = path_to_raw
        self.name = path_to_raw.stem.lower()
        self.fields = None

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self._logger.info(f"Sections {cf.sections()}")
        if self.name + "-config" in cf.sections():
            section = self.name + "-config"
        else:
            self._logger.error(
                f"Logical corpus configuration {self.name} not found in config file.")
        self.MetadataDisplayed = cf.get(
            section, "MetadataDisplayed").split(",")
        self.SearcheableField = cf.get(section, "SearcheableField").split(",")
        self.EmbeddingsToIndex = cf.get(section, "EmbeddingsToIndex", fallback="").split(",")
        
        return

    def get_docs_raw_info(self) -> List[dict]:
        """Extracts the information contained in the corpus parquet file and transforms into a list of dictionaries.

        Returns:
        --------
        json_lst: list[dict]
            A list of dictionaries containing information about the corpus.
        """

        ddf = pd.read_parquet(self.path_to_raw).fillna("")
        self._logger.info(ddf.head())
        
        # Exploit dataset metadata as independent columns
        df_meta = ddf["metadata"].apply(pd.Series)
        df_meta["pdf_id"] = ddf["pdf_id"]
        df = ddf.merge(df_meta, how="inner", on="pdf_id")
        df = df.drop('metadata', axis=1)
        
        self._logger.info(
            f"-- -- Metadata extracted OK. Columns: {ddf.columns}")
        try:
            df = df.drop('page', axis=1)
        except Exception as e:
            self._logger.info(f"-- -- Page column not found in the metadata: {e}")
            
        # If the id_field is in the SearcheableField, remove it and add the id field (new name for the id_field)
        if "pdf_id" in self.SearcheableField:
            self.SearcheableField.remove("pdf_id")
            self.SearcheableField.append("id")
        self._logger.info(f"SearcheableField {self.SearcheableField}")
        
        date_field = None
        if not "date" in self.SearcheableField:
            for el in ["modDdate","creationDate"]:
                if el in self.SearcheableField:
                    self._logger.info(f"Date field {el} found in SearcheableField. Replacing it by date.")
                    date_field = el
                    self.SearcheableField.remove(el)
                    self.SearcheableField.append("date")
        if date_field is None:
            date_field = "creationDate"
        
        # Check that date is valid
        def convert_date(date):
            date_format = "%Y-%m-%d %H:%M:%S"
            
            try:
                date_ = date.replace("D:", "")
                if date_.endswith("'"):
                    date_ = date.split("'")[0]
                date_str_corrected = date_.replace("'", ":")
                date_time_obj = datetime.strptime(date_str_corrected, date_format)
                date_time_np = pd.to_datetime(date_time_obj)
                
                return date_time_np
            except:
                return pd.to_datetime('now') # TODO: There should be a better way of handeling this
            
        df["creationDate"] = df["creationDate"].apply(convert_date)
        self._logger.info("-- -- Dates from creationDate converted OK.")
        df["modDate"] = df["modDate"].apply(convert_date)
        self._logger.info("-- -- Dates from modDdate converted OK.")
        df["date"] = df[date_field]#.combine_first(df["modDdate"])
        self._logger.info(f"-- -- Dates from date created OK based on {date_field}.")

        # Rename id-field to id and date-field to date
        df = df.rename(columns={"pdf_id": "id"})
        
        # Rename tr_tokens to lemmas
        if "tr_tokens" in df.columns:
            df = df.rename(columns={"tr_tokens": "lemmas"})

        self._logger.info(df.columns)

        # Get number of words per document based on the lemmas column
        # NOTE: Document whose lemmas are empty will have a length of 0
        if "lemmas" in df.columns:
            df["nwords_per_doc"] = df["lemmas"].apply(lambda x: len(x.split()))
        else:
            df["nwords_per_doc"] = df["raw_text"].apply(lambda x: len(x))

        # Get BoW representation
        # We dont read from the gensim dictionary that will be associated with the tm models trained on the corpus since we want to have the bow for all the documents, not only those kept after filering extremes in the dictionary during the construction of the logical corpus
        # check none values: df[df.isna()]
        if "lemmas" in df.columns:
            df['lemmas_'] = df['lemmas'].apply(
                lambda x: x.split() if isinstance(x, str) else [])
        else:
            df['lemmas_'] = df['summary'].apply(
                lambda x: x if isinstance(x, list) else [])
        dictionary = Dictionary()
        df['bow'] = df['lemmas_'].apply(
            lambda x: dictionary.doc2bow(x, allow_update=True) if x else [])
        df['bow'] = df['bow'].apply(
            lambda x: [(dictionary[id], count) for id, count in x] if x else [])
        df['bow'] = df['bow'].apply(lambda x: None if len(x) == 0 else x)
        df = df.drop(['lemmas_'], axis=1)
        df['bow'] = df['bow'].apply(lambda x: ' '.join(
            [f'{word}|{count}' for word, count in x]).rstrip() if x else None)

        self._logger.info("-- -- BoW calculated OK.")

        # Ger embeddings of the documents
        def get_str_embeddings(vector):
            repr = " ".join(
                [f"e{idx}|{val}" for idx, val in enumerate(vector.split())]).rstrip()

            return repr
        
        if self.EmbeddingsToIndex:
            for col in self.EmbeddingsToIndex:
                if col in df.columns:
                    df[col] = df[col].apply(get_str_embeddings)

        # Convert dates information to the format required by Solr ( ISO_INSTANT, The ISO instant formatter that formats or parses an instant in UTC, such as '2011-12-03T10:15:30Z')
        df, cols = convert_datetime_to_strftime(df)
        df[cols] = df[cols].applymap(parseTimeINSTANT)

        self._logger.info("-- -- Dates converted OK.")

        # Create SearcheableField by concatenating all the fields that are marked as SearcheableField in the config file
        df['SearcheableField'] = df[self.SearcheableField].apply(
            lambda x: ' '.join(x.astype(str)), axis=1)

        self._logger.info("-- -- SearcheableField created OK.")

        # Save corpus fields
        self.fields = df.columns.tolist()

        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)
        # save json to file
        with open(self.path_to_raw.parent / f"{self.name}.json", 'w') as f:
            json.dump(json_lst, f)

        self._logger.info(f"-- -- JSON created OK. Metadata: {df.columns}")
        
        return json_lst

    def get_corpora_update(
        self,
        id: int
    ) -> List[dict]:
        """Creates the json to update the 'corpora' collection in Solr with the new logical corpus information.
        """

        # TODO: Update
        fields_dict = [{"id": id,
                        "corpus_name": self.name,
                        "corpus_path": self.path_to_raw.as_posix(),
                        "fields": self.fields,
                        "MetadataDisplayed": self.MetadataDisplayed,
                        "SearcheableFields": self.SearcheableField}]

        return fields_dict

    def get_corpora_SearcheableField_update(
        self,
        id: int,
        field_update: list,
        action: str
    ) -> List[dict]:

        json_lst = [{"id": id,
                    "SearcheableFields": {action: field_update},
                     }]

        return json_lst

    def get_corpus_SearcheableField_update(
        self,
        new_SearcheableFields: str,
        action: str
    ):

        ddf = dd.read_parquet(self.path_to_raw).fillna("")

        # Rename id-field to id, title-field to title and date-field to date
        ddf = ddf.rename(
            columns={self.id_field: "id",
                     self.title_field: "title",
                     self.date_field: "date"})

        with ProgressBar():
            df = ddf.compute(scheduler='processes')

        if action == "add":
            new_SearcheableFields = [
                el for el in new_SearcheableFields if el not in self.SearcheableField]
            if self.title_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.title_field)
                new_SearcheableFields.append("title")
            if self.date_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.date_field)
                new_SearcheableFields.append("date")
            new_SearcheableFields = list(
                set(new_SearcheableFields + self.SearcheableField))
        elif action == "remove":
            if self.title_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.title_field)
                new_SearcheableFields.append("title")
            if self.date_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.date_field)
                new_SearcheableFields.append("date")
            new_SearcheableFields = [
                el for el in self.SearcheableField if el not in new_SearcheableFields]

        df['SearcheableField'] = df[new_SearcheableFields].apply(
            lambda x: ' '.join(x.astype(str)), axis=1)

        not_keeps_cols = [el for el in df.columns.tolist() if el not in [
            "id", "SearcheableField"]]
        df = df.drop(not_keeps_cols, axis=1)

        # Create json from dataframe
        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)

        new_list = []
        for d in json_lst:
            d["SearcheableField"] = {"set": d["SearcheableField"]}
            new_list.append(d)

        return new_list, new_SearcheableFields


# if __name__ == '__main__':
#    corpus = Corpus(pathlib.Path("/Users/lbartolome/Documents/GitHub/EWB/data/source/Cordis.json"))
#    json_lst = corpus.get_docs_raw_info()
#    new_list = corpus.get_corpus_SearcheableField_update(["Call"], action="add")
#    fields_dict = corpus.get_corpora_update(1)
