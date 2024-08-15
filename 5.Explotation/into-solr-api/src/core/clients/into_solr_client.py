"""
This module provides a specific class for handeling the Solr API responses and requests of the NP-Solr-Service.

Author: Lorena Calvo-Bartolomé
Date: 17/04/2023
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Proyect))
"""

import configparser
import logging
import pathlib
import re
from typing import List, Union

# from src.core.clients.external.np_inferencer_client import XXInferencerClient
from src.core.clients.base.solr_client import SolrClient
from src.core.entities.corpus import Corpus
from src.core.entities.model import Model
from src.core.entities.queries import Queries


class IntoSolrClient(SolrClient):

    def __init__(self,
                 logger: logging.Logger,
                 config_file: str = "/config/config.cf") -> None:
        super().__init__(logger)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.solr_config = "into_config"
        self.batch_size = int(cf.get('restapi', 'batch_size'))
        self.corpus_col = cf.get('restapi', 'corpus_col')
        self.no_meta_fields = cf.get('restapi', 'no_meta_fields').split(",")
        self.no_meta_fields = cf.get('restapi', 'no_meta_fields').split(",")
        self.path_source = pathlib.Path(cf.get('restapi', 'path_source'))
        self.thetas_max_sum = int(cf.get('restapi', 'thetas_max_sum'))
        self.betas_max_sum = int(cf.get('restapi', 'betas_max_sum'))

        # Create Queries object for managing queries
        self.querier = Queries()

        # Create InferencerClient to send requests to the Inferencer API
        # TODO: Uncomment and update if necessary
        # self.inferencer = XXInferencerClient(logger)

        return

    # ======================================================
    # CORPUS-RELATED OPERATIONS
    # ======================================================

    def index_corpus(
        self,
        corpus_raw: str
    ) -> None:
        """
        This method takes the name of a corpus raw file as input. It creates a Solr collection with the stem name of the file, which is obtained by converting the file name to lowercase (for example, if the input is 'Cordis', the stem would be 'cordis'). However, this process occurs only if the directory structure (self.path_source / corpus_raw / parquet) exists.

        After creating the Solr collection, the method reads the corpus file, extracting the raw information of each document. Subsequently, it sends a POST request to the Solr server to index the documents in batches.

        Parameters
        ----------
        corpus_raw : str
            The string name of the corpus raw file to be indexed.

        """

        # 1. Get full path and stem of the logical corpus
        corpus_to_index = self.path_source / (corpus_raw + ".parquet")
        corpus_logical_name = corpus_to_index.stem.lower()

        self.logger.info(f"Corpus to index: {corpus_to_index}")
        self.logger.info(f"Corpus logical name: {corpus_logical_name}")

        # 2. Create collection
        corpus, err = self.create_collection(
            col_name=corpus_logical_name, config=self.solr_config)
        if err == 409:
            self.logger.info(
                f"-- -- Collection {corpus_logical_name} already exists.")
            return
        else:
            self.logger.info(
                f"-- -- Collection {corpus_logical_name} successfully created.")

        # 3. Add corpus collection to self.corpus_col. If Corpora has not been created already, create it
        corpus, err = self.create_collection(
            col_name=self.corpus_col, config=self.solr_config)
        if err == 409:
            self.logger.info(
                f"-- -- Collection {self.corpus_col} already exists.")

            # 3.1. Do query to retrieve last id in self.corpus_col
            # http://localhost:8983/solr/#/{self.corpus_col}/query?q=*:*&q.op=OR&indent=true&sort=id desc&fl=id&rows=1&useParams=
            sc, results = self.execute_query(q='*:*',
                                             col_name=self.corpus_col,
                                             sort="id desc",
                                             rows="1",
                                             fl="id")
            if sc != 200:
                self.logger.error(
                    f"-- -- Error getting latest used ID. Aborting operation...")
                return
            # Increment corpus_id for next corpus to be indexed
            corpus_id = int(results.docs[0]["id"]) + 1
        else:
            self.logger.info(
                f"Collection {self.corpus_col} successfully created.")
            corpus_id = 1

        # 4. Create Corpus object and extract info from the corpus to index
        corpus = Corpus(corpus_to_index)
        json_docs = corpus.get_docs_raw_info()
        corpus_col_upt = corpus.get_corpora_update(id=corpus_id)

        # 5. Index corpus and its fiels in CORPUS_COL
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} info in {self.corpus_col} starts.")
        self.index_documents(corpus_col_upt, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} info in {self.corpus_col} completed.")

        # 6. Index documents in corpus collection
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} in {corpus_logical_name} starts.")
        self.index_documents(json_docs, corpus_logical_name, self.batch_size)
        self.logger.info(
            f"-- -- Indexing of {corpus_logical_name} in {corpus_logical_name} completed.")

        return

    def list_corpus_collections(self) -> Union[List, int]:
        """Returns a list of the names of the corpus collections that have been created in the Solr server.

        Returns
        -------
        corpus_lst: List
            List of the names of the corpus collections that have been created in the Solr server.
        """

        sc, results = self.execute_query(q='*:*',
                                         col_name=self.corpus_col,
                                         fl="corpus_name")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus collections in {self.corpus_col}. Aborting operation...")
            return

        corpus_lst = [doc["corpus_name"] for doc in results.docs]

        return corpus_lst, sc

    def get_corpus_coll_fields(self, corpus_col: str) -> Union[List, int]:
        """Returns a list of the fields of the corpus collection given by 'corpus_col' that have been defined in the Solr server.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose fields are to be retrieved.

        Returns
        -------
        models: list
            List of fields of the corpus collection
        sc: int
            Status code of the request
        """
        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="fields")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting fields of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["fields"], sc

    def get_corpus_raw_path(self, corpus_col: str) -> Union[pathlib.Path, int]:
        """Returns the path of the logical corpus file associated with the corpus collection given by 'corpus_col'.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose path is to be retrieved.

        Returns
        -------
        path: pathlib.Path
            Path of the logical corpus file associated with the corpus collection given by 'corpus_col'.
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="corpus_path")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus path of {corpus_col}. Aborting operation...")
            return

        self.logger.info(results.docs[0]["corpus_path"])
        return pathlib.Path(results.docs[0]["corpus_path"]), sc

    def get_id_corpus_in_corpora(self, corpus_col: str) -> Union[int, int]:
        """Returns the ID of the corpus collection given by 'corpus_col' in the self.corpus_col collection.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose ID is to be retrieved.

        Returns
        -------
        id: int
            ID of the corpus collection given by 'corpus_col' in the self.corpus_col collection.
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="id")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus ID. Aborting operation...")
            return

        return results.docs[0]["id"], sc

    def get_corpus_MetadataDisplayed(self, corpus_col: str) -> Union[List, int]:
        """Returns a list of the fileds of the corpus collection indicating what metadata will be displayed in the NP front upon user request.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose MetadataDisplayed are to be retrieved.
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="MetadataDisplayed")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting MetadataDisplayed of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["MetadataDisplayed"], sc

    def get_corpus_SearcheableField(self, corpus_col: str) -> Union[List, int]:
        """Returns a list of the fields used for autocompletion in the document search in the similarities function and in the document search function.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose SearcheableField are to be retrieved.
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="SearcheableFields")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting SearcheableField of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["SearcheableFields"], sc

    def get_corpus_models(self, corpus_col: str) -> Union[List, int]:
        """Returns a list with the models associated with the corpus given by 'corpus_col'

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection whose models are to be retrieved.

        Returns
        -------
        models: list
            List of models associated with the corpus
        sc: int
            Status code of the request
        """

        sc, results = self.execute_query(q='corpus_name:"'+corpus_col+'"',
                                         col_name=self.corpus_col,
                                         fl="models")

        if sc != 200:
            self.logger.error(
                f"-- -- Error getting models of {corpus_col}. Aborting operation...")
            return

        return results.docs[0]["models"], sc

    def delete_corpus(self,
                      corpus_raw: str) -> None:
        """Given the name of a corpus raw file as input, it deletes the Solr collection associated with it. Additionally, it removes the document entry of the corpus in the self.corpus_col collection and all the models that have been trained with such a corpus.

        Parameters
        ----------
        corpus_raw : str
            The string name of the corpus raw file to be deleted.
        """

        # 1. Get stem of the logical corpus
        corpus_to_delete = self.path_source / (corpus_raw + ".parquet")
        corpus_logical_name = corpus_to_delete.stem.lower()

        # 2. Delete corpus collection
        _, sc = self.delete_collection(col_name=corpus_logical_name)
        if sc != 200:
            self.logger.error(
                f"-- -- Error deleting corpus collection {corpus_logical_name}")
            return

        # 3. Get ID and associated models of corpus collection in self.corpus_col
        sc, results = self.execute_query(q='corpus_name:'+corpus_logical_name,
                                         col_name=self.corpus_col,
                                         fl="id,models")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus ID. Aborting operation...")
            return

        # 4. Delete all models associated with the corpus if any
        if "models" in results.docs[0].keys():
            for model in results.docs[0]["models"]:
                _, sc = self.delete_collection(col_name=model)
                if sc != 200:
                    self.logger.error(
                        f"-- -- Error deleting model collection {model}")
                    return

        # 5. Remove corpus from self.corpus_col
        sc = self.delete_doc_by_id(
            col_name=self.corpus_col, id=results.docs[0]["id"])
        if sc != 200:
            self.logger.error(
                f"-- -- Error deleting corpus from {self.corpus_col}")
        return

    def check_is_corpus(self, corpus_col) -> bool:
        """Checks if the collection given by 'corpus_col' is a corpus collection.

        Parameters
        ----------
        corpus_col : str
            Name of the collection to be checked.

        Returns
        -------
        is_corpus: bool
            True if the collection is a corpus collection, False otherwise.
        """

        corpus_colls, sc = self.list_corpus_collections()
        if corpus_col not in corpus_colls:
            self.logger.error(
                f"-- -- {corpus_col} is not a corpus collection. Aborting operation...")
            return False

        return True

    def check_corpus_has_model(self, corpus_col, model_name) -> bool:
        """Checks if the collection given by 'corpus_col' has a model with name 'model_name'.

        Parameters
        ----------
        corpus_col : str
            Name of the collection to be checked.
        model_name : str
            Name of the model to be checked.

        Returns
        -------
        has_model: bool
            True if the collection has the model, False otherwise.
        """

        corpus_fields, sc = self.get_corpus_coll_fields(corpus_col)
        if 'doctpc_' + model_name not in corpus_fields:
            self.logger.error(
                f"-- -- {corpus_col} does not have the field doctpc_{model_name}. Aborting operation...")
            return False
        return True

    def modify_corpus_SearcheableFields(
        self,
        SearcheableFields: str,
        corpus_col: str,
        action: str
    ) -> None:
        """
        Given a list of fields, it adds them to the SearcheableFields field of the corpus collection given by 'corpus_col' if action is 'add', or it deletes them from the SearcheableFields field of the corpus collection given by 'corpus_col' if action is 'delete'.

        Parameters
        ----------
        SearcheableFields : str
            List of fields to be added to the SearcheableFields field of the corpus collection given by 'corpus_col'.
        corpus_col : str
            Name of the corpus collection whose SearcheableFields field is to be updated.
        action : str
            Action to be performed. It can be 'add' or 'delete'.
        """

        # 1. Get full path
        corpus_path, _ = self.get_corpus_raw_path(corpus_col)

        SearcheableFields = SearcheableFields.split(",")

        # 2. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 3. Create Corpus object, get SearcheableField and index information in corpus collection
        corpus = Corpus(corpus_path)
        corpus_update, new_SearcheableFields = corpus.get_corpus_SearcheableField_update(
            new_SearcheableFields=SearcheableFields,
            action=action)
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {corpus_col} collection")
        self.index_documents(corpus_update, corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {self.corpus_col} completed.")

        # 4. Get self.corpus_col update
        corpora_id, _ = self.get_id_corpus_in_corpora(corpus_col)
        corpora_update = corpus.get_corpora_SearcheableField_update(
            id=corpora_id,
            field_update=new_SearcheableFields,
            action="set")
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {self.corpus_col} starts.")
        self.index_documents(corpora_update, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing new SearcheableField information in {self.corpus_col} completed.")

        return

    # ======================================================
    # MODEL-RELATED OPERATIONS
    # ======================================================

    def index_model(self, model_path: str) -> None:
        """
        Given the string path of a model created with the ITMT (i.e., the name of one of the folders representing a model within the TMmodels folder), it extracts the model information and that of the corpus used for its generation. It then adds a new field in the corpus collection of type 'VectorField' and name 'doctpc_{model_name}, and index the document-topic proportions in it. At last, it index the rest of the model information in the model collection.

        Parameters
        ----------
        model_path : str
            Path to the folder of the model to be indexed.
        """

        # 1. Get stem of the model folder
        model_to_index = self.path_source / model_path
        model_name = model_to_index.stem.lower()
    
        # 2. Create collection
        _, err = self.create_collection(
            col_name=model_name, config=self.solr_config)
        if err == 409:
            self.logger.info(
                f"-- -- Collection {model_name} already exists.")
            return
        else:
            self.logger.info(
                f"-- -- Collection {model_name} successfully created.")

        # 3. Create Model object and extract info from the corpus to index
        model = Model(model_to_index)
        json_docs, corpus_name = model.get_model_info_update(action='set')
        if not self.check_is_corpus(corpus_name):
            return
        corpora_id, _ = self.get_id_corpus_in_corpora(corpus_name)
        field_update = model.get_corpora_model_update(
            id=corpora_id, action='add')

        # 4. Add field for the doc-tpc distribution associated with the model being indexed in the document associated with the corpus
        self.logger.info(
            f"-- -- Indexing model information of {model_name} in {self.corpus_col} starts.")

        self.index_documents(field_update, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Indexing of model information of {model_name} info in {self.corpus_col} completed.")

        # 5. Modify schema in corpus collection to add field for the doc-tpc distribution and the similarities associated with the model being indexed
        model_key = 'doctpc_' + model_name
        sim_model_key = 'sim_' + model_name
        self.logger.info(
            f"-- -- Adding field {model_key} in {corpus_name} collection")
        _, err = self.add_field_to_schema(
            col_name=corpus_name, field_name=model_key, field_type='VectorField')
        self.logger.info(
            f"-- -- Adding field {sim_model_key} in {corpus_name} collection")
        _, err = self.add_field_to_schema(
            col_name=corpus_name, field_name=sim_model_key, field_type='VectorFloatField')

        # 6. Index doc-tpc information in corpus collection
        self.logger.info(
            f"-- -- Indexing model information in {corpus_name} collection")
        self.index_documents(json_docs, corpus_name, self.batch_size)

        self.logger.info(
            f"-- -- Indexing model information in {model_name} collection")
        json_tpcs = model.get_model_info()

        self.index_documents(json_tpcs, model_name, self.batch_size)

        return

    def list_model_collections(self) -> Union[List[str], int]:
        """Returns a list of the names of the model collections that have been created in the Solr server.

        Returns
        -------
        models_lst: List[str]
            List of the names of the model collections that have been created in the Solr server.
        sc: int
            Status code of the request.
        """
        sc, results = self.execute_query(q='*:*',
                                         col_name=self.corpus_col,
                                         fl="models")
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting corpus collections in {self.corpus_col}. Aborting operation...")
            return

        models_lst = [model for doc in results.docs if bool(
            doc) for model in doc["models"]]
        self.logger.info(f"-- -- Models found: {models_lst}")

        return models_lst, sc

    def delete_model(self, model_path: str) -> None:
        """
        Given the string path of a model created with the ITMT (i.e., the name of one of the folders representing a model within the TMmodels folder), 
        it deletes the model collection associated with it. Additionally, it removes the document-topic proportions field in the corpus collection and removes the fields associated with the model and the model from the list of models in the corpus document from the self.corpus_col collection.

        Parameters
        ----------
        model_path : str
            Path to the folder of the model to be indexed.
        """

        # 1. Get stem of the model folder
        model_to_index = pathlib.Path(model_path)
        model_name = pathlib.Path(model_to_index).stem.lower()

        # 2. Delete model collection
        _, sc = self.delete_collection(col_name=model_name)
        if sc != 200:
            self.logger.error(
                f"-- -- Error occurred while deleting model collection {model_name}. Stopping...")
            return
        else:
            self.logger.info(
                f"-- -- Model collection {model_name} successfully deleted.")

        # 3. Create Model object and extract info from the corpus associated with the model
        model = Model(model_to_index)
        json_docs, corpus_name = model.get_model_info_update(action='remove')
        sc, results = self.execute_query(q='corpus_name:'+corpus_name,
                                         col_name=self.corpus_col,
                                         fl="id")
        if sc != 200:
            self.logger.error(
                f"-- -- Corpus collection not found in {self.corpus_col}")
            return
        field_update = model.get_corpora_model_update(
            id=results.docs[0]["id"], action='remove')

        # 4. Remove field for the doc-tpc distribution associated with the model being deleted in the document associated with the corpus
        self.logger.info(
            f"-- -- Deleting model information of {model_name} in {self.corpus_col} starts.")
        self.index_documents(field_update, self.corpus_col, self.batch_size)
        self.logger.info(
            f"-- -- Deleting model information of {model_name} info in {self.corpus_col} completed.")

        # 5. Delete doc-tpc information from corpus collection
        self.logger.info(
            f"-- -- Deleting model information from {corpus_name} collection")
        self.index_documents(json_docs, corpus_name, self.batch_size)

        # 6. Modify schema in corpus collection to delete field for the doc-tpc distribution and similarities associated with the model being indexed
        model_key = 'doctpc_' + model_name
        sim_model_key = 'sim_' + model_name
        self.logger.info(
            f"-- -- Deleting field {model_key} in {corpus_name} collection")
        _, err = self.delete_field_from_schema(
            col_name=corpus_name, field_name=model_key)
        self.logger.info(
            f"-- -- Deleting field {sim_model_key} in {corpus_name} collection")
        _, err = self.delete_field_from_schema(
            col_name=corpus_name, field_name=sim_model_key)

        return

    def check_is_model(self, model_col) -> bool:
        """Checks if the model_col is a model collection. If not, it aborts the operation.

        Parameters
        ----------
        model_col : str
            Name of the model collection.

        Returns
        -------
        is_model : bool
            True if the model_col is a model collection, False otherwise.
        """

        model_colls, sc = self.list_model_collections()
        if model_col not in model_colls:
            self.logger.error(
                f"-- -- {model_col} is not a model collection. Aborting operation...")
            return False
        return True

    # ======================================================
    # AUXILIARY FUNCTIONS
    # ======================================================
    def custom_start_and_rows(self, start, rows, col) -> Union[str, str]:
        """Checks if start and rows are None. If so, it returns the number of documents in the collection as the value for rows and 0 as the value for start.

        Parameters
        ----------
        start : str
            Start parameter of the query.
        rows : str
            Rows parameter of the query.
        col : str
            Name of the collection.

        Returns
        -------
        start : str
            Final start parameter of the query.
        rows : str
            Final rows parameter of the query.
        """
        if start is None:
            start = str(0)
        if rows is None:
            numFound_dict, sc = self.do_Q3(col)
            rows = str(numFound_dict['ndocs'])

            if sc != 200:
                self.logger.error(
                    f"-- -- Error executing query Q3. Aborting operation...")
                return

        return start, rows

    # ======================================================
    # QUERIES
    # ======================================================

    def do_Q1(self,
              corpus_col: str,
              doc_id: str,
              model_name: str) -> Union[dict, int]:
        """Executes query Q1.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection.
        id : str
            ID of the document to be retrieved.
        model_name : str
            Name of the model to be used for the retrieval.

        Returns
        -------
        thetas: dict
            JSON object with the document-topic proportions (thetas)
        sc : int
            The status code of the response.  
        """

        # 0. Convert corpus and model names to lowercase
        corpus_col = corpus_col.lower()
        model_name = model_name.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Check that corpus_col has the model_name field
        if not self.check_corpus_has_model(corpus_col, model_name):
            return

        # 3. Execute query
        q1 = self.querier.customize_Q1(id=doc_id, model_name=model_name)
        params = {k: v for k, v in q1.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q1['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q1. Aborting operation...")
            return

        # 4. Return -1 if thetas field is not found (it could happen that a document in a collection has not thetas representation since it was not keeped within the corpus used for training the model)
        if 'doctpc_' + model_name in results.docs[0].keys():
            resp = {'thetas': results.docs[0]['doctpc_' + model_name]}
        else:
            resp = {'thetas': -1}

        return resp, sc

    def do_Q2(self, corpus_col: str) -> Union[dict, int]:
        """
        Executes query Q2.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection

        Returns
        -------
        json_object: dict
            JSON object with the metadata fields of the corpus collection in the form: {'metadata_fields': [field1, field2, ...]}
        sc: int
            The status code of the response
        """

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Execute query (to self.corpus_col)
        q2 = self.querier.customize_Q2(corpus_name=corpus_col)
        params = {k: v for k, v in q2.items() if k != 'q'}
        sc, results = self.execute_query(
            q=q2['q'], col_name=self.corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q2. Aborting operation...")
            return

        # 3. Get Metadatadisplayed fields of corpus_col
        Metadatadisplayed, sc = self.get_corpus_MetadataDisplayed(corpus_col)
        if sc != 200:
            self.logger.error(
                f"-- -- Error getting Metadatadisplayed of {corpus_col}. Aborting operation...")
            return

        # 4. Filter metadata fields to be displayed in the NP
        # meta_fields = [field for field in results.docs[0]
        #               ['fields'] if field in Metadatadisplayed]

        return {'metadata_fields': Metadatadisplayed}, sc

    def do_Q3(self, col: str) -> Union[dict, int]:
        """Executes query Q3.

        Parameters
        ----------
        col : str
            Name of the collection

        Returns
        -------
        json_object : dict
            JSON object with the number of documents in the corpus collection
        sc : int
            The status code of the response
        """

        # 0. Convert collection name to lowercase
        col = col.lower()

        # 1. Check that col is either a corpus or a model collection
        if not self.check_is_corpus(col) and not self.check_is_model(col):
            return

        # 2. Execute query
        q3 = self.querier.customize_Q3()
        params = {k: v for k, v in q3.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q3['q'], col_name=col, **params)

        # 3. Filter results
        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q3. Aborting operation...")
            return

        return {'ndocs': int(results.hits)}, sc

    # def do_Q5(self,
    #           corpus_col: str,
    #           model_name: str,
    #           doc_id: str,
    #           start: str,
    #           rows: str) -> Union[dict, int]:
    #     """Executes query Q5.

    #     Parameters
    #     ----------
    #     corpus_col : str
    #         Name of the corpus collection
    #     model_name: str
    #         Name of the model to be used for the retrieval
    #     doc_id: str
    #         ID of the document whose similarity is going to be checked against all other documents in 'corpus_col'
    #      start: str
    #         Offset into the responses at which Solr should begin displaying content
    #     rows: str
    #         How many rows of responses are displayed at a time

    #     Returns
    #     -------
    #     json_object: dict
    #         JSON object with the results of the query.
    #     sc : int
    #         The status code of the response.
    #     """

    #     # 0. Convert corpus and model names to lowercase
    #     corpus_col = corpus_col.lower()
    #     model_name = model_name.lower()

    #     # 1. Check that corpus_col is indeed a corpus collection
    #     if not self.check_is_corpus(corpus_col):
    #         return

    #     # 2. Check that corpus_col has the model_name field
    #     if not self.check_corpus_has_model(corpus_col, model_name):
    #         return

    #     # 3. Execute Q1 to get thetas of document given by doc_id
    #     thetas_dict, sc = self.do_Q1(
    #         corpus_col=corpus_col, model_name=model_name, doc_id=doc_id)
    #     thetas = thetas_dict['thetas']

    #     # 4. Check that thetas are available on the document given by doc_id. If not, infer them
    #     if thetas == -1:
    #         # Get text (lemmas) of the document so its thetas can be inferred
    #         lemmas_dict, sc = self.do_Q15(
    #             corpus_col=corpus_col, doc_id=doc_id)
    #         lemmas = lemmas_dict['lemmas']

    #         inf_resp = self.inferencer.infer_doc(text_to_infer=lemmas,
    #                                              model_for_inference=model_name)
    #         if inf_resp.status_code != 200:
    #             self.logger.error(
    #                 f"-- -- Error attaining thetas from {lemmas} while executing query Q5. Aborting operation...")
    #             return

    #         thetas = inf_resp.results[0]['thetas']
    #         self.logger.info(
    #             f"-- -- Thetas attained in {inf_resp.time} seconds: {thetas}")

    #     # 4. Customize start and rows
    #     start, rows = self.custom_start_and_rows(start, rows, corpus_col)

    #     # 5. Execute query
    #     q5 = self.querier.customize_Q5(
    #         model_name=model_name, thetas=thetas,
    #         start=start, rows=rows)
    #     params = {k: v for k, v in q5.items() if k != 'q'}

    #     sc, results = self.execute_query(
    #         q=q5['q'], col_name=corpus_col, **params)

    #     if sc != 200:
    #         self.logger.error(
    #             f"-- -- Error executing query Q5. Aborting operation...")
    #         return

    #     # 6. Normalize scores
    #     for el in results.docs:
    #         el['score'] *= (100/(self.thetas_max_sum ^ 2))

    #     return results.docs, sc

    def do_Q6(self,
              corpus_col: str,
              doc_id: str) -> Union[dict, int]:
        """Executes query Q6.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection
        doc_id: str
            ID of the document whose metadata is going to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Get meta fields
        meta_fields_dict, sc = self.do_Q2(corpus_col)
        meta_fields = ','.join(meta_fields_dict['metadata_fields'])

        self.logger.info("-- -- These are the meta fields: " + meta_fields)

        # 3. Execute query
        q6 = self.querier.customize_Q6(id=doc_id, meta_fields=meta_fields)
        params = {k: v for k, v in q6.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q6['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q6. Aborting operation...")
            return

        return results.docs, sc

    def do_Q7(self,
              corpus_col: str,
              string: str,
              start: str = None,
              rows: str = None) -> Union[dict, int]:
        """Executes query Q7.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection
        string: str
            String to be searched in the title of the documents

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Get number of docs in the collection (it will be the maximum number of docs to be retireved) if rows is not specified
        if rows is None:
            q3 = self.querier.customize_Q3()
            params = {k: v for k, v in q3.items() if k != 'q'}

            sc, results = self.execute_query(
                q=q3['q'], col_name=corpus_col, **params)

            if sc != 200:
                self.logger.error(
                    f"-- -- Error executing query Q3. Aborting operation...")
                return
            rows = results.hits
        if start is None:
            start = str(0)

        # 2. Execute query
        q7 = self.querier.customize_Q7(
            title_field='SearcheableField',
            string=string,
            start=start,
            rows=rows)
        params = {k: v for k, v in q7.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q7['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q7. Aborting operation...")
            return

        return results.docs, sc

    def do_Q8(self,
              model_col: str,
              start: str,
              rows: str) -> Union[dict, int]:
        """Executes query Q8.

        Parameters
        ----------
        model_col: str
            Name of the model collection
        start: str
            Index of the first document to be retrieved
        rows: str
            Number of documents to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert model name to lowercase
        model_col = model_col.lower()

        # 1. Check that model_col is indeed a model collection
        if not self.check_is_model(model_col):
            return

        # 3. Customize start and rows
        start, rows = self.custom_start_and_rows(start, rows, model_col)

        # 4. Execute query
        q8 = self.querier.customize_Q8(start=start, rows=rows)
        params = {k: v for k, v in q8.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q8['q'], col_name=model_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q8. Aborting operation...")
            return

        return results.docs, sc

    def do_Q9(self,
              corpus_col: str,
              model_name: str,
              topic_id: str,
              start: str,
              rows: str) -> Union[dict, int]:
        """Executes query Q9.

        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection on which the query will be carried out
        model_name: str
            Name of the model collection on which the search will be based
        topic_id: str
            ID of the topic whose top-documents will be retrieved
        start: str
            Index of the first document to be retrieved
        rows: str
            Number of documents to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert corpus and model names to lowercase
        corpus_col = corpus_col.lower()
        model_name = model_name.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Check that corpus_col has the model_name field
        if not self.check_corpus_has_model(corpus_col, model_name):
            return

        # 3. Customize start and rows
        start, rows = self.custom_start_and_rows(start, rows, corpus_col)
        # We limit the maximum number of results since they are top-documnts
        # If more results are needed pagination should be used

        if int(rows) > 100:
            rows = "100"

        # 5. Execute query
        q9 = self.querier.customize_Q9(
            model_name=model_name,
            topic_id=topic_id,
            start=start,
            rows=rows)
        params = {k: v for k, v in q9.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q9['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q9. Aborting operation...")
            return

        # 6. Return a dictionary with names more understandable to the end user
        proportion_key = "payload(doctpc_{},t{})".format(model_name, topic_id)
        for dict in results.docs:
            if proportion_key in dict.keys():
                dict["topic_relevance"] = dict.pop(proportion_key)*0.1
            dict["num_words_per_doc"] = dict.pop("nwords_per_doc")

        # 7. Get the topic's top words
        words, sc = self.do_Q10(
            model_col=model_name,
            start=topic_id,
            rows=1,
            only_id=False)
        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q10 when using in Q9. Aborting operation...")
            return

        dict_bow, sc = self.do_Q18(
            corpus_col=corpus_col,
            ids=",".join([d['id'] for d in results.docs]),
            words=",".join(words[0]['tpc_descriptions'].split(", ")),
            start=start,
            rows=rows)

        # 7. Merge results
        def replace_payload_keys(dictionary):
            new_dict = {}
            for key, value in dictionary.items():
                match = re.match(r'payload\(bow,(\w+)\)', key)
                if match:
                    new_key = match.group(1)
                else:
                    new_key = key
                new_dict[new_key] = value
            return new_dict

        merged_tpcs = []
        for d1 in results.docs:
            id_value = d1['id']

            # Find the corresponding dictionary in dict2
            d2 = next(item for item in dict_bow if item["id"] == id_value)

            new_dict = {
                "id": id_value,
                "topic_relevance": d1.get("topic_relevance", 0),
                "num_words_per_doc": d1.get("num_words_per_doc", 0),
                "counts": replace_payload_keys({key: d2[key] for key in d2 if key.startswith("payload(bow,")})
            }

            merged_tpcs.append(new_dict)

        return merged_tpcs, sc

    def do_Q10(self,
               model_col: str,
               start: str,
               rows: str,
               only_id: bool) -> Union[dict, int]:
        """Executes query Q10.

        Parameters
        ----------
        model_col: str
            Name of the model collection whose information is being retrieved
        start: str
            Index of the first document to be retrieved
        rows: str
            Number of documents to be retrieved

        Returns
        -------
        json_object: dict
            JSON object with the results of the query.
        sc : int
            The status code of the response.
        """

        # 0. Convert model name to lowercase
        model_col = model_col.lower()

        # 1. Check that model_col is indeed a model collection
        if not self.check_is_model(model_col):
            return

        # 3. Customize start and rows
        start, rows = self.custom_start_and_rows(start, rows, model_col)

        # 4. Execute query
        q10 = self.querier.customize_Q10(
            start=start, rows=rows, only_id=only_id)
        params = {k: v for k, v in q10.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q10['q'], col_name=model_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q10. Aborting operation...")
            return

        return results.docs, sc

    # def do_Q14(self,
    #            corpus_col: str,
    #            model_name: str,
    #            text_to_infer: str,
    #            start: str,
    #            rows: str) -> Union[dict, int]:
    #     """Executes query Q14.

    #     Parameters
    #     ----------
    #     corpus_col : str
    #         Name of the corpus collection
    #     model_name: str
    #         Name of the model to be used for the retrieval
    #     text_to_infer: str
    #         Text to be inferred
    #      start: str
    #         Offset into the responses at which Solr should begin displaying content
    #     rows: str
    #         How many rows of responses are displayed at a time

    #     Returns
    #     -------
    #     json_object: dict
    #         JSON object with the results of the query.
    #     sc : int
    #         The status code of the response.
    #     """

    #     # 0. Convert corpus and model names to lowercase
    #     corpus_col = corpus_col.lower()
    #     model_name = model_name.lower()

    #     # 1. Check that corpus_col is indeed a corpus collection
    #     if not self.check_is_corpus(corpus_col):
    #         return

    #     # 2. Check that corpus_col has the model_name field
    #     if not self.check_corpus_has_model(corpus_col, model_name):
    #         return

    #     # 3. Make request to Inferencer API to get thetas of text_to_infer
    #     inf_resp = self.inferencer.infer_doc(text_to_infer=text_to_infer,
    #                                          model_for_inference=model_name)
    #     if inf_resp.status_code != 200:
    #         self.logger.error(
    #             f"-- -- Error attaining thetas from {text_to_infer} while executing query Q14. Aborting operation...")
    #         return

    #     thetas = inf_resp.results[0]['thetas']
    #     self.logger.info(
    #         f"-- -- Thetas attained in {inf_resp.time} seconds: {thetas}")

    #     # 4. Customize start and rows
    #     start, rows = self.custom_start_and_rows(start, rows, corpus_col)

    #     # 5. Execute query
    #     q14 = self.querier.customize_Q14(
    #         model_name=model_name, thetas=thetas,
    #         start=start, rows=rows)
    #     params = {k: v for k, v in q14.items() if k != 'q'}

    #     sc, results = self.execute_query(
    #         q=q14['q'], col_name=corpus_col, **params)

    #     if sc != 200:
    #         self.logger.error(
    #             f"-- -- Error executing query Q14. Aborting operation...")
    #         return

    #     # 6. Normalize scores
    #     for el in results.docs:
    #         el['score'] *= (100/(self.thetas_max_sum ^ 2))

    #     return results.docs, sc

    def do_Q15(self,
               corpus_col: str,
               doc_id: str) -> Union[dict, int]:
        """Executes query Q15.

        Parameters
        ----------
        corpus_col : str
            Name of the corpus collection.
        id : str
            ID of the document to be retrieved.

        Returns
        -------
        lemmas: dict
            JSON object with the document's lemmas.
        sc : int
            The status code of the response.  
        """

        # 0. Convert corpus and model names to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Execute query
        q15 = self.querier.customize_Q15(id=doc_id)
        params = {k: v for k, v in q15.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q15['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q15. Aborting operation...")
            return

        return {'lemmas': results.docs[0]['lemmas']}, sc

    def do_Q18(self,
               corpus_col: str,
               ids: str,
               words: str,
               start: str,
               rows: str
               ) -> Union[dict, int]:

        # 0. Convert corpus name to lowercase
        corpus_col = corpus_col.lower()

        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return

        # 2. Execute query
        start, rows = self.custom_start_and_rows(start, rows, corpus_col)

        q18 = self.querier.customize_Q18(
            ids=ids.split(","),
            words=words.split(","),
            start=start,
            rows=rows)
        params = {k: v for k, v in q18.items() if k != 'q'}

        sc, results = self.execute_query(
            q=q18['q'], col_name=corpus_col, **params)

        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q18. Aborting operation...")
            return

        return results.docs, sc
    
    def do_Q21(
        self,
        corpus_col:str,
        search_doc:str,
        start:int,
        rows:int,
        embedding_model:str = "word2vec",
        lang:str = "es",
    ) -> Union[dict,int]:
        """
        Executes query Q21.
        
        Parameters
        ----------
        corpus_col: str
            Name of the corpus collection
        search_doc: str
            Document to look documents similar to
        start: int
            Index of the first document to be retrieved
        rows: int
            Number of documents to be retrieved
        embedding_model: str
            Name of the embedding model to be used
        lang: str
            Language of the text to be embedded
        
        Returns
        -------
        response: dict
            JSON object with the results of the query.
        """
        
        # 0. Convert corpus to lowercase
        corpus_col = corpus_col.lower()
        
        # 1. Check that corpus_col is indeed a corpus collection
        if not self.check_is_corpus(corpus_col):
            return
        
        """
        # 3. Get embedding from search_doc
        resp = self.nptooler.get_embedding(
            text_to_embed=search_doc,
            embedding_model=embedding_model,
            lang=lang
        )
        
        if resp.status_code != 200:
            self.logger.error(
                f"-- -- Error attaining embeddings from {search_doc} while executing query Q21. Aborting operation...")
            return

        embs = resp.results
        """
        
        embs = [0.43476945,0.016387263,0.26944,-0.2268373,-0.50270677,0.26392245,-0.76436365,0.29373062,0.43987417,-0.13844043,-0.018064413,-0.5541143,-0.24966642,-0.22129494,0.10368134,-0.8268186,0.072215736,0.1628627,0.95777524,0.15325502,0.86426353,-0.6612361,-0.82884026,-1.3789778,-0.6690669,0.37577382,-0.40974313,0.60106015,0.029740669,-0.21773168,0.008427236,0.27177694,-0.27198428,-0.34045005,-0.30714673,0.4375183,0.12257917,-0.16347873,-0.17768495,-0.094940975,0.68592584,-0.25744003,-0.085299775,0.5492455,-0.6254935,0.13471587,-0.9328081,0.0795569,0.49563593,-0.11565469,-0.10682628,-0.37435934,0.5270455,0.4203443,0.29736355,0.31792605,0.07224808,-0.3982504,-0.28418943,0.74440634,0.6940946,0.13152888,-0.154606,0.38162568,0.16586113,-0.7705116,-0.20912956,0.07706675,-0.6236154,-0.1824399,-0.07046993,0.011883747,-0.21902907,-0.1388048,-0.09443099,0.93165386,-0.5125291,0.7912348,-0.2719726,-0.3354356,-0.2593685,0.012113055,-0.17670462,-0.1961541,0.08315842,1.1916754,-0.037811838,-0.81978405,0.08592582,-0.09634283,0.1281497,-0.18085983,-0.33547947,-0.63516825,0.5753385,-0.32937384,-0.61489713,-0.58786446,-0.30333793,-0.31342417,0.19234352,-0.21404403,1.2052969,0.22266823,0.22841473,1.2955854,-0.54786265,-0.34811413,0.1540088,0.031869538,-0.21781018,0.4840439,-0.27591628,0.15726987,-0.055472735,-0.07087238,0.20219386,0.33977675,-0.07540169,0.24387443,0.7058482,0.010319335,-0.25955436,0.057796143,-1.091804,-0.15660603,0.32575056,0.44894904,0.09531232,0.41339725,-0.061400414,0.04649459,-0.19821845,0.065326735,-0.57589775,0.024462078,-0.5048993,-0.10955325,-0.3043058,0.3452477,0.55761456,0.46121892,0.3762671,-0.19743529,-0.18866003,0.013127988,0.39129335,-0.08001162,-0.08923894,-0.27696207,0.77383405,0.26956192,0.7499522,-0.39616132,-0.1901207,0.6245706,-0.20131928,-0.1036204,0.26158422,-0.1528708,-0.36101368,-0.37913114,0.15107998,-0.4434616,1.0626075,-0.35628563,-0.10912636,0.6204656,-0.31436607,0.3446509,0.48604077,-0.15894675,-0.5364556,0.23938906,0.40352175,0.1542047,0.016308568,-0.0059034256,-0.5017843,0.10039508,-0.3990124,0.1701152,0.5166876,0.37650633,0.71688366,0.05114093,0.40246606,-0.46833116,-0.13663109,-0.8844573,0.8859121,-0.623087,-0.57919824,0.18081217,0.18413055,-0.50423884,-0.15212613,-0.14969614,0.2558053,0.08067332,0.079981685,-0.42943022,-0.9292731,-0.18931393,0.23684402,-0.36053467,-0.54400486,-0.3226445,0.030402206,-0.7058517,-0.39969447,0.3174817,0.099203184,0.05559642,0.2696942,1.0002756,0.46431842,-0.3712722,0.18469542,-1.2297863,-0.15573552,-0.22805443,0.4464478,-0.017383419,-0.15833169,-0.331337,-0.25421172,-0.3146153,-0.014839137,-0.14538512,0.005074834,-0.23897263,-0.43153587,-0.093488365,-0.025423184,0.017690048,-0.020552106,0.5249591,0.5101021,-0.35903162,-0.16362873,0.50887334,0.37910074,0.33389476,0.4388084,0.9146589,-0.6489698,-0.6960696,0.23283997,-0.43232608,0.33486655,-0.12995529,-0.041123614,0.19853161,0.09891914,-0.39984015,0.44520766,0.7606907,0.6399169,-0.2532633,0.66281116,-0.21473381,0.18151587,0.087683074,0.4287957,-0.42792386,-0.4740776,-0.2043713,-0.71792126,-0.8329998,0.1944973,-0.26702505,0.17474014,0.32408834,-0.09434943,-0.23812869,0.44927162,0.29112008,0.9031879,0.46045762,0.40257907,-0.59744585,-0.16057903,-1.0420835,-0.031324066,-0.29866064,0.28506696,0.06481971,-0.69456136,-0.2252677,-0.21845661,-0.3384555,-0.4792,0.47628266,0.4245689,0.2746709,-0.2228963,-0.51091385,0.102849334,0.36325485,0.0023932308,0.16552252,0.2481147,-0.81691706,-0.19609737,0.08790246,-0.07660289,0.08179385,-0.67164725,-0.23683557,-0.16097635,0.010593624,-0.35995078,0.0022905106,0.27630058,0.1724369,0.4921047,0.11745614,-0.4863417,0.25910622,0.38039485,-0.014505591,-0.2872815,0.26453918,0.18785831,0.09406737,-0.5314545,0.40368176,0.036592282,0.36022702,1.3392574,-0.04948273,0.22686109,-0.64379585,-0.021065932,-0.41318464,-0.17245626,0.25454065,-0.33352157,0.62226796,0.05168508,0.070139945,4.307152E-4,-0.48346096,-0.26239455,-0.2964214,-0.2401578,-0.9948127,-0.038084015,-0.23819686,-0.12005405,1.2277222,-0.46933728,-0.13521941,0.5098553,-0.45863792,0.237674,-0.18924654,0.67202294,0.31427044,-0.77004313,0.104095474,0.41136035,-0.057235647,-0.31106317,-0.57346815,-0.19944073,-0.7568532,-0.4074218,-0.2926014,-0.073944926,0.53106856,0.017895944,0.833481,0.13960493,-0.53015673,0.31992012,0.23640999,0.09054877,-0.45666718,0.32025203,-0.62229514,0.18915124,-0.030949965,-0.14568248,0.21674013,0.3271467,0.5125451,-0.22645622,0.15173616,0.4337574,0.21211304,0.1688275,-0.25850597,-0.22339244,-0.19802782,-0.29184765,-0.4591317,-0.5479063,0.17056498,-0.03894631,-0.16373764,0.068230495,0.88943255,0.11809969,0.6359954,-0.34515542,-0.3261984,0.027425118,0.17584924,0.05965006,0.470124,-0.37717104,-0.55277234,0.46376443,-0.5782989,0.68646157,-0.03721615,0.44419628,-0.683365,-0.49516284,-0.260983,0.26827693,-0.43865085,0.008115292,-0.4417026,-0.109316885,-0.15984787,-0.18982442,-0.24572311,0.59582996,-0.17217314,-0.72846967,-0.5981007,-0.013497455,0.32840687,-0.037008822,-0.032524638,-0.280818,-0.30081737,-0.4275377,0.8407314,0.3882762,-0.5353317,-0.09821905,-0.09841338,0.5482911,0.19900095,0.088589,-0.03202291,-1.0707476,-0.15882626,0.14891073,0.16772616,-0.24609709,0.1410357,-0.10391743,0.65835917,-0.26518482,0.26503435,0.17074043,0.35352844,0.027732845,0.12461681,0.41809165,-0.4648223,0.667137,-0.5955262,-0.28724164,1.0423768,0.74886334,-0.6498284,-0.2999231,-0.13276163,0.34551734,-0.6454793,0.018725723,0.11422047,-0.4879559,0.2554816,-0.062998794,-0.050702117,0.54692996,-0.59967756,0.17228207,0.46537572,-0.30208862,-0.21315522,-0.62959987,0.14654629,-0.33027565,-0.3829431,0.092909984,0.31598163,-0.047342908,-0.08738011,0.062552184,-0.33954248,-0.04640117,0.4253206,0.47196043,0.22599107,0.002361605,0.78665996,-0.4526068,-0.050543956,-0.07817449,0.2504223,-0.2726627,0.14161652,-0.045840874,0.19497253,0.027848106,-0.1023411,0.27602413,-0.49598914,-0.15787357,0.2738722,-0.9189128,0.024710856,-0.12067303,-0.07471655,0.094023556,-0.6095124,0.1978234,-0.042016964,-0.3246808,0.40458652,-0.23516291,-0.14059949,0.92340755,0.15483956,-0.1887632,-0.25267848,1.0051613,-0.36225653,0.2657023,-0.056754112,0.71353126,0.5310824,-0.22840638,0.55411047,0.66635436,0.18606652,-0.3616059,-0.5203337,-0.46922383,0.03581067,0.86772674,0.3507317,-0.039706387,2.0952857,-0.08578525,0.0990981,0.15682231,-0.15139842,0.3352148,0.038103744,0.2675766,0.2913952,0.009223325,0.57994395,0.35208756,0.5407821,0.29627532,0.20166017,-0.52270895,-0.78470635,-0.65131545,0.123898365,0.4443702,0.08111531,0.40249532,-0.24629378,-0.45405793,0.24747303,0.31210202,-0.6077901,-0.4970783,0.21323192,-0.04014772,-0.33190185,0.14649951,-0.03694479,-0.19544904,0.09040105,0.18658146,-0.220358,0.24275471,-0.019446831,-0.37889177,-0.38558978,-0.46424717,0.13132657,-0.045324925,0.5798052,0.0051814597,0.4795247,0.06919734,0.3757192,0.16846883,0.28751707,0.10501699,0.43713674,0.24462485,0.6967442,-0.14819083,0.6774105,-0.16408545,0.43124264,-0.4997476,2.90964E-5,-0.16829139,0.6439614,-0.122325465,0.6336666,0.44537655,0.5778942,-0.8776643,0.13483605,-0.022481192,-0.41440365,-0.91466105,-0.1194299,-0.06316508,0.19786614,0.48397124,-0.48172802,-0.15886396,0.11199583,-0.5998373,0.20699345,0.4936974,0.0314122,0.17471224,-0.29542482,-0.6983849,-0.46343508,-0.26897597,-0.15754482,0.0102432985,0.21584362,-0.07632518,-0.17270605,-0.63237053,0.5108812,0.05648468,-0.3123486,-0.5686965,0.30774665,-0.05534008,0.59001184,-0.5606506,0.17384169,0.8138527,-0.2828868,0.08381078,0.8541508,-0.13381037,0.7212215,-0.30679977,0.12201821,0.12863223,-0.14330302,-0.30478722,-0.53687227,-0.26475465,0.34265178,0.23737629,0.20600241,0.03483468,-0.105352275,0.23096353,0.15284327,0.4903605,0.15921843,-0.29265612,-0.051954385,0.43325022,0.49880248,-0.29798084,0.06885228,0.7008615,-0.006458057,-0.35160583,-0.554354,0.5085587,0.40038705,-0.07592892,-0.08410908,-0.3054403,0.17877477,-0.11363028,0.21559891,0.75580895,0.06331884,0.5191567,-0.19934198,-0.04455646,1.0126762,-0.13862564,0.32735747,-0.01710283,-0.056295976,0.113857925,-0.08879856,0.0044269306,0.027867423,0.05719342,-0.26818413,-0.015502917,0.15371937,-0.24038656,-0.16720691,0.15865159,-0.7591585,0.07048313,-0.13499051,-0.41905904,0.83811474,0.16635947,0.28487283,-0.025252242,-0.11290101,-0.4997291,0.026326604,0.6840597,-0.04482829,-1.1106675,0.4271162,0.3766631,-0.057563905,0.27533716,0.43993962,0.8408988,-0.21045989,-0.28558987,-0.6527756,0.05009932,0.3789838,-0.65734893,-0.28299218,0.31316176,-0.25827748,-0.35537633,0.25452107,-0.15735385,0.63725257,0.32893524,-1.1806483,0.22561467,0.5584551,-0.28253236,0.04497099,0.36017838,0.15038355,-0.3078425,0.36355227,1.0093014,-0.91262794,-0.18646196,0.06201566,-0.36294127,-0.2920443,0.15141901,-0.3274223,-0.31966445,-0.13620177]
        
        #self.logger.info(
        #    f"-- -- Embbedings for doc {search_doc} attained in {resp.time} seconds: {embs}")
        
        #with open("/data/source/embs.txt", 'w') as file:
        #    file.write(embs)
         
        # 4. Customize start and rows
        start, rows = self.custom_start_and_rows(start, rows, corpus_col)
        
        # 5. Calculate cosine similarity between the embedding of search_doc and the embeddings of the documents in the corpus
        #distance = "cosine"

        # 5. Execute query
        q21 = self.querier.customize_Q21(
            doc_embeddings=embs,
            #distance=distance,
            start=start,
            rows=rows
        )
        params = {k: v for k, v in q21.items() if k != 'query'}
        json_body = {k: v for k, v in q21.items() if k == 'query'}
        
        self.logger.info(f"-- -- Query Q21 params: {params}")
        self.logger.info(f"-- -- Query Q21 json_body: {json_body}")
        

        # Assuming q21['q'] contains the query string and params contains other Solr params
        sc, results = self.execute_query(
            q=params,
            col_name=corpus_col,
            type="post",
            json_body=json_body
        )
        if sc != 200:
            self.logger.error(
                f"-- -- Error executing query Q21. Aborting operation...")
            return

        return results.docs, sc
