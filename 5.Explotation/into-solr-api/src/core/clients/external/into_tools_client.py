"""
This  module provides 2 classes to handle Into Tools API responses and requests.

The IntoToolsResponse class handles Into Tools API response and errors, while the IntoToolsClient class handles requests to the Into Tools API.

Author: Lorena Calvo-Bartolomé
Date: 15/08/2023
"""

import logging
import os
from urllib.parse import urlencode

import requests
from src.core.clients.external.api_generic.client import Client
from src.core.clients.external.api_generic.response import Response


class IntoToolsResponse(Response):
    """
    A class to handle Into Tools API response and errors.
    """

    def __init__(
        self,
        resp: requests.Response,
        logger: logging.Logger
    ) -> None:

        super().__init__(resp, logger)
        return


class IntoToolsClient(Client):
    """
    A class to handle Into Tools API requests.
    """

    def __init__(
        self,
        logger: logging.Logger,
        timeout: int = 120
    ) -> None:
        """
        Parameters
        ----------
        logger : logging.Logger
            The logger object to log messages and errors.
        timeout : int, optional
            The timeout of the request in seconds, by default 120.
        """

        super().__init__(logger, "Into Tools Client")

        # Get the NP Tools URL from the environment variables
        self.intotools_url = os.environ.get('INTO_TOOLS_URL')
        self.timeout = timeout

        return

    def _do_request(
        self,
        type: str,
        url: str,
        timeout: int = 120,
        **params
    ) -> IntoToolsResponse:
        """Sends a request to the Into Tools API and returns an object of the IntoToolsResponse class.

        Parameters
        ----------
        type : str
            The type of the request.
        url : str
            The URL of the Inferencer API.
        timeout : int, optional
            The timeout of the request in seconds, by default 10.
        **params: dict
            The parameters of the request.

        Returns
        -------
        IntoToolsResponse: IntoToolsResponse
            An object of the IntoToolsResponse class.
        """

        # Send request
        resp = super()._do_request(type, url, timeout, **params)

        # Parse NP Tools response
        inf_resp = IntoToolsResponse(resp, self.logger)

        return inf_resp

    def get_embedding(
        self,
        text_to_embed: str,
        sentence_transformer_model: str,
    ) -> IntoToolsResponse:
        """Get the embedding of a word using the given model.

        Parameters
        ----------
        word_to_embed : str
            The word to embed.
        sentence_transformer_model : str
            Sentence transformer model to be used for embeddings

        Returns
        -------
        IntoToolsResponse: IntoToolsResponse
            An object of the IntoToolsResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        params_ = {
            'text_to_embed': text_to_embed,
            'sentence_transformer_model': sentence_transformer_model,
        }

        encoded_params = urlencode(params_)

        url_ = '{}/embedder/getEmbedding/?{}'.format(
            self.intotools_url, encoded_params)

        self.logger.info(f"-- -- get_embedding - URL: {url_}")

        # Send request to NPtooler
        resp = self._do_request(
            type="get", url=url_, timeout=self.timeout, headers=headers_)

        self.logger.info(f"-- -- get_embedding - Response: {resp}")

        return resp

    def get_lemmas(
        self,
        text_to_lemmatize: str,
        lang: str,
    ) -> IntoToolsResponse:
        """Get the lemmas of a text.

        Parameters
        ----------
        text_to_lemmatize : str
            The word to lemmatize.
        embedding_model : str
        lang : str
            The language of the text to be lemmatized (es/en)

        Returns
        -------
        IntoToolsResponse: IntoToolsResponse
            An object of the IntoToolsResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        params_ = {
            'text_to_lemmatize': text_to_lemmatize,
            'lang': lang
        }

        encoded_params = urlencode(params_)

        url_ = '{}/lemmatizer/getLemmas/?{}'.format(
            self.intotools_url, encoded_params)

        self.logger.info(f"-- -- get_lemmas - URL: {url_}")

        # Send request to NPtooler
        resp = self._do_request(
            type="get", url=url_, timeout=self.timeout, headers=headers_)

        self.logger.info(f"-- -- get_lemmas - Response: {resp}")

        return resp

    def get_thetas(
        self,
        text_to_infer: str,
        model_for_infer: str,
    ) -> IntoToolsResponse:
        """Get the thetas representation for a document based on a given trained topic model. 
        The format of the response from the NP Tools API is as follows:

        {
            "responseHeader": {
                "status": 200,
                "time": 2.7594828605651855
            },
            "response": [
                {
                "id": 0,
                "thetas": "t0|188 t1|244 t2|210 t3|249 t4|109"
                }
            ]
        }

        Parameters
        ----------
        text_to_infer : str
            Text to be inferred.
        model_for_infer : str
            The model to be used for inference.

        Returns
        -------
        IntoToolsResponse: IntoToolsResponse
            An object of the IntoToolsResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        params_ = {
            'text_to_infer': text_to_infer,
            'model_for_infer': model_for_infer,
        }

        encoded_params = urlencode(params_)

        url_ = '{}/inferencer/inferDoc/?{}'.format(
            self.intotools_url, encoded_params)

        self.logger.info(f"-- -- get_thetas - URL: {url_}")

        # Send request to NPtooler
        resp = self._do_request(
            type="get", url=url_, timeout=self.timeout, headers=headers_)

        self.logger.info(f"-- -- get_thetas - Response: {resp}")

        return resp
