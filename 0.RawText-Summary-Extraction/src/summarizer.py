"""
Class to summarize a PDF file using Llama Index and OpenAI API / Open Source models.

Author: Lorena Calvo-Bartolomé
Date: 04/02/2024
"""

import logging
import os
import pathlib
from dotenv import load_dotenv
from llama_index.core.schema import Document
from llama_index.core import DocumentSummaryIndex, VectorStoreIndex
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext
from src.utils import messages_to_prompt, completion_to_prompt


class Summarizer(object):
    def __init__(
        self,
        model="HuggingFaceH4/zephyr-7b-beta",
        temperature: float = 0,
        chunk_size: int = 1024,
        instructions: str = None
    ) -> None:

        path_env = pathlib.Path(os.getcwd()).parent / '.env'
        load_dotenv(path_env)
        self._api_key = os.getenv("OPENAI_API_KEY")

        logging.basicConfig(level='INFO')
        self._logger = logging.getLogger('PDFParser')

        if instructions:
            self._instructions = instructions
        else:
            self._instructions = \
                """You are a helpful AI assistant working with the generation of summaries of PDF documents. Please summarize the given document by sections in such a way that the outputted text can be used as input for a topic modeling algorithm. Dont start with 'The document can be summarized...' or 'The document is about...'. Just start with the first section of the document.
            """

        if model.startswith("gpt"):
            llm=OpenAI(
                temperature=temperature,
                model=model)
            self._service_context = ServiceContext.from_defaults(
                llm=llm,
                chunk_size=chunk_size,
            )
            self._logger.info(f"-- Using OpenAI model {model}")
        elif model.startswith("HuggingFace"):
            llm = HuggingFaceLLM(
                model_name=model,
                tokenizer_name=model,
                context_window=3900,
                max_new_tokens=256,
                #generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                device_map="auto",
            )
            self._service_context = ServiceContext.from_defaults(
                llm=llm,
                chunk_size=chunk_size,
                embed_model="local"
            )
            self._logger.info(f"-- Using HuggingFace model {model}")
            
        # Set up service context
        self._service_context = ServiceContext.from_defaults(
            llm=llm,
            chunk_size=chunk_size,
            embed_model="local"
        )

    def _get_llama_docs(
        self,
        pdf_file: pathlib.Path
    ) -> list[Document]:
        """Get Llama documents from a PDF file.

        Parameters
        ----------
        pdf_file : pathlib.Path
            Path to the PDF file.

        Returns
        -------
        list[Document]
            List of Llama Documents.
        """

        loader = PyMuPDFLoader(pdf_file.as_posix())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )

        langchain_docs = loader.load_and_split(text_splitter)

        docs = [Document.from_langchain_format(doc) for doc in langchain_docs]

        return docs

    def _save_results(
        self,
        index: DocumentSummaryIndex,
        summary: str,
        path_save: pathlib.Path
    ) -> None:
        """Save the summary to a txt file and the index to a directory.

        Parameters
        ----------
        index : DocumentSummaryIndex
            Llama index.
        summary : str
            Summary of the PDF file.
        path_save : pathlib.Path
            Path to save the summary and the index.
        """

        # Save summary to txt
        txt_path = path_save / "summary.txt"
        with open(txt_path, 'w') as file:
            file.write(summary)

        # Save index
        index_path = path_save / 'index'
        index_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(
            persist_dir=index_path.as_posix())

        return

    def summarize(
        self,
        pdf_file: pathlib.Path,
        path_save: pathlib.Path
    ) -> None:
        """Summarize a PDF file using Llama Index.

        Parameters
        ----------
        pdf_file : pathlib.Path
            Path to the PDF file.
        path_save : pathlib.Path
            Path to save the summary and the index.
        """

        # Get Llama docs
        docs = self._get_llama_docs(pdf_file)

        # Build Llama index
        index = VectorStoreIndex.from_documents(docs, service_context=self._service_context)

        query_engine = index.as_query_engine()

        # Make query to obtain summary
        results = query_engine.query(self._instructions)
        self._logger.info(f"-- -- Summary: {results.response}")

        # Save results
        self._save_results(index, results.response, path_save)

        return
