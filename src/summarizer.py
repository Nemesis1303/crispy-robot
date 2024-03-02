import logging
import os
import pathlib
from llama_index.llms.openai import OpenAI
from  llama_index.core.schema import Document
from llama_index.core import ServiceContext, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.query_engine import RetrieverQueryEngine

class Summarizer(object):
    def __init__(
        self, 
        model="gpt-4"
    ) -> None:
        
        self._api_key = os.environ['OPENAI_API_KEY']

        logging.basicConfig(level='INFO')
        self._logger = logging.getLogger('PDFParser')
                
        self._service_context = ServiceContext.from_defaults(
        llm = OpenAI(
            temperature=0,
            model=model),
            chunk_size=1024
        ) # Parametrize this; maybe read from config
        
        # Default mode of building the index
        self._response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", use_async=True
        ) # Parametrizee this
    
    def _get_llama_docs(
        self,
        pdf_file: pathlib.Path):
        
        loader = PyMuPDFLoader(pdf_file.as_posix())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )
        
        langchain_docs = loader.load_and_split(text_splitter)
        
        docs = [Document.from_langchain_format(doc) for doc in langchain_docs] 
         
        return docs
    
    def _build_llama_index(
        self,
        docs: list) -> None:
        
        # Create the index
        doc_summary_index = DocumentSummaryIndex.from_documents(
            docs,
            service_context=self._service_context,
            response_synthesizer=self._response_synthesizer,
            show_progress=True,
        )
        
        return doc_summary_index
    
    def _save_results(
        self,
        index: DocumentSummaryIndex,
        summary: str,
        path_save: pathlib.Path):
        
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
        instructions: str,
        path_save: pathlib.Path):
        
        # Get Llama docs
        docs = self._get_llama_docs(pdf_file)
        
        # Build Llama index
        index = self._build_llama_index(docs)
        
        # Use LLM-powered retrieval vs Embedding-based retrieval (higher latency and cost but returns more relevant docs)
        retriever = DocumentSummaryIndexLLMRetriever(
            index,
            # choice_select_prompt=None,
            # choice_batch_size=10,
            # choice_top_k=1,
            # format_node_batch_fn=None,
            # parse_choice_select_answer_fn=None,
            # service_context=None
        )
        
        # Assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=self._response_synthesizer,
        )

        # Make query to obtain summary
        results = query_engine.query(instructions)
        self._logger.info(f"Summary: {results.response}")
        
        # Save results
        self._save_results(index, results.response, path_save)
        
        return 