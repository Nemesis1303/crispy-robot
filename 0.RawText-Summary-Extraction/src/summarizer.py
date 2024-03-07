import logging
import os
import pathlib
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from  llama_index.core.schema import Document
from llama_index.core import ServiceContext, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex, VectorStoreIndex
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Summarizer(object):
    def __init__(
        self, 
        model="gpt-4"
    ) -> None:
        
        path_env = pathlib.Path(os.getcwd()).parent / '.env'
        load_dotenv(path_env)
        self._api_key = os.getenv("OPENAI_API_KEY")

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
        index = VectorStoreIndex.from_documents(docs)
        
        query_engine = index.as_query_engine()

        # Make query to obtain summary
        results = query_engine.query(instructions)
        self._logger.info(f"Summary: {results.response}")
        
        # Save results
        self._save_results(index, results.response, path_save)
        
        return 