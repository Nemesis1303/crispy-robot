import pathlib

import nest_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from llama_index import Document, ServiceContext, get_response_synthesizer
from llama_index.indices.document_summary import (
    DocumentSummaryIndex, DocumentSummaryIndexLLMRetriever)
from llama_index.llms import OpenAI
from llama_index.query_engine import RetrieverQueryEngine

nest_asyncio.apply()

"""
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
"""

pdf_file = pathlib.Path("data/TDS-BBCC06105-2_Technical Description_GB-1.pdf")


def main():
    loader = UnstructuredPDFLoader(pdf_file.as_posix())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    langchain_docs = loader.load_and_split(text_splitter)
    docs = [Document.from_langchain_format(doc) for doc in langchain_docs]

    # LLM (gpt-3.5-turbo)
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        chunk_size=1024
    )
    # default mode of building the index
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", use_async=True
    )

    # Create the index
    doc_summary_index = DocumentSummaryIndex.from_documents(
        docs,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )

    # Use LLM-powered retrieval vs Embedding-based retrieval (higher latency and cost but returns more relevant docs)
    retriever = DocumentSummaryIndexLLMRetriever(
        doc_summary_index,
        # choice_select_prompt=None,
        # choice_batch_size=10,
        # choice_top_k=1,
        # format_node_batch_fn=None,
        # parse_choice_select_answer_fn=None,
        # service_context=None
    )

    instructions = \
        """ 
        You are a helpful AI assistant working with technical descriptions of air conditioner units. 
        
        Please summarize the technical description for the Roof-top air conditioner 680 by sections in such a way that the outputted text can be used as input for a topic modeling algorithm.
    """

    # retrieved_nodes = retriever.retrieve(instructions)
    # print(retrieved_nodes[0].score)
    # print(retrieved_nodes[0].node.get_text())

    # use retriever as part of a query engine
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize"
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # make query to obtain summary
    results = query_engine.query(instructions)

    # Save summary to txt
    pathlib.Path(
        f"data/txts/{pdf_file.stem}").mkdir(parents=True, exist_ok=True)
    with open(f"data/txts/{pdf_file.stem}/summary.txt", 'w') as file:
        file.write(results.response)

    import pdb
    pdb.set_trace()
    # Save index for later use
    pathlib.Path(
        f"data/extracts/{pdf_file.stem}").mkdir(parents=True, exist_ok=True)
    doc_summary_index.storage_context.persist(
        persist_dir=f"data/extracts/{pdf_file.stem}")

    return


if __name__ == "__main__":
    main()
