import pathlib

import nest_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from llama_index import Document, ServiceContext, get_response_synthesizer
from llama_index.indices.document_summary import (
    DocumentSummaryIndex, DocumentSummaryIndexLLMRetriever)
from llama_index.query_engine import RetrieverQueryEngine
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from tqdm import tqdm
import json
import fitz
import os
from PIL import Image
import pytesseract


nest_asyncio.apply()

def get_openaikey(file):
  with open(file, 'r') as file:
    openaikey = file.read()
  return openaikey

# Update to path to .env file with key
openaikey = get_openaikey(file=pathlib.Path(".env").as_posix())

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


def generate_json(metadata, pages, header, footer):
    document_json = {
        "metadata": metadata,
        "header": header,
        "footer": footer,
        "pages": [
            {"page_number": i + 1, "text": page} for i, page in enumerate(pages)
        ]
    }
    return document_json

pdf_file = pathlib.Path("data/TDS-BBCC06105-2_Technical Description_GB-1.pdf")


def extract_images(pdf_path):
    #Open PDF file
    pdf_file = fitz.open(pdf_path)

    # Calculate number of pages in PDF file
    page_nums = len(pdf_file)

    # Create empty list to store images information
    images_list = []

    # Extract all images information from each page
    for page_num in range(page_nums):
        page_content = pdf_file[page_num]
        images_list.extend(page_content.get_images())
    
    #print(images_list)    
    
    #Raise error if PDF has no images
    if len(images_list)==0:
        raise ValueError(f'No images found in {pdf_path}')
    
    #Save all the extracted images
    for i, image in enumerate(images_list, start=1):
        #Extract the image object number
        xref = image[0]
        #Extract image
        base_image = pdf_file.extract_image(xref)
        #Store image bytes
        image_bytes = base_image['image']
        #Store image extension
        image_ext = base_image['ext']
        #Generate image file name
        image_name = str(i) + '.' + image_ext
        #Save image
        with open(os.path.join("/Users/lbartolome/Documents/GitHub/crispy-robot/data/images", image_name) , 'wb') as image_file:
            image_file.write(image_bytes)
            image_file.close()
            
            
def extract_text():
    
    import easyocr
    
    reader = easyocr.Reader(['en'])
    text = reader.readtext('/Users/lbartolome/Documents/GitHub/crispy-robot/data/images/1.jpeg')
    print( text )

def main():
    
    extract_images(pdf_file.as_posix())
    extract_text()
    
    
    loader2 = PyMuPDFLoader(pdf_file.as_posix())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    langchain_docs2 = loader2.load_and_split(text_splitter)
    
    docs = [Document.from_langchain_format(doc) for doc in langchain_docs2] 
    
    text_ = [doc.page_content for doc in langchain_docs2]
    
    parameters = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant trained on the task of formatting the text extracted from a PDF document. Your are going to have given a string representing the content of a page of the document. Your task is to format the text in such a way that it is human readable. You must keep all the original content, just remove weird characters. If you see that the text is unnecessarily split, put it together. Remove extra spaces."
                 },
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "frequency_penalty": 0.0
        }

    from openai import OpenAI
    client = OpenAI(
        api_key=openaikey
    )
    
    text_formatted = []
    for el in tqdm(text_):
        gpt_prompt = f"Format the following page: {el}"
        
        message = [{"role": "user", "content": gpt_prompt}]
        parameters["messages"] = [parameters["messages"][0], *message]
        
        response = client.chat.completions.create(
            **parameters
        )
        text_formatted.append(response.choices[0].message.content)
    
    text_formatted = [text_formatted[i].replace(f"Sheet {i+1} of 25", "") for i in range(len(text_formatted))]
    header_footer = text_formatted[0].split("invalid")[0] + "invalid\n"
    header = header_footer.split("\n\nChecked")[0]
    footer = "Checked" + header_footer.split("\n\nChecked")[1]
    text_formatted = [text_formatted[i].replace(header_footer, "") for i in range(len(text_formatted))]
    
    metadata_doc = langchain_docs2[0].metadata
    metadata_doc.pop("page")
    result_json = generate_json(langchain_docs2[0].metadata, text_formatted, header, footer)
    with open(f"data/txts/{pdf_file.stem}/extracted_text.json", "w", encoding="utf-8") as json_file:
        json.dump(result_json, json_file, indent=2)

    #import pdb; pdb.set_trace()
    ###############    
    # LLM (gpt-3.5-turbo)
    from llama_index.llms import OpenAI
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            temperature=0,
            model="gpt-4"),
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
        #You are a helpful AI assistant working with technical descriptions of air conditioner units. 
        
        #Please summarize the technical description for the Roof-top air conditioner 680 by sections in such a way that the outputted text can be used as input for a topic modeling algorithm.
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
