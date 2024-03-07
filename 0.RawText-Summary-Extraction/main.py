import pathlib
from src.pdf_parser import PDFParser
from src.summarizer import Summarizer

def main():
    
    # Define the path to the PDF file
    pdf_file = pathlib.Path(
        "../data/analitica_negocio.pdf")
    #pdf_file = pathlib.Path(
    #    "data/TDS-BBCC06105-2_Technical Description_GB-1.pdf"
    #)
    
    # Create a directory to save the extracted content (one directory per PDF file)
    path_save = pathlib.Path("../data") / pdf_file.stem
    path_save.mkdir(parents=True, exist_ok=True)
    path_save.joinpath("images").mkdir(parents=True, exist_ok=True)
    path_save.joinpath("tables").mkdir(parents=True, exist_ok=True)
    
    # TODO: This needs to be generic
    
    # instructions = \
    #     """ 
    #     You are a helpful AI assistant working with technical descriptions of air conditioner units. 
        
    #     Please summarize the technical description for the Roof-top air conditioner 680 by sections in such a way that the outputted text can be used as input for a topic modeling algorithm.
    # """
    
    instructions = \
    """You are a helpful AI assistant working with the generation of summaries of PDF documents. Please summarize the given document by sections in such a way that the outputted text can be used as input for a topic modeling algorithm. Dont start with 'The document can be summarized...' or 'The document is about...'. Just start with the first section of the document.
    """
    
    pdf_parser = PDFParser()
    summarizer = Summarizer()
    

    summarizer.summarize(
        pdf_file=pdf_file,
        instructions=instructions,
        path_save=path_save
    )
    
    content = pdf_parser.parse(
        pdf_path=pdf_file,
        path_save=path_save)
    
    return


if __name__ == "__main__":
    main()