import pathlib
from src.pdf_parser import PDFParser

def main():
    pdf_parser = PDFParser()
    content = pdf_parser.parser(pathlib.Path(
        "data/TDS-BBCC06105-2_Technical Description_GB-1.pdf"))
    
    import pdb; pdb.set_trace()
    return


if __name__ == "__main__":
    main()