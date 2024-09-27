import pathlib
from src.pdf_parser import PDFParser
from src.summarizer import Summarizer
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file',
                        default="/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/data/casos.pdf/S0004-06142005000200009-1.pdf")
    parser.add_argument('--path_save', type=str,
                        help='Path to save the extracted content',
                        default="/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/data/casos_processed")
    args = parser.parse_args()

    pdf_file = pathlib.Path(args.pdf_path)
    path_save = pathlib.Path(args.path_save) / pdf_file.stem

    # Create a directory for the extracts (one directory per PDF file)
    path_save.mkdir(parents=True, exist_ok=True)
    path_save.joinpath("images").mkdir(parents=True, exist_ok=True)
    path_save.joinpath("tables").mkdir(parents=True, exist_ok=True)

    # Create a PDFParser and parse the PDF file
    pdf_parser = PDFParser(
        extract_header_footer=False,
        generate_img_desc=False,
        generate_table_desc=False,
    )
    pdf_parser.parse(pdf_path=pdf_file, path_save=path_save)

    # Create a Summarizer with the default parameters and summarize the PDF file
    ## TO USE THE OPENAI MODEL, UNCOMMENT THE FOLLOWING LINE
    summarizer = Summarizer(
        model_type="openai",
        model_name="gpt-4",
        instructions="Proporcione un resumen conciso del texto proporcionado en el mismo idioma que el texto.",
    )
    ## TO USE THE HUGGINGFACE MODEL, UNCOMMENT THE FOLLOWING LINE
    #summarizer = Summarizer(
    #    model_type="hf",
    #    model_name="HuggingFaceH4/zephyr-7b-beta",
    #)
    summarizer.summarize(pdf_file=pdf_file, path_save=path_save)
    
    return


if __name__ == "__main__":
    main()
