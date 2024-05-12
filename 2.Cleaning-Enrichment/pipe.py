"""
Main script to apply the NLP pipeline to clean and enrich text data.

Author: Lorena Calvo-Bartolomé
Date: 12/05/2024
"""

import argparse
import logging
import yaml
import pandas as pd
from src.nlp_pipeline import NLPpipeline

def cal_element_pipe(
    pipe,
    element,
    df,
    col_calculate_on,
    replace_acronyms=False
) -> pd.DataFrame:

    if element == "lang_id":
        df = pipe.get_lang(df, col_calculate_on)
    elif element == "acronyms":
        df = pipe.get_acronyms(df, col_calculate_on)
    elif element == "lemmas":
        df = pipe.get_lemmas(df, col_calculate_on, replace_acronyms)
    elif element == "ngrams":
        df = pipe.get_ngrams(df, col_calculate_on)
    elif element == "embeddings":
        df = pipe.get_context_embeddings(df, col_calculate_on)
    elif element == "ner_generic":
        df = pipe.get_ner_generic(df, col_calculate_on)
    elif element == "ner_specific":
        df = pipe.get_ner_specific(df, col_calculate_on)
    else:
        raise ValueError(f"-- -- Element '{element}' not recognized.")
    
    return df

if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser(
        description="Options for clean and enrich text data.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--source",
        default="../data/output4/data.parquet",
        help="Path to source data file."
    )
    
    args = parser.parse_args()
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Load config file
    with open(args.config, "r") as f:
        config = dict(yaml.safe_load(f))
        
    # Load pipeline
    pipe = config.get("pipe", [])
    
    if not pipe:
        raise ValueError("-- -- No pipeline defined in config file.")
    
    # Create pipeline with options from config file
    options_pipe = config.get("options_pipe", {})
    nlp_pipe = NLPpipeline(**options_pipe)
    
    # Load data
    df = pd.read_parquet(args.source)
    df_aux = df.copy().iloc[0:5]
    col_calculate_on = "raw_text"
    print(df_aux)
    
    # Apply pipeline
    replace_acronyms = True if "acronyms" in pipe else False
    for element in pipe:
        logger.info(f"-- -- Calculating element '{element}'")
        df_aux = cal_element_pipe(
            nlp_pipe,
            element,
            df_aux,
            col_calculate_on=col_calculate_on,
            replace_acronyms=replace_acronyms
        )
    import pdb; pdb.set_trace()
    
