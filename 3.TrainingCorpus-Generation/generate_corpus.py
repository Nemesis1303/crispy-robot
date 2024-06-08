"""
Generate a corpus from a set of documents.

This script generates a corpus from a set of documents by filtering the documents based on the presence of Named Entities (NERs) and preprocessing the text data, if required.
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Any

import pandas as pd
import yaml
from src.adhoc_preproc.preprocessor import Preprocessor
from src.docs_selection.doc_selector import DocSelector
from src.utils.utils import select_file_from_directory

def main(config: Dict[str, Any], logger: logging.Logger, df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to generate a corpus from a set of documents based on the provided configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    logger : logging.Logger
        Logger for logging information and errors.
    df : pd.DataFrame
        DataFrame containing the documents.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    mode = config.get('mode', 0)
    preproc = config.get('preproc', True)
    dc = DocSelector(logger=logger)

    try:
        if mode == 0:  # filter docs by ner with frequency
            df = dc.filter_docs_by_ners(df, with_freq=True, **config['doc_selector'])
            preproc = False
        elif mode == 1:  # filter docs by ner without frequency
            df = dc.filter_docs_by_ners(df, with_freq=False, **config['doc_selector'])
            col_preproc = config['doc_selector']['target_label'] + "_NER_IN_ROW"
        elif mode == 2:  # filter docs if ner
            df = dc.filter_docs_if_ner(
                df, **{k: v for k, v in config['doc_selector'].items() if k != 'remove_empty'}
            )
            col_preproc = config['doc_selector']['lemmas_col']
        elif mode == 3:  # just use lemmas
            col_preproc = config['doc_selector']['lemmas_col']
        else:
            raise ValueError(f"Mode {mode} not recognized")

        if preproc:
            if config["preprocessor"]["exec"]["mode"] == "manual":
                # Ask for user to select the stopwords and equivalent terms files
                stops = select_file_from_directory(config["preprocessor"]["exec"]["path_stw"], "stopwords")
                eqs = select_file_from_directory(config["preprocessor"]["exec"]["path_eq"], "equivalences")
            elif config["preprocessor"]["exec"]["mode"] == "auto":
                # use all stopwords files in the directory
                stops = config["preprocessor"]["exec"]["path_stw"]
                # use all equivalent terms files in the directory
                eqs = config["preprocessor"]["exec"]["path_eq"]

            pc = Preprocessor(
                stw_files=stops,
                eq_files=eqs,
                **config["preprocessor"]["object_creation"]
            )

            df = pc.preproc(df, col_preproc)

        return df

    except KeyError as e:
        logger.error(f"Missing required configuration key: {e}. Exiting...")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"{e}. Exiting...")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a corpus from a set of documents.")
    parser.add_argument('--config', type=str, help='Path to the configuration file.', default="config/config.yaml")
    parser.add_argument('--mode', type=int, help='Mode of operation.')
    parser.add_argument('--preproc', type=bool, help='Enable preprocessing.')
    parser.add_argument('--lemmas_col', type=str, help='Column name with the lemmas of the documents.')
    parser.add_argument('--ner_col', type=str, help='Column name with the NERs of the documents.')
    parser.add_argument('--target_label', type=str, help='Target NER label to filter the documents.')
    parser.add_argument('--remove_empty', type=bool, help='Remove empty documents.')
    parser.add_argument('-s', '--source', help='Input parquet file', required=True)
    parser.add_argument('-o', '--output', help='Output parquet file', required=True)
    args = parser.parse_args()

    logger = logging.getLogger("TrainingCorpus-Generation")
    logging.basicConfig(level=logging.INFO)

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error opening config file: {e}. Exiting...")
        sys.exit(1)

    # Override config values with command-line arguments if provided
    if args.mode is not None:
        config['mode'] = args.mode
    if args.preproc is not None:
        config['preproc'] = args.preproc
    if 'doc_selector' not in config:
        config['doc_selector'] = {}
    if args.lemmas_col:
        config['doc_selector']['lemmas_col'] = args.lemmas_col
    if args.ner_col:
        config['doc_selector']['ner_col'] = args.ner_col
    if args.target_label:
        config['doc_selector']['target_label'] = args.target_label
    if args.remove_empty:
        config['doc_selector']['remove_empty'] = args.remove_empty

    try:
        df = pd.read_parquet(args.source)
    except Exception as e:
        logger.error(f"Error loading data: {e}. Exiting...")
        sys.exit(1)
    logger.info(f"Data loaded from {args.source}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns}")

    df = main(config, logger, df)

    try:
        df.to_parquet(args.output)
    except Exception as e:
        logger.error(f"Error saving data: {e}. Exiting...")
        sys.exit(1)