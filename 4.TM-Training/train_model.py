"""
Train a topic model on the output of the previous step.
"""

import argparse
import logging
import sys
from typing import Any, Dict, Type

import pandas as pd
import yaml
from src.train.tm_trainer import MalletLDATrainer, BERTopicTrainer


def create_model(model_name: str, **kwargs: Any) -> Any:
    """
    Create an instance of a trainer for the specified model.

    Parameters
    ----------
    model_name : str
        The name of the model to create a trainer for.
    **kwargs : dict
        Additional keyword arguments to pass to the trainer class.

    Returns
    -------
    trainer_instance : Any
        An instance of the trainer class for the specified model.

    Raises
    ------
    ValueError
        If the model_name does not correspond to a valid trainer class.
    """

    trainer_mapping: Dict[str, Type[Any]] = {
        'MalletLda': MalletLDATrainer,
        'BERTopic': BERTopicTrainer,
    }

    trainer_class = trainer_mapping.get(model_name)
    if trainer_class is None:
        raise ValueError(f"-- -- Invalid trainer name: {model_name}")

    trainer_instance = trainer_class(**kwargs)

    return trainer_instance


def main(
    config: Dict[str, Any],
    logger: logging.Logger,
    source: str,
    model_path: str
) -> pd.DataFrame:
    """
    Main function to train a corpus from a set of documents based on the provided configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    logger : logging.Logger
        Logger for logging information and errors.
    source : str
        Path to the source file (parquet format).
    model_path : str
        Path to save the trained model.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """

    model_type = config['model_type']
    model_params = config['dft_config_models'].get(model_type, {})
    model_params['num_topics'] = config['ntopics']
    model_params['model_path'] = model_path
    train_args = {
        key: value for key, value in {
            "text_col": config["text_col"],
            "raw_text_col": config.get("raw_text_col") if model_type == "BERTopic" else None
        }.items() if value is not None
    }

    trainer = create_model(model_type, **model_params)
    training_time = trainer.train(source, **train_args)

    logger.info(f"-- -- Training completed in {training_time} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a corpus from a set of documents.")
    parser.add_argument(
        '--config', type=str, help='Path to the configuration file.', default="config/config.yaml")
    parser.add_argument('--ntopics', type=int,
                        help='Number of topics to train the model with.')
    parser.add_argument('--model_type', type=str,
                        help='Type of model to train.', default='MalletLda')
    parser.add_argument(
        '-s', '--source', help='Input parquet file', required=True)
    parser.add_argument(
        '-o', '--output', help='Folder where the topic modeling output will be saved', required=True)
    args = parser.parse_args()

    logger = logging.getLogger("TM-Training")
    logging.basicConfig(level=logging.INFO)

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error opening config file: {e}. Exiting...")
        sys.exit(1)

    # Override config values with command-line arguments if provided
    if args.ntopics is not None:
        config['ntopics'] = args.ntopics
    if args.model_type is not None:
        config['model_type'] = args.model_type

    df = main(config, logger, args.source, args.output)
