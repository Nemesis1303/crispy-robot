import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union

def unpickler(file: str) -> object:
    """Unpickle file

    Parameters
    ----------
    file : str
        The path to the file to unpickle.

    Returns
    -------
    object
        The unpickled object.
    """
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob: object) -> int:
    """Pickle object to file

    Parameters
    ----------
    file : str
        The path to the file where the object will be pickled.
    ob : object
        The object to pickle.

    Returns
    -------
    int
        0 if the operation is successful.
    """
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return 0

def file_lines(fname: Path) -> int:
    """
    Count number of lines in file

    Parameters
    ----------
    fname: Path
        The file whose number of lines is calculated.

    Returns
    -------
    int
        Number of lines in the file.
    """
    with fname.open('r', encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_embeddings_from_str(df: pd.DataFrame, logger: Union[logging.Logger, None] = None) -> np.ndarray:
    """
    Get embeddings from a DataFrame, assuming there is a column named 'embeddings' with the embeddings as strings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the embeddings as strings in a column named 'embeddings'
    logger : Union[logging.Logger, None], optional
        Logger for logging errors, by default None

    Returns
    -------
    np.ndarray
        Array of embeddings
    """

    if "embeddings" not in df.columns:
        if logger:
            logger.error(
                f"-- -- DataFrame does not contain embeddings column"
            )
        else:
            print(
                f"-- -- DataFrame does not contain embeddings column"
            )
        
    embeddings = df.embeddings.values.tolist()
    if isinstance(embeddings[0], str):
        embeddings = np.array(
            [np.array(el.split(), dtype=np.float32) for el in embeddings])

    return np.array(embeddings)