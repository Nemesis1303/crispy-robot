"""
This module contains some auxiliary function for the management of NP backend's entities.

Author: Lorena Calvo-Bartolomé
Date: 12/04/2023
Modifed: 15/08/2024 (Updated for Into-Solr-Service (IntoKnown Proyect))
"""


import math
import random
from datetime import datetime

import numpy as np
import pytz


def is_valid_xml_char_ordinal(i):
    """
    Defines whether char is valid to use in xml document
    XML standard defines a valid char as::
    Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    """
    # conditions ordered by presumed frequency
    return (
        0x20 <= i <= 0xD7FF
        or i in (0x9, 0xA, 0xD)
        or 0xE000 <= i <= 0xFFFD
        or 0x10000 <= i <= 0x10FFFF
    )


def clean_xml_string(s):
    """
    Cleans string from invalid xml chars
    Solution was found there::
    http://stackoverflow.com/questions/8733233/filtering-out-certain-bytes-in-python
    """
    return "".join(c for c in s if is_valid_xml_char_ordinal(ord(c)))


def convert_datetime_to_strftime(df):
    """
    Converts all columns of type datetime64[ns] in a dataframe to strftime format.
    """
    columns = []
    for column in df.columns:
        if df[column].dtype == "datetime64[ns]":
            columns.append(column)
            df[column] = df[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df, columns


def parseTimeINSTANT(time):
    """
    Parses a string representing an instant in time and returns it as an Instant object.
    """
    format_string = '%Y-%m-%d %H:%M:%S'
    if isinstance(time, str) and time != "foo":
        dt = datetime.strptime(time, format_string)
        dt_utc = dt.astimezone(pytz.UTC)
        return clean_xml_string(dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
    elif time == "foo":
        return clean_xml_string("")
    else:
        if math.isnan(time):
            return clean_xml_string("")


def sum_up_to(
    vector: np.ndarray,
    max_sum: int
) -> np.ndarray:
    """It takes in a vector and a max_sum value and returns a NumPy array with the same shape as vector but with the values adjusted such that their sum is equal to max_sum using integer values.

    Parameters
    ----------
    vector: 
        The vector to be adjusted.
    max_sum: int
        Number representing the maximum sum of the vector elements.

    Returns:
    --------
    x: np.ndarray
        A NumPy array of the same shape as vector but with the values adjusted such that their sum is equal to max_sum.
    """
    x = np.array(list(map(np.int_, vector * max_sum))).ravel()
    pos_idx = list(np.where(x > 0)[0])
    current_sum = np.sum(x)
    difference = max_sum - current_sum
    
    while difference != 0:
        idx = random.choice(pos_idx)
        
        # Determine the adjustment direction based on the difference
        adjust = np.sign(difference)
        
        # Only adjust if it won't cause the value to go negative
        if x[idx] + adjust >= 0:
            x[idx] += adjust
            difference -= adjust
        
        # If no positive adjustments can be made, break out to prevent infinite loop
        if adjust > 0 and all(x[pos_idx] >= 0):
            break
        
    return x


def process_line(line):
    id_ = line.rsplit(' 0 ')[0].strip()
    id_ = int(id_.strip('"'))
    return id_


def calculate_beta_ds(betas):
    """Calculates beta with down-scoring
    Emphasizes words appearing less frequently in topics
    """

    ntopics = betas.shape[0]
    size_vocab = betas.shape[1]
    betas_ds = np.copy(betas)
    if np.min(betas_ds) < 1e-12:
        betas_ds += 1e-12
    deno = np.reshape((sum(np.log(betas_ds)) /
                       ntopics), (size_vocab, 1))
    deno = np.ones((ntopics, 1)).dot(deno.T)
    betas_ds = betas_ds * (np.log(betas_ds) - deno)
    return betas_ds
