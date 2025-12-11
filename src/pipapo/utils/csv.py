"""Csv utils."""

import pathlib
from typing import Any

import numpy as np
import pandas

from pipapo.utils.path_utils import check_if_file_exist


def export_csv(dictionary: dict, file_path: pathlib.Path | str) -> None:
    """Export dictionary to csv.

    Args:
        dictionary :Container dict
        file_path:File path to be stored at.
    """
    labels = []
    data_arrays = []
    for label, data in dictionary.items():
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data.shape) == 1:
            data_arrays.append(data.flatten())
            labels.append(label)
        elif data.shape[1] == 1:
            data_arrays.append(data.flatten())
            labels.append(label)
        else:
            for i, column in enumerate(data.T):
                data_arrays.append(column.flatten())
                labels.append(label + f"_{i}")

    pd_dataframe = pandas.DataFrame.from_dict(dict(zip(labels, data_arrays)))
    pd_dataframe.to_csv(file_path, sep=",", index=False)


def import_csv(file_path: str, **kwargs: Any) -> dict:
    """Import dictionary from csv.

    Args:
        file_path: Path to the csv file.

    Returns:
        Container dict
    """
    check_if_file_exist(file_path)
    pandas_dataframe = pandas.read_csv(file_path, **kwargs)
    dictionary = {}
    for column in pandas_dataframe:
        dictionary[column] = pandas_dataframe[column].to_numpy()
    return dictionary
