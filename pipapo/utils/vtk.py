"""Vtk utils."""
from pathlib import Path

import numpy as np
import pyvista as pv

from pipapo.utils.io import check_if_file_exist

# pylint: disable=E1101


def dictionary_to_polydata(data_dict):
    """Convert dictionary to polydata/

    Args:
        data_dict (dict): Dictionary with the data

    Returns:
        pyvista.PolyData: pyvista object
    """
    points = pv.PolyData(data_dict["position"])

    for name, data in data_dict.items():
        points[name] = data

    return points


def data_to_dictionary(pyvista_data):
    """Convert polydata to dictionary.

    Args:
        pyvista_data (obj): Pyvista object.

    Returns:
        dict: Dictionary with data
    """
    dictionary = {}
    for key, value in pyvista_data.point_data.items():
        dictionary[key] = value
    dictionary["id"] = np.arange(len(value)).reshape(-1, 1)
    dictionary["position"] = pyvista_data.points
    return dictionary


def import_vtk(file_path):
    """Import vtk data to dictionary.

    Args:
        file_path (str): Path to file

    Returns:
        dict: dictionary with the particle data
    """
    pyvista_data = pv.read(file_path)
    return data_to_dictionary(pyvista_data)


def export_vtk(dictionary, file_path):
    """Export dictionary to vtk.

    Args:
        dictionary (dict): Data to be exported
        file_path (str): Path to store file
    """
    polydata = dictionary_to_polydata(dictionary)
    polydata.save(file_path)
