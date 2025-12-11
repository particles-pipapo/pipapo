"""Vtk utils."""

import pathlib

import numpy as np
import pyvista as pv

from pipapo.utils.type_hinting import float_np_array, int_np_array


def dictionary_to_polydata(data_dict: dict) -> pv.PolyData:
    """Convert dictionary to polydata.

    Args:
        data_dict: Dictionary with the data

    Returns:
        pyvista object
    """
    points = pv.PolyData(data_dict["position"])

    for name, data in data_dict.items():
        points[name] = data

    return points


def data_to_dictionary(
    pyvista_data: pv.DataObject,
) -> dict[str, float_np_array | int_np_array]:
    """Convert polydata to dictionary.

    Args:
        pyvista_data: Pyvista object.

    Returns:
        Dictionary with data
    """
    dictionary = {}

    for key, value in pyvista_data.point_data.items():
        dictionary[key] = np.array(value)
    dictionary["position"] = np.array(pyvista_data.points)
    return dictionary


def import_vtk(file_path: pathlib.Path | str) -> dict:
    """Import vtk data to dictionary.

    Args:
        file_path: pathlib.Path to file

    Returns:
        Dictionary with the particle data
    """
    pyvista_data = pv.read(file_path)
    return data_to_dictionary(pyvista_data)


def export_vtk(dictionary: dict, file_path: pathlib.Path | str) -> None:
    """Export dictionary to vtk.

    Args:
        dictionary: Data to be exported
        file_path: pathlib.Path to store file
    """
    polydata = dictionary_to_polydata(dictionary)
    polydata.save(file_path)
