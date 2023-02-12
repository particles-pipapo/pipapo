"""Vtk utils."""
from pathlib import Path

import numpy as np
from pipapo.utils.io import check_if_file_exist

import vtk  # pylint: disable=W0406
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # pylint: disable=E

# pylint: disable=E1101


def export_polydata(polydata, file_path):
    """Export polydata.

    Args:
        polydata (vtk.PolyData): Polydata object ot be exported
        file_path (pathlib.Path): Path to be exported.
    """
    if file_path.suffix == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    elif file_path.suffix == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        raise IOError(f"Can not export {file_path.suffix} files.")
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()


def add_numpy_array_to_polydata(polydata, name, data):
    """Add numpy array to polydata object.

    Args:
        polydata (vtk.PolyData): Polydata object
        name (str): Name of the field
        data (np.array): Field data
    """
    data_vtk = numpy_to_vtk(data)
    data_vtk.SetName(name)
    polydata.GetPointData().AddArray(data_vtk)


def dictionary_to_polydata(data_dict):
    """Convert dictionary to polydata/

    Args:
        data_dict (dict): Dictionary with the data

    Returns:
        vtk.PolyData: Polydata object
    """
    polydata = vtk.vtkPolyData()

    position = numpy_to_vtk(data_dict["position"])
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(position)
    polydata.SetPoints(points_vtk)
    data_dict.pop("position")

    for name, data in data_dict.items():
        add_numpy_array_to_polydata(polydata, name, data)

    return polydata


def import_polydata(file_path):
    """Import polydata from path.

    Args:
        file_path (pathlib.Path): Path to the vtk-file

    Returns:
        vtk.PolyData: Polydata object
    """
    file_path = Path(file_path)
    check_if_file_exist(file_path)
    if file_path.suffix == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif file_path.suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise IOError(f"Can not import {file_path.suffix} files.")
    reader.SetFileName(str(file_path))
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def polydata_to_dictionary(polydata):
    """Convert polydata to dictionary.

    Args:
        polydata (vtk.Polydata): Polydata object.

    Returns:
        dict: Dictionary with data
    """
    dictionary = {}
    for i in range(polydata.GetPointData().GetNumberOfArrays()):
        data = polydata.GetPointData().GetArray(i)
        name = data.GetName()
        data = vtk_to_numpy(data)
        dictionary[name] = data.reshape(-1, 1)
    dictionary["id"] = np.arange(len(data)).reshape(-1, 1)
    points = polydata.GetPoints().GetData()
    dictionary["position"] = vtk_to_numpy(points)
    return dictionary


def import_vtk(file_path):
    """Import vtk data to dictionary.

    Args:
        file_path (str): Path to file

    Returns:
        dict: dictionary with the particle data
    """
    polydata = import_polydata(file_path)
    return polydata_to_dictionary(polydata)


def export_vtk(dictionary, file_path):
    """Export dictionary to vtk.

    Args:
        dictionary (dict): Data to be exported
        file_path (str): Path to store file
    """
    polydata = dictionary_to_polydata(dictionary)
    export_polydata(polydata, file_path)
