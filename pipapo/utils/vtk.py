import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pathlib import Path
import numpy as np
from .io import check_if_file_exist


def export_polydata(polydata, file_path):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()


def add_numpy_array_to_polydata(polydata, name, data):
    data_vtk = numpy_to_vtk(data)
    data_vtk.SetName(name)
    polydata.GetPointData().AddArray(data_vtk)


def dictionary_to_polydata(data_dict):
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
    file_path = Path(file_path)
    check_if_file_exist(file_path)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(file_path))
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def polydata_to_dictionary(polydata):
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
    polydata = import_polydata(file_path)
    return polydata_to_dictionary(polydata)


def export_vtk(dictionary, file_path):
    polydata = dictionary_to_polydata(dictionary)
    export_polydata(polydata, file_path)
