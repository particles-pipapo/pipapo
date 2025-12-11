"""Io utils."""

import pathlib

from pipapo.utils.csv import export_csv
from pipapo.utils.vtk import export_vtk


def export(dictionary: dict, file_path: pathlib.Path | str) -> None:
    """Export dictionary as csv, vtk or vtp.

    Args:
        dictionary :data to be exported
        file_path:export file path
    """
    file_path = pathlib.Path(file_path)
    if file_path.suffix in [".vtk", ".vtp"]:

        export_vtk(dictionary, file_path)
    elif file_path.suffix == ".csv":

        export_csv(dictionary, file_path)
    else:
        raise IOError(
            f"Filetype {file_path.suffix} unknown. Supported export file types are vtk, vtp or "
            "csv."
        )
