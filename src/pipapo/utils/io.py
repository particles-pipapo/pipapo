"""Io utils."""
from pathlib import Path


def check_if_file_exist(file_path):
    """Check if file exists.

    Args:
        file_path (str): file path
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(
            f"File {str(Path(file_path).resolve())} does not exist."
        )


def pathify(file_path):
    """Simple wrapper to avoid importing pathlib.

    Args:
        file_path (str, pathlib.Path): Path to be wrapped

    Returns:
        pathlib.path: Path object
    """
    return Path(file_path)


def export(dictionary, file_path):
    """Export dictionary as csv, vtk or vtp.

    Args:
        dictionary (dict): data to be exported
        file_path (pathlib.Path): export file path
    """
    file_path = pathify(file_path)
    if file_path.suffix in [".vtk", ".vtp"]:
        from pipapo.utils.vtk import export_vtk

        export_vtk(dictionary, file_path)
    elif file_path.suffix == ".csv":
        from pipapo.utils.csv import export_csv

        export_csv(dictionary, file_path)
    else:
        raise IOError(
            f"Filetype {file_path.suffix} unknown. Supported export file types are vtk, vtp or "
            "csv."
        )
