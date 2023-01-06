"""Io utils."""
from pathlib import Path


def check_if_file_exist(file_path):
    """Check if file exists.

    Args:
        file_path (str): _description_
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(
            f"File {str(Path(file_path).resolve())} does not exist."
        )
