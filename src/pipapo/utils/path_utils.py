"""Path utils."""

import pathlib


def check_if_file_exist(file_path: pathlib.Path | str) -> None:
    """Check if file exists.

    Args:
        file_path:file path
    """
    if not pathlib.Path(file_path).is_file():
        raise FileNotFoundError(
            f"File {str(pathlib.Path(file_path).resolve())} does not exist."
        )
