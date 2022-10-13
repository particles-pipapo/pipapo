from pathlib import Path


def check_if_file_exist(file_path):
    if not Path(file_path).is_file():
        raise FileNotFoundError(
            f"File {str(Path(file_path).resolve())} does not exist."
        )
