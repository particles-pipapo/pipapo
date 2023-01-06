from setuptools import find_packages, setup
import os
import sys


def read(fname):
    """Function to read the README file.

    Args:
        fname (str): File name to be read

    Returns:
        The content of the file fname
    """
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()


def read_requirements(fname):
    """Load the requirement file `fname` and remove comments denoted by '#'.

    Args:
        fname (str): File name

    Returns:
        packages (list): List of the required packages
    """
    packages = []
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        for line in f:
            line = line.partition("#")[0].rstrip()
            if line:
                packages.append(line)
    return packages


# Exit the installation process in case of incompatibility of the python version
REQUIRED_PYTHON_VERSION = "3.10"
system_current_version = f"{sys.version_info[0]}.{sys.version_info[1]}"

if system_current_version != REQUIRED_PYTHON_VERSION:
    message = (
        f"\n\nYour python version is {system_current_version}, however pipapo requires "
        f"{REQUIRED_PYTHON_VERSION}\n"
    )
    raise ImportError(message)

# Packages useful for developing
DEVELOPER_EXTRAS = [
    "pylint>=2.12",
    "isort>=5.0",
    "black==22.3.0",
    "pre-commit",
    "pip-tools",
    "vulture>=2.3",
]

setup(
    name="pipapo",
    packages=find_packages(include=["pipapo"]),
    version="0.0.1",
    description="pipapo - python implemented particles postprocessor",
    author="the patricles",
    license="gpl v3",
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": DEVELOPER_EXTRAS},
    long_description=read("README.md"),
)
