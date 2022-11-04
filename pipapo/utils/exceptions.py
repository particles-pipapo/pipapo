"""Custom error classes."""


class ContainerError(Exception):
    """Container exception."""


class LenError(ContainerError):
    """Length exception."""
