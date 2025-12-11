"""Utility classes to handle different iterable types uniformly."""

from typing import Any, Iterable, Protocol

import numpy as np


class IterableHandler[T](Protocol):
    """Protocol for handling different iterable types uniformly."""

    def get_single_item(self, data: T, i: int) -> T:
        """Get a single item from the iterable at index i."""

    def get_iterable_items(self, data: T, indices: Iterable[int]) -> T:
        """Get multiple items from the iterable at specified indices."""

    def get_slice_items(self, data: T, slice_data: slice) -> T:
        """Get items from the iterable using a slice object."""

    def remove_single_item(self, data: T, i: int) -> T:
        """Remove a single item from the iterable at index i."""

    def remove_iterable_items(self, data: T, i: Iterable[int]) -> T:
        """Remove multiple items from the iterable at specified indices."""

    def extend(self, data: T, new_data: T) -> T:
        """Extend the iterable with new data."""


class ListHandler:
    """Handler for list type iterables."""

    def get_single_item(self, data: list, i: int) -> list:
        """Get a single item from the list at index i."""
        return [data[i]]

    def get_iterable_items(self, data: list, indices: Iterable[int]) -> list:
        """Get multiple items from the list at specified indices."""
        return [data[j] for j in indices]

    def get_slice_items(self, data: list, slice_data: slice) -> list:
        """Get items from the list using a slice object."""
        return data[slice_data]

    def remove_single_item(self, data: list, i: int) -> list:
        """Remove a single item from the list at index i."""
        data.pop(i)
        return data

    def remove_iterable_items(self, data: list, i: Iterable[int]) -> list:
        """Remove multiple items from the list at specified indices."""
        for j in sorted(i)[::-1]:
            data.pop(j)
        return data

    def extend(self, data: list, new_data: list) -> list:
        """Extend the list with new data."""
        return data + new_data


class NDArrayHandler:
    """Handler for numpy ndarray type iterables."""

    def get_single_item(self, data: np.ndarray, i: int) -> np.ndarray:
        """Get a single item from the ndarray at index i."""
        return data[i : i + 1]

    def get_iterable_items(
        self, data: np.ndarray, indices: Iterable[int]
    ) -> np.ndarray:
        """Get multiple items from the ndarray at specified indices."""
        return data[indices]  # type: ignore[call-overload]

    def get_slice_items(self, data: np.ndarray, slice_data: slice) -> np.ndarray:
        """Get items from the ndarray using a slice object."""
        return data[slice_data]  # type: ignore[call-overload]

    def remove_single_item(self, data: np.ndarray, i: int) -> np.ndarray:
        """Remove a single item from the ndarray at index i."""
        return np.delete(data, [i], axis=0)

    def remove_iterable_items(self, data: np.ndarray, i: Iterable[int]) -> np.ndarray:
        """Remove multiple items from the ndarray at specified indices."""
        return np.delete(data, i, axis=0)  # type: ignore[call-overload]

    def extend(self, data: np.ndarray, new_data: np.ndarray) -> np.ndarray:
        """Extend the ndarray with new data."""
        return np.concatenate((data, new_data), axis=0)


class Handlers:  # pylint: disable = too-few-public-methods
    """Class to manage different iterable handlers."""

    def __init__(self, handlers: dict[str, IterableHandler]):
        """Initialize with a dictionary of handlers."""
        self._handlers = handlers

    def __call__(self, iterable_variable: Any) -> IterableHandler:
        """Get the appropriate handler for the given iterable variable."""
        return self._handlers[type(iterable_variable).__name__]
