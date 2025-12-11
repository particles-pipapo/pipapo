"""Base container."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from copy import deepcopy
from typing import Generic, Self, TypeVar

import numpy as np

from pipapo.utils.iterable_handler import Handlers, ListHandler, NDArrayHandler
from pipapo.utils.type_hinting import int_np_array

P = TypeVar("P", bound=Mapping)
D = TypeVar("D", bound=Mapping)


HANDLERS = Handlers(
    {list.__name__: ListHandler(), np.ndarray.__name__: NDArrayHandler()}
)


class Container(Generic[P, D]):
    """Generic container for properties and data fields."""

    _safe = True
    _handlers: Handlers = HANDLERS

    def __init__(self, properties: P, data: D | None):
        """Intiialize container.

        Args:
            properties: Container properties.
            data: Container data fields.
        """
        self._properties = properties
        if data is None:
            self._field_data: D = {}  # type: ignore[assignment]
        else:
            self._field_data = data

        self._check_lens()

    def _check_lens(self) -> None:
        """Check if lengths match."""
        if self._safe:
            len(self)

    @property
    def data(self) -> D:
        """Get data property."""
        self._check_lens()
        return self._field_data

    @data.setter
    def data(self, new_data: D) -> None:
        """Set data.

        Args:
            new_data: New data
        """
        self._field_data = new_data
        self._check_lens()

    def __str__(self) -> str:
        """Get string representation of the container."""
        string = type(self).__name__ + f" of size {len(self)}"
        string += "\n Properties:\n   " + "\n   ".join(self._properties.keys())
        if self._field_data:
            string += "\n Data:\n   " + "\n   " + "\n   ".join(self._field_data.keys())
        return string

    def __len__(self) -> int:
        """Length of the container."""
        lengths_property: set = set()
        lengths_data: set = set()

        for k, v in self._properties.items():
            try:
                lengths_property = lengths_property.union((len(v),))

            except TypeError as exc:
                raise TypeError(
                    f"Properties '{k}', does not have a `len` method."
                ) from exc

        for k, v in self._field_data.items():
            try:
                lengths_data = lengths_data.union((len(v),))

            except TypeError as exc:
                raise TypeError(f"Data '{k}', does not have a `len` method.") from exc

        if lengths_data and lengths_data != lengths_property:
            raise ValueError(
                f"Data and Properties have different lengths: \n"
                f" Properties: {lengths_property}\n Data: {lengths_data}"
            )

        if not lengths_property:
            return 0

        if len(set(lengths_property)) == 1:
            return list(lengths_property)[0]

        error_string = "\n  ".join(
            (f"{f}: {l}" for f, l in zip(self._properties, lengths_property))
        )
        if lengths_data:
            error_string += "\n  " + "\n  ".join(
                (f"{f}: {l}" for f, l in zip(self._field_data, lengths_data))
            )
        raise ValueError("Data are not equal sized: \n  " + error_string)

    def __getitem__(self, i: int | np.integer | Iterable[int] | int_np_array) -> Self:
        """Get item(s) from the container.

        Args:
            i: Index or slice to get item(s).

        Returns:
            A new container with the selected item(s).
        """
        data = {}
        properties = {}

        if isinstance(i, (int)):
            l = len(self)
            if i >= l or i < -l:
                raise IndexError(f"Index {i} out of range for size {l}")
            data = {
                key: self._handlers(value).get_single_item(value, i)
                for key, value in self._field_data.items()
            }
            properties = {
                key: self._handlers(value).get_single_item(value, i)
                for key, value in self._properties.items()
            }

        elif isinstance(i, Iterable):
            data = {
                key: self._handlers(value).get_iterable_items(value, i)
                for key, value in self._field_data.items()
            }
            properties = {
                key: self._handlers(value).get_iterable_items(value, i)
                for key, value in self._properties.items()
            }

        elif isinstance(i, slice):
            data = {
                key: self._handlers(value).get_slice_items(value, i)
                for key, value in self._field_data.items()
            }
            properties = {
                key: self._handlers(value).get_slice_items(value, i)
                for key, value in self._properties.items()
            }
        else:
            raise IndexError("error")

        new_container = self.create_new_instance(
            properties,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
            self._additional_properties(),
        )
        return new_container

    @classmethod
    def create_new_instance(
        cls, properties: P, data: D, additional_properties: dict
    ) -> Self:
        """Create a new instance of the same type as self.

        Args:
            properties: The properties for the new instance.
            data: The data for the new instance.
            additional_properties: Additional properties that might be constant.

        Returns:
            A new instance of the same type as self.
        """
        return cls(properties, data, **additional_properties)

    def remove(self, i: int | Iterable[int]) -> None:
        """Remove item(s) from the container.

        Args:
            i: Index or indices of the item(s) to remove.
        """
        if isinstance(i, int):
            for key in self._field_data:
                self._field_data[key] = self._handlers(  # type:ignore[index]
                    self._field_data[key]
                ).remove_single_item(self._field_data[key], i)
            for key in self._properties:
                self._properties[key] = self._handlers(  # type:ignore[index]
                    self._properties[key]
                ).remove_single_item(self._properties[key], i)

        elif isinstance(i, Iterable):
            for key in self._field_data:
                self._field_data[key] = self._handlers(  # type:ignore[index]
                    self._field_data[key]
                ).remove_iterable_items(self._field_data[key], i)
            for key in self._properties:
                self._properties[key] = self._handlers(  # type:ignore[index]
                    self._properties[key]
                ).remove_iterable_items(self._properties[key], i)
        else:
            raise TypeError("Can only remove entries by int or iteratables of ints")

    def append(self, other: Container) -> None:
        """Append another container to this one.

        Args:
            other: The container to append.
        """
        for key, value in self._field_data.items():
            self._field_data[key] = self._handlers(value).extend(  # type:ignore[index]
                value, other._field_data[key]  # pylint: disable=protected-access
            )
        for key, value in self._properties.items():
            self._properties[key] = self._handlers(value).extend(  # type:ignore[index]
                value, other._properties[key]  # pylint: disable=protected-access
            )

    def deepcopy(self) -> Self:
        """Create a deep copy of the container."""
        return self.create_new_instance(
            deepcopy(self._properties),
            deepcopy(self._field_data),
            additional_properties=self._additional_properties(),
        )

    def to_dict(self) -> dict[str, Collection]:
        """Return dictionary.

        Returns:
            Properties and data of the dictionnary
        """
        return self._properties | self._field_data  # type: ignore[operator]

    def _additional_properties(self) -> dict:
        """Return additional properties.

        Returns:
            Additional data
        """
        return {}
