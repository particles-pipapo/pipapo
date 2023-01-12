"""Dataclasses."""
import dataclasses
import warnings
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
from pipapo.utils.exceptions import ContainerError, LenError


class Container:
    """Simple container."""

    def __init__(
        self, default_initialization=None, datatype=None, *field_names, **fields
    ):
        """Initialize container.

        field_names or fields can be provided, but not both!
        Args:
            default_initialization (obj, optional): Initialization for the iterables. Defaults to
            None.
            datatype (obj, optional): Type of object this container holds. Defaults to None.
        """
        if bool(field_names) and bool(fields):
            raise ContainerError(
                "You provided field_names and fields. You can only provide the one or the other!"
            )
        if default_initialization is None:
            default_initialization = []

        self.id = default_initialization  # pylint: disable=C0103
        self.field_names = ["id"]

        if field_names:
            self.field_names = list(set(field_names).union(self.field_names))
            for key in field_names:
                setattr(self, key, default_initialization.copy())

        if fields:
            self.field_names = list(set(fields.keys()).union(self.field_names))
            len_array = []
            for key, value in fields.items():
                setattr(self, key, value.copy())
                len_array.append(len(value))
            len_array = set(len_array)
            if len(len_array) != 1:
                array_lens = ", ".join(
                    [
                        f"{key}: {length}"
                        for key, length in zip(fields.keys(), len_array)
                    ]
                )
                raise ValueError(f"Dimension mismatch between fields! {array_lens}")
            if "id" not in fields:
                self.id = np.arange(list(len_array)[0]).tolist()
        self.field_names.sort()
        if datatype is None:
            self._update_data_type()
        else:
            self.datatype = datatype
        self._current_idx = 0

    def _add_field_name(self, name):
        """Add field names to field_names attribute.

        Args:
            name (str): new field name
        """
        if not name.isidentifier():
            raise NameError(
                f"The provided name {name} is not a valid identifier! Remove spaces, leading"
                " numbers and special characters!"
            )
        self.field_names.append(name)
        self.field_names.sort()
        self._update_data_type()

    def _values(self):
        """Values method. Similar to the values method of a dict.

        Returns:
            list: List of fields.
        """
        self.field_names.sort()
        return [getattr(self, f) for f in self.field_names]

    def _items(self):
        """Items method. Similar to the items method of a dict.

        Returns:
            tuple: Tuple consisting of names and field values
        """
        return tuple(zip(self.field_names, self._values()))

    def __len__(self):
        """Len method.

        Returns:
            int: number of elements in the container.
        """
        lens = {safe_len(v) for v in self._values()}
        if len(lens) == 1:
            return list(lens)[0]

        if len(lens) == 0:
            return 0

        field_lengths = ", ".join([f"{k}: {len(v)}" for k, v in self._items()])
        raise LenError(f"Different lengths between fields: {field_lengths}")

    def __bool__(self):
        """Boolean method.

        Returns:
            bool: true if the container contains a least one element
        """
        return bool(len(self))

    def __iter__(self):
        """Make the container an iterator.

        Returns:
            Container: a copy of the current container.
        """
        return self.copy()

    def copy(self):
        """Copy function.

        Returns:
            Container: Returns a copy of self.
        """
        return deepcopy(self)

    def to_dict(self):
        """Create dictionary from container.

        Returns:
            dict: dictionary with current description.
        """
        return dict(self._items())

    def __next__(self):
        """Next method.

        Makes the container class a generator.

        Returns:
            obj: object of type self.datatype
        """
        if self._current_idx >= len(self):
            self._current_idx = 0
            raise StopIteration()

        item = self.datatype(
            **{key: value[self._current_idx] for key, value in self._items()}
        )
        self._current_idx += 1
        return item

    def add_field(self, field_name, field):
        """Add field to container.

        Args:
            field_name (str): name of the new field
            field (list): field values
        """
        if not len(field) == len(self):
            raise Exception(
                f"Dimension mismatch while adding new field. Current length is {len(self)} but the"
                f" field you want to add has length {len(field)}"
            )

        self._add_field_name(field_name)
        setattr(self, field_name, field.copy())

    def remove_field(self, field_name):
        """Remove field by name.

        Args:
            field_name (str): Field to be deleted.
        """
        if not field_name in self.field_names:
            warnings.warn(f"Field {field_name} does not exist, nothing was deleted!")
        else:
            self.field_names.remove(field_name)
            delattr(self, field_name)
            self._update_data_type()

    def evaluate_function(self, fun, add_as_field=False, field_name=None):
        """Evaluate function on container.

        This method evaluates the function fun for every element. Define the argument of the
        function is of type self.datatype.

        Args:
            fun (fun): Function to be evaluated
            add_as_field (bool, optional): Should the result be added as field. Defaults to False.
            field_name (str optional): Name of the field to be added.Defaults to None.
        """
        result = []
        for element in self:
            result.append(fun(element))
        if add_as_field:
            if not field_name:
                field_name = fun.__name__
                if field_name == "<lambda>":
                    raise NameError(
                        "You are using a lambda function. Please add a name to the field you want"
                        " to add with the keyword 'field_name'"
                    )
            self.add_field(field_name, result)
        return result

    def __getitem__(self, i):
        """Getitem method.

        Args:
            i (int, slice, iterable): Indexes to get the items

        Returns:
            self.datatype: If index i is an int
            type(self): If index i is iterable or slice
        """
        if isinstance(i, int):
            if i >= len(self) or i < -len(self):
                raise IndexError(f"Index {i} out of range for size {self.__len__()}")
            item = self.datatype(**{key: value[i] for key, value in self._items()})
        elif isinstance(i, Iterable):
            item = type(self)(
                **{key: [value[j] for j in i] for key, value in self._items()}
            )
        else:
            item = type(self)(**{key: value[i] for key, value in self._items()})
        return item

    def _map_id_to_index(self, element_id):
        """Map id to index.

        As the ids do not need to be continuous this helper method maps an id to its index.

        Args:
            element_id (int): Id of desired element

        Returns:
            int: index of element.
        """
        idx = np.where(np.array(self.id) == element_id)[0]
        if len(idx) > 1:
            raise IndexError(f"More than one element found with id {int(element_id)}.")

        if len(idx) == 0:
            raise IndexError(f"No element with id {int(element_id)} found!")

        return int(idx[0])

    def reset_ids(self):
        """Reset ids to a continuous range."""
        self.id = np.arange(len(self)).tolist()

    def get_by_id(self, ids):
        """Get elements by ids.

        Basically a wrapper for __getitem__ based on ids instead of indexes.

        Args:
            ids (int, slice, iterable): Ids to get the items

        Returns:
            self.datatype: If ids is an int
            type(self): If ids is iterable or slice
        """
        if isinstance(ids, int):
            return self[self._map_id_to_index(ids)]

        if not isinstance(ids, (Iterable, slice)):
            raise TypeError(
                f"Ids need to be iterable (list, tuple, ...) or an int but which type {type(ids)}"
                " is not."
            )

        indexes = []
        for idx in ids:
            indexes.append(self._map_id_to_index(idx))

        return self[indexes]

    def update_by_id(self, element, element_id):
        """Update particle by id.

        Args:
            element (self.datatype): Element to be updated.
            element_id (int): Id of the element to be updated.
        """
        idx = self._map_id_to_index(element_id)

        for field, value in element.items():
            if field == "id":
                continue
            field_array = getattr(self, field)
            field_array[idx] = value

    def __list__(self):
        """Create list of self.datatype elements from container.

        Returns:
            list: List consisting of self.datatype elements
        """
        return [
            self.datatype(
                {key: value[i] for key, value in self._items()}
                for i in range(len(self))
            )
        ]

    def _wrap_numpy(self, npfun, field_name, concatenate=False, **kwargs):
        """Wrapper to evaluate numpy functions on fields.

        In case the array is concatenated no kwargs are passed as it becomes a 1d array.
        Args:
            npfun (fun): Numpy function to be evaluated on field
            field_name (str): Field name
            concatenate (bool, optional): If array is to be concatenated. Defaults to False.

        Returns:
            npfun function call
        """
        field = getattr(self, field_name).copy()
        if not self._check_if_equal_length(field_name):
            if concatenate:
                field = nested_flatten(field)
            else:
                raise NotImplementedError(
                    f"The {npfun.__name__} computation is currently only available for (n,1)-arrays"
                    ", but the length varies between elements."
                )

        return npfun(field, **kwargs)

    def mean_of_field(self, field_name, concatenate=False, **kwargs):
        """Compute mean on field

        Args:
            field_name (str): Field name
            concatenate (bool, optional): If array is to be concatenated. Defaults to False.

        Returns:
            mean of field
        """
        return self._wrap_numpy(np.mean, field_name, concatenate=concatenate, **kwargs)

    def standard_deviation_of_field(self, field_name, concatenate=False, **kwargs):
        """Standard deviation of field.

        Args:
            field_name (str): Field name
            concatenate (bool, optional): If array is to be concatenated. Defaults to False.

        Returns:
            standard deviation of field
        """
        return self._wrap_numpy(np.std, field_name, concatenate=concatenate, **kwargs)

    def histogram_of_field(self, field_name, concatenate=False, **kwargs):
        """Histogram of field.

        Args:
            field_name (str): Field name
            concatenate (bool, optional): If array is to be concatenated. Defaults to False.

        Returns:
            Histogram of field
        """
        if self._check_if_field_is_nested(field_name) and not concatenate:
            raise NotImplementedError(
                "Histogram is currently only available for (n,1)-arrays."
            )
        return self._wrap_numpy(
            np.histogram, field_name, concatenate=concatenate, **kwargs
        )

    def _check_if_field_is_nested(self, field_name):
        """Check if field is not 1d.

        Args:
            field_name (str): Field name to be checked.

        Returns:
            bool: True if field is not 1d
        """
        field = getattr(self, field_name)
        return not all(isinstance(f, (float, int)) for f in field)

    def _check_if_equal_length(self, field_name):
        """Check if elements of field have equal dimension.

        Args:
            field_name (str): Field name to be checked.

        Returns:
            bool: True if every element has the same dimension.
        """
        if self._check_if_field_is_nested(field_name):
            lens = {safe_len(v) for v in getattr(self, field_name)}
            return len(lens) == 1

        return True

    def add_element(self, new_elements):
        """Add one or multiple elements.

        Args:
            new_elements (self.datatype,iterable): Element(s) to be added.
        """
        for element in make_iterable(new_elements):
            self._add_single_element(element)

    def _append_to_field_array(self, field_name, new_value):
        """Append values to field array.

        Args:
            field_name (str): Field that is to be modified.
            new_value (obj): Value to be added.
        """
        getattr(self, field_name).append(new_value)

    def _add_single_element(self, new_element):
        """Add a single element to self.

        Args:
            new_element (self.datatype): New element to be added.
        """
        if not isinstance(new_element, self.datatype):
            raise TypeError(
                f"Argument must be of type '{self.datatype}' not type '{type(new_element)}'!"
            )
        new_id_start = max(self.id) + 1
        existing_fields = [key for key in self.field_names if key != "id"]
        for field in existing_fields:
            new_value = getattr(new_element, field)
            self._append_to_field_array(field, new_value)
        self._append_to_field_array("id", new_id_start)

    def _update_data_type(self):
        """Update self.datatype.

        Only needed in case fields are added or removed.
        """
        self.datatype = dataclasses.make_dataclass("DataContainer", self.field_names)

    def where(self, condition, index_only=True):
        """Where method.

        Args:
            condition (np.array,list): Containing booleans
            index_only (bool, optional): Only returns the indexes. Defaults to True.

        Returns:
            np.ndarray: If only indexes are used
            type(self): If elements are returned.
        """
        indexes = np.where(condition)[0]
        if index_only:
            return indexes

        return self[indexes]

    def remove_element_by_id(self, element_id, reset_ids=False):
        """Remove elements by id.

        Wraps around remove element with the correct ids.

        Args:
            element_id (int): Id of element to be removed.
            reset_ids (bool, optional): Reset the ids. Defaults to False.
        """
        idx = self._map_id_to_index(element_id)
        self.remove_element(idx, reset_ids=reset_ids)

    def remove_element(self, idx, reset_ids=False):
        """Remove element by index.

        Args:
            idx (int,iterable): Indexes of elements to be removed.
            reset_ids (bool, optional): Reset the ids. Defaults to False.
        """
        for field_name in self.field_names:
            field_array = getattr(self, field_name)
            setattr(
                self, field_name, self._remove_from_field_by_index(field_array, idx)
            )
        if reset_ids:
            self.reset_ids()

    def _remove_from_field_by_index(self, field, idx):
        """Remove entry from field.

        Args:
            field (obj): Field data.
            idx (int): Index to be deleted.

        Returns:
            modified field.
        """
        return field[:idx] + field[idx + 1 :]


class NumpyContainer(Container):
    """Numpy container.

    All field are numpy arrays."""

    def __init__(self, datatype=None, *field_names, **fields):
        """Initialise container.

        Args:
            datatype (obj, optional): Type of elements. Defaults to None.
        """
        super().__init__(
            np.array([]),
            datatype,
            *field_names,
            **fields,
        )
        for field_name in self.field_names:
            field = getattr(self, field_name)
            setattr(self, field_name, transform_to_2d_np_array(field))
        self.id = transform_to_2d_np_array(self.id)

    def _remove_from_field_by_index(self, field, idx):
        """Remove entry from field.

        Args:
            field (obj): Field data.
            idx (int): Index to be deleted.

        Returns:
            modified field.
        """
        return np.delete(field, idx, 0)

    def _append_to_field_array(self, field_name, new_value):
        """Append values to field array.

        Args:
            field_name (str): Field that is to be modified.
            new_value (obj): Value to be added.
        """
        if isinstance(new_value, (int, float)):
            new_value = np.array([new_value])

        if len(getattr(self, field_name)):
            field_shape = getattr(self, field_name)[-1].shape
            new_value = new_value.reshape(field_shape).flatten()
            setattr(
                self, field_name, np.row_stack((getattr(self, field_name), new_value))
            )
        else:
            setattr(self, field_name, np.atleast_2d(new_value))

    def add_field(self, field_name, field):
        """Add field to container.

        Args:
            field_name (str): name of the new field
            field (list): field values
        """
        super().add_field(field_name, transform_to_2d_np_array(field))

    def reset_ids(self):
        """Reset ids to a continuous range."""
        self.id = transform_to_2d_np_array(np.arange(len(self)))

    def _check_if_equal_length(self, field_name):
        """We assume np.ndarrays are always of equal length

        Returns:
            bool: True
        """
        return True

    def _check_if_field_is_nested(self, field_name):
        """Check if field is 1d.

        Args:
            field_name (str): field_name

        Returns:
            bool: True if array is not 1d
        """
        return getattr(self, field_name).shape[1] > 1

    def evaluate_function(self, fun, add_as_field=False, field_name=None):
        """Evaluate function on container.

        This method evaluates the function fun for every element. Define the argument of the
        function is of type self.datatype.

        Args:
            fun (fun): Function to be evaluated
            add_as_field (bool, optional): Should the result be added as field. Defaults to False.
            field_name (str optional): Name of the field to be added.Defaults to None.
        """
        return transform_to_2d_np_array(
            super().evaluate_function(fun, add_as_field, field_name)
        )


def check_if_1d_np_array(array):
    """Check if np.ndarray is 1d.

    Args:
        array (np.ndarray): Array to be checked

    Returns:
        bool: True if 1d
    """
    return len(array.shape) == 1


def transform_to_2d_np_array(array):
    """Transform array to 2d.

    Args:
        array (np.ndarray): Array to be checked.

    Returns:
        np.ndarray: Transformed array.
    """
    if isinstance(array, list):
        array = np.array(array)

    if check_if_1d_np_array(array):
        array = array.reshape(-1, 1)
    return array


def has_len(obj):
    """Checks if obj has len method.

    Args:
        obj (obj): object to be checked

    Returns:
        bool: True if has len method
    """
    return hasattr(obj, "__len__")


def safe_len(obj):
    """Safe len function.

    If attribute has no length method the default length is 1.

    Args:
        obj (obj): Object to be checked.

    Returns:
        int: Length of object
    """
    if has_len(obj):
        return len(obj)

    return 1


def make_iterable(obj):
    """Make object iterable as generator.

    Args:
        obj (obj): object to make iterable.

    Yields:
        values of obj
    """
    if isinstance(obj, Iterable):
        for object_item in obj:
            yield object_item
    else:
        yield obj


def nested_flatten(nested_list):
    """Flatten nested lists.

    Args:
        nested_list (list): List of lists of lists...

    Returns:
        flatten list.
    """
    result = []
    for item in nested_list:
        if not isinstance(item, list):
            result.append(item)
        else:
            result.extend(nested_flatten(item))
    return result
