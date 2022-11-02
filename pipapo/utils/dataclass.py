import dataclasses
import warnings
from collections.abc import Iterable
from copy import deepcopy

import numpy as np


class Container:
    def __init__(
        self, default_intialization=None, datatype=None, *field_names, **fields
    ):
        if bool(field_names) and bool(fields):
            raise Exception(
                "You provided field_names and fields. You can only provide the one or the other!"
            )
        if default_intialization is None:
            default_intialization = []

        self.id = default_intialization
        self.field_names = ["id"]

        if field_names:
            self.field_names = list(set(field_names).union(self.field_names))
            for key in field_names:
                setattr(self, key, default_intialization.copy())

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
        self.field_names.append(name)
        self.field_names.sort()
        self._update_data_type()

    def _values(self):
        self.field_names.sort()
        return [getattr(self, f) for f in self.field_names]

    def _items(self):
        return tuple(zip(self.field_names, self._values()))

    def __len__(self):
        lens = {safe_len(v) for v in self._values()}
        if len(lens) == 1:
            return list(lens)[0]
        elif len(lens) == 0:
            return 0

        field_lengths = ", ".join([f"{k}: {len(v)}" for k, v in self._items()])
        raise ValueError(f"Different lengths between fields: {field_lengths}")

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return self.copy()

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return dict(self._items())

    def __next__(self):
        if self._current_idx >= len(self):
            raise StopIteration()

        item = self.datatype(
            **{key: value[self._current_idx] for key, value in self._items()}
        )
        self._current_idx += 1
        return item

    def add_field(self, field_name, field):
        if not len(field) == len(self):
            raise Exception(
                f"Dimension mismatch while adding new field. Current length is {len(self)} but the field you want to add has length {len(field)}"
            )

        setattr(self, field_name, field.copy())
        self._add_field_name(field_name)

    def remove_field(self, field_name):
        if not field_name in self.field_names:
            warnings.warn(f"Field {field_name} does not exist, nothing was deleted!")
        else:
            self.field_names.remove(field_name)
            delattr(self, field_name)
            self._update_data_type()

    def evaluate_function(self, fun, add_as_field=False, field_name=None):
        result = []
        for s in self:
            result.append(fun(s))
        if add_as_field:
            if not field_name:
                field_name = fun.__name__
                if field_name == "<lambda>":
                    raise NameError(
                        "You are using a lambda function. Please add a name to the field you want to add with the keyword 'field_name'"
                    )
            self.add_field(field_name, result)
        return result

    def __getitem__(self, i):
        if isinstance(i, int):
            if i > len(self):
                raise IndexError(f"Index {i} out of range for size {self.__len__()}")
            item = self.datatype(**{key: value[i] for key, value in self._items()})
        elif isinstance(i, list):
            item = type(self)(
                **{key: [value[j] for j in i] for key, value in self._items()}
            )
        else:
            item = type(self)(**{key: value[i] for key, value in self._items()})
        return item

    def _map_id_to_index(self, id):
        idx = np.where(np.array(self.id) == id)[0]
        if len(idx) > 1:
            raise IndexError(f"More than one element found with id {int(id)}.")
        elif len(idx) == 0:
            raise IndexError(f"No element with id {int(id)} found!")

        return int(idx[0])

    def reset_ids(self):
        self.id = np.arange(len(self)).tolist()

    def get_by_id(self, ids):
        if isinstance(ids, int):
            return self[self._map_id_to_index(ids)]
        elif not isinstance(ids, Iterable):
            raise TypeError(
                f"Ids need to be iterable (list, tuple, ...) or an int but which type {type(ids)} is not."
            )

        indexes = []
        for idx in ids:
            indexes.append(self._map_id_to_index(idx))

        return self[indexes]

    def update_by_id(self, element, id):
        idx = self._map_id_to_index(id)

        for field, value in element.items():
            if field == "id":
                continue
            field_array = getattr(self, field)
            field_array[idx] = value

    def __list__(self):
        return [
            self.datatype(
                {key: value[i] for key, value in self._items()}
                for i in range(len(self))
            )
        ]

    def mean_of_field(self, field_name, **kwargs):
        if not self._check_if_equal_length(field_name):
            raise NotImplementedError(
                "The mean computation is currently only avaible for (n,1)-arrays, but the length varies between elements."
            )
        return np.mean(getattr(self, field_name), **kwargs)

    def standard_deviation_of_field(self, field_name, **kwargs):
        if not self._check_if_equal_length(field_name):
            raise NotImplementedError(
                "The standard deviation computation is currently only avaible for (n,1)-array, but the length varies between elements."
            )
        return np.std(getattr(self, field_name), **kwargs)

    def histogram_of_field(self, field_name, **kwargs):
        if self._check_if_field_is_nested(field_name):
            raise NotImplementedError(
                "Histogram is currently only avaible for (n,1)-arrays."
            )
        return np.histogram(getattr(self, field_name), **kwargs)

    def _check_if_field_is_nested(self, field_name):
        field = getattr(self, field_name)
        return not all(isinstance(f, (float, int)) for f in field)

    def _check_if_equal_length(self, field_name):
        if self._check_if_field_is_nested(field_name):
            lens = {safe_len(v) for v in getattr(self, field_name)}
            return len(lens) == 1
        else:
            return True

    def add_element(self, new_elements):
        if not isinstance(new_elements, (self.datatype, type(self))):
            raise TypeError(
                f"Argument must be of type '{self.datatype}' or '{type(self)}' not {type(new_elements)}"
            )
        for element in make_iterable(new_elements):
            self._add_element(element)

    def _append_to_field_array(self, field, new_value):
        getattr(self, field).append(new_value)

    def _add_element(self, new_element):
        new_id_start = max(self.id) + 1
        existing_fields = [key for key in self.field_names if key != "id"]
        for field in existing_fields:
            new_value = getattr(new_element, field)
            self._append_to_field_array(field, new_value)
        self._append_to_field_array("id", new_id_start)

    def _update_data_type(self):
        self.datatype = dataclasses.make_dataclass("DataContainer", self.field_names)

    def where(self, condition, index_only=True):
        indexes = np.where(condition)[0]
        if index_only:
            return indexes
        else:
            return self[indexes]

    def remove_element_by_id(self, id, reset_ids=False):
        idx = self._map_id_to_index(id)
        for field in self.field_names:
            field_array = getattr(self, field)
            setattr(self, field, self._remove_from_array_by_index(field_array, idx))
        if reset_ids:
            self.reset_ids()

    def _remove_from_array_by_index(self, array, idx):
        return array[:idx] + array[idx + 1 :]


class NumpyContainer(Container):
    def __init__(self, datatype=None, *field_names, **fields):
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

    def _remove_from_array_by_index(self, array, idx):
        return np.delete(array, idx, 0)

    def _append_to_field_array(self, field, new_value):
        if isinstance(new_value, (int, float)):
            new_value = np.array([new_value])

        if len(getattr(self, field)):
            field_shape = getattr(self, field)[-1].shape
            new_value = new_value.reshape(field_shape).flatten()
            setattr(self, field, np.row_stack((getattr(self, field), new_value)))
        else:
            setattr(self, field, np.atleast_2d(new_value))

    def add_field(self, field_name, field):
        super().add_field(field_name, transform_to_2d_np_array(field))

    def reset_ids(self):
        self.id = transform_to_2d_np_array(np.arange(len(self)))

    def _check_if_equal_length(self, field_name):
        return True

    def _check_if_field_is_nested(self, field_name):
        return getattr(self, field_name).shape[1] > 1


def check_if_1d_np_array(array):
    return len(array.shape) == 1


def transform_to_2d_np_array(array):
    if isinstance(array, list):
        array = np.array(array)

    if check_if_1d_np_array(array):
        array = array.reshape(-1, 1)
    return array


def safe_len(obj):
    if hasattr(obj, "__len__"):
        return len(obj)
    else:
        return 1


def make_iterable(obj):
    if isinstance(obj, Iterable):
        for o in obj:
            yield o
    else:
        yield obj
