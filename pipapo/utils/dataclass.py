import warnings
from collections.abc import Iterable
from copy import deepcopy

import numpy as np


class Container:
    def __init__(self, default_intialization, *field_names, **fields):
        if bool(field_names) and bool(fields):
            raise Exception(
                "You provided field_names and fields. You can only provide the one or the other!"
            )

        self.id = default_intialization
        self.field_names = ["id"]

        if field_names:
            self.field_names = list(set(field_names).union(self.field_names))
            for key in field_names:
                setattr(self, key, default_intialization)

        if fields:
            self.field_names = list(set(fields.keys()).union(self.field_names))
            len_array = []
            for key, value in fields.items():
                setattr(self, key, value)
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
        self._current_idx = 0

    def _add_field_name(self, name):
        self.field_names.append(name)
        self.field_names.sort()

    def _values(self):
        self.field_names.sort()
        return [getattr(self, f) for f in self.field_names]

    def _items(self):
        return tuple(zip(self.field_names, self._values()))

    def __len__(self):
        lens = {len(v) for v in self._values()}
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

        new_dict = {key: value[self._current_idx] for key, value in self._items()}
        self._current_idx += 1
        return new_dict

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
                raise IndexError(f"index {i} out of range for size {self.__len__()}")
            new_dict = {key: value[i] for key, value in self._items()}
        elif isinstance(i, list):
            new_dict = {key: [value[j] for j in i] for key, value in self._items()}
        else:
            new_dict = {key: value[i] for key, value in self._items()}
        return new_dict

    def _map_id_to_index(self, id):
        idx = np.where(np.array(self.id) == id)[0]
        if len(idx) > 1:
            raise IndexError(f"More than one element found with id {id}.")
        elif len(idx) == 0:
            raise IndexError(f"Not element with id {id} found!")

        return int(idx[0])

    def reset_ids(self):
        self.id = np.arange(len(self)).tolist()

    def get_by_id(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        else:
            try:
                ids = list(ids)
            except Exception as exc:
                raise Exception("Could not convert ids to list.") from exc

        indexes = []
        for idx in ids:
            indexes.extend(self._map_id_to_index(idx))

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
            {key: value[i] for key, value in self._items()} for i in range(len(self))
        ]

    def mean_of_field(self, field_name, **kwargs):
        return np.mean(getattr(self, field_name), **kwargs)

    def standard_deviation_of_field(self, field_name, **kwargs):
        return np.std(getattr(self, field_name), **kwargs)

    def histogram_of_field(self, field_name, **kwargs):
        return np.histogram(getattr(self, field_name), **kwargs)

    def add_element(self, new_elements):
        if isinstance(new_elements, type(self)):
            for element in new_elements:
                self._add_element(element)
        else:
            self._add_element(new_elements)

    def _append_to_field_array(self, field, new_value):
        getattr(self, field).append(new_value)

    def _add_element(self, new_element):
        new_id_start = max(self.id)
        existing_fields = [key for key in self.field_names if key != "id"]
        for field in existing_fields:
            new_value = getattr(new_element, field)
            self._append_to_field_array(field, new_value)

        self._append_to_field_array("id", new_id_start + 1)


class NumpyContainer(Container):
    def __init__(self, datatype, *field_names, **fields):
        super().__init__(np.array([]), *field_names, **fields)
        self.datatype = datatype
        self.id = np.array(self.id).astype(int).reshape(-1, 1)
        for field_name in self.field_names:
            field = getattr(self, field_name)
            if check_if_1d_np_array(field):
                setattr(self, field_name, transform_to_2d_np_array(field))

    def __next__(self):
        return self.datatype(**super().__next__())

    def __getitem__(self, i):
        # Create the correct types
        if isinstance(i, int):
            return self.datatype(**super().__getitem__(i))
        else:
            return type(self)(**super().__getitem__(i))

    def __list__(self):
        return [self.datatype(**dictionary) for dictionary in super().__list__()]

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
            setattr(self, field, np.delete(field_array, idx, 0))
        if reset_ids:
            self.reset_ids()

    def _append_to_field_array(self, field, new_value):
        if isinstance(new_value, (int, float)):
            new_value = np.array([new_value])
        if len(getattr(self, field)):
            field_shape = getattr(self, field)[-1]
            new_value = new_value.reshape(field_shape.shape).flatten()
            setattr(self, field, np.row_stack((getattr(self, field), new_value)))
        else:
            setattr(self, field, np.atleast_2d(new_value))

    def add_element(self, new_elements):
        if not isinstance(new_elements, (self.datatype, type(self))):
            raise TypeError(
                f"Argument must be of type '{self.datatype}' or '{type(self)}' not {type(new_elements)}"
            )
        if isinstance(new_elements, self.datatype):
            new_elements = [new_elements]

        new_id_start = len(self)
        existing_fields = [key for key in self.field_names if key != "id"]
        for i, element in enumerate(new_elements):
            for field in existing_fields:
                new_value = getattr(element, field)
                self._append_to_field_array(field, new_value)

            self._append_to_field_array("id", i + new_id_start)

    def add_field(self, field_name, field):
        super().add_field(field_name, transform_to_2d_np_array(field))

    def reset_ids(self):
        super().reset_ids()
        self.id = transform_to_2d_np_array(self.id)


def check_if_1d_np_array(array):
    return len(array.shape) == 1


def transform_to_2d_np_array(array):
    if isinstance(array, list):
        array = np.array(array)

    if check_if_1d_np_array(array):
        array = array.reshape(-1, 1)
    return array
