import numpy as np
from pathlib import Path
from pipapo.utils.vtk import dictionary_from_vtk_file, vtk_file_from_dictionary

from pipapo.utils.csv import export_csv
import warnings


class Particle:
    def __init__(self, position, radius, **fields):
        self.position = position
        self.radius = radius
        for key, value in fields.items():
            setattr(self, key, value)

    def evaluate_function(self, fun, add_as_field=False, field_name=None):
        result = fun(self)
        if add_as_field:
            if not field_name:
                field_name = fun.__name__
            setattr(self, field_name, result)

        return result

    def items(self):
        return self.__dict__.items()

    def volume(self, add_as_field=False):
        volume_sphere = lambda p: 4 * np.pi / 3 * p.radius * p.radius * p.radius
        return self.evaluate_function(volume_sphere, add_as_field, field_name="volume")

    def surface(self, add_as_field=False):
        surface_sphere = lambda p: 4 * np.pi * p.radius * p.radius
        return self.evaluate_function(
            surface_sphere, add_as_field=True, field_name="surface"
        )

    def __str__(self):
        string = "\npipapo particle\n"
        string += "\n".join([f"  {key}: {value}" for key, value in self.items()])
        return string


MANDOTORY_FIELDS = ["id", "position", "radius"]


class Particles:
    def __init__(self, *field_names, **fields):
        field_names = MANDOTORY_FIELDS + list(
            set(field_names).difference(MANDOTORY_FIELDS)
        )
        self.field_names = []
        for key in field_names:
            setattr(self, key, [])
            self.field_names.append(key)
        for key, value in fields.items():
            setattr(self, key, value)
            self.field_names.append(key)

        self._current_idx = 0

    def _add_field_name(self, name):
        self.field_names.append(name)

    def _values(self):
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

    def add_particle(self, new_particles):
        if not isinstance(new_particles, (Particle, Particles)):
            raise TypeError(
                "new_particles must be of type 'Particle' or 'Particles' not %s"
                % type(new_particles)
            )
        if isinstance(new_particles, Particle):
            new_particles = [new_particles]

        new_id_start = len(self)
        existing_fields = [key for key in self.field_names if key != "id"]
        for i, particle in enumerate(new_particles):
            for field in existing_fields:
                new_value = getattr(particle, field)
                self._append_to_field_array(field, new_value)

            self._append_to_field_array("id", i + new_id_start)

    def _append_to_field_array(self, field, new_value):
        if isinstance(new_value, (int, float)):
            new_value = np.array([new_value])
        field_shape = getattr(self, field)[-1]
        new_value = new_value.reshape(field_shape.shape).flatten()
        setattr(self, field, np.row_stack((getattr(self, field), new_value)))

    def __getitem__(self, i):
        if isinstance(i, int):
            if i > len(self):
                raise IndexError(f"index {i} out of range for size {self.__len__()}")
            new_dict = {key: value[i] for key, value in self._items()}
            return Particle(**new_dict)

        new_dict = {key: value[i] for key, value in self._items()}
        return Particles(**new_dict)

    def __str__(self):
        string = "\npipapo particles set\n"
        string += f"  with {len(self)} particles\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return self.copy()

    def copy(self):
        return Particles(**self.to_dict())

    def __next__(self):
        if self._current_idx >= len(self):
            raise StopIteration()

        new_dict = {key: value[self._current_idx] for key, value in self._items()}
        self._current_idx += 1
        return Particle(**new_dict)

    def __list__(self):
        return [
            Particle(**{key: value[i] for key, value in self._items()})
            for i in range(len(self))
        ]

    def evaluate_function(self, fun, add_as_field=False, field_name=None):
        result = []
        for p in self:
            result.append(fun(p))
        result = np.array(result)
        if add_as_field:
            if not field_name:
                field_name = fun.__name__
            setattr(self, field_name, result)
            self._add_field_name(field_name)
        return result

    def volume_sum(self):
        return np.sum([p.volume() for p in self])

    def surface_sum(self):
        return np.sum([p.surface() for p in self])

    def to_dict(self):
        return dict(self._items())

    def export(self, file_path):
        if self:
            file_path = Path(file_path)
            if file_path.suffix == ".vtk":
                vtk_file_from_dictionary(self.to_dict(), file_path)
            elif file_path.suffix == ".csv":
                export_csv(self.to_dict(), file_path)
            else:
                raise TypeError(
                    f"Filetype {file_path.suffix} unknown. Supported export file types are vtk or csv."
                )
        else:
            warnings.warn("Empty particles set, nothing was exported.")

    @classmethod
    def from_vtk(cls, file_path):
        dictionary = dictionary_from_vtk_file(file_path)
        if not "radius" in dictionary.keys():
            if "diameter" in dictionary.keys():
                dictionary["radius"] = dictionary["diameter"] / 2
            else:
                raise KeyError(
                    f"No 'radius' or 'diameter' found in {Path(file_path).resolve()}!"
                )
        return cls(**dictionary)
