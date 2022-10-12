import numpy as np


class Particle:
    def __init__(self, position, radius, **fields):
        self.__dict__.update({"position": position, "radius": radius})
        self.__dict__.update(fields)

    def run_function(self, fun, add_as_field=False, field_name=None):
        result = fun(self)
        if add_as_field:
            if not field_name:
                field_name = fun.__name__
            self.__dict__.update({field_name: result})
        return result

    def volume(self, add_as_field=False):
        volume_sphere = lambda p: 4 * np.pi / 3 * p.radius * p.radius * p.radius
        return self.run_function(volume_sphere, add_as_field, field_name="volume")

    def surface(self, add_as_field=False):
        volume_sphere = lambda p: 4 * np.pi * p.radius * p.radius
        return self.run_function(volume_sphere, add_as_field=True, field_name="surface")

    def __str__(self):
        string = "\npipapo particle\n"
        string += "\n".join(
            [f"  {key}: {value}" for key, value in self.__dict__.items()]
        )
        return string


MANDOTORY_FIELDS = ["id", "position", "radius"]


class Particles:
    def __init__(self, *field_names, **fields):
        field_names = MANDOTORY_FIELDS + list(
            set(field_names).difference(MANDOTORY_FIELDS)
        )
        for key in field_names:
            self.__dict__.update({key: []})
        self.__dict__.update(fields)
        self.field_names = list(self.__dict__.keys())
        self._current_idx = 0

    def _add_field_name(self, name):
        self.field_names.append(name)

    def _values(self):
        return [self.__dict__[f] for f in self.field_names]

    def items(self):
        return tuple(zip(self.field_names, self._values()))

    def __len__(self):
        lens = {len(v) for v in self._values()}
        if len(lens) == 1:
            return list(lens)[0]
        elif len(lens) == 0:
            return 0

        field_lengths = ", ".join([f"{k}: {len(v)}" for k, v in self.items()])
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
        existing_fields_shape = [
            value[-1] for key, value in self.items() if key != "id"
        ]
        for i, particle in enumerate(new_particles):
            for field, field_shape in zip(existing_fields, existing_fields_shape):
                new_value = getattr(particle, field)
                if isinstance(new_value, (int, float)):
                    new_value = np.array([new_value])
                new_value = new_value.reshape(field_shape.shape).flatten()
                self.__dict__[field] = np.row_stack(
                    (
                        self.__dict__[field],
                        new_value,
                    )
                )
            self.__dict__["id"] = np.row_stack(
                (
                    self.__dict__["id"],
                    np.array([i + new_id_start]),
                )
            )

    def __getitem__(self, i):
        if isinstance(i, int):
            if i > len(self):
                raise IndexError(f"index {i} out of range for size {self.__len__()}")
            new_dict = {key: value[i] for key, value in self.items()}
            return Particle(**new_dict)

        new_dict = {key: value[i] for key, value in self.items()}
        return Particles(**new_dict)

    def __str__(self):
        string = "\npipapo particles set\n"
        string += f"  with {len(self)} particles\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def __bool__(self):
        return len(self)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= len(self):
            self._current_idx = 0
            raise StopIteration()

        particle = self[self._current_idx]
        self._current_idx += 1
        return particle

    def run_function(self, fun, add_as_field=False, field_name=None):
        result = []
        for i, p in enumerate(self):
            result.append(fun(p))
        result = np.array(result)
        if add_as_field:
            if not field_name:
                field_name = fun.__name__
            self.__dict__.update({field_name: result})
            self._add_field_name(field_name)
        return result

    def volume_sum(self):
        volume = 0
        for p in self:
            volume += p.volume()
        return float(volume)

    def surface_sum(self):
        surface = 0
        for p in self:
            surface += p.surface()
        return float(surface)
