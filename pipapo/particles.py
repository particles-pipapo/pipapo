import jax.numpy as jnp


class particles:
    def __init__(self, *field_names, **fields):
        for key in field_names:
            self.__dict__.update({key: []})
        self.__dict__.update(fields)

    def __len__(self):
        lens = list({len(v) for v in self.__dict__.values()})

        if len(lens) == 1:
            return lens[0]
        elif len(lens) == 0:
            return 0

        raise ValueError("Different lengths between fields")

    def __getitem__(self, i):
        if isinstance(i, int):
            if i > len(self):
                raise IndexError(f"index {i} out of range for size {self.__len__()}")

        new_dict = {key: value[i] for key, value in self.__dict__.items()}
        return particles(**new_dict)

    def __str__(self):
        string = "Particles set\n"
        string += f"  with {len(self)} particles\n"
        string += f"  with fields: {', '.join(list(self.__dict__.keys()))}"
        return string

    def __bool__(self):
        return len(self)

    def __iter__(self):
        return self

    def __next__(self):
        pass
