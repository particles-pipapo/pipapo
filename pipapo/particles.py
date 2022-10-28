import warnings
from pathlib import Path

import numpy as np

from pipapo.utils.bining import get_bounding_box
from pipapo.utils.csv import export_csv, import_csv
from pipapo.utils.dataclass import NumpyContainer
from pipapo.utils.vtk import dictionary_from_vtk_file, vtk_file_from_dictionary


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
                if field_name == "<lambda>":
                    raise NameError(
                        "You are using a lambda function. Please add a name to the field you want to add with the keyword 'field_name'"
                    )
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


# Id is handeld by the dataclass
MANDATORY_FIELDS = ["position", "radius"]


class ParticleContainer(NumpyContainer):
    def __init__(self, *field_names, **fields):
        if fields:
            if not set(MANDATORY_FIELDS).intersection(list(fields.keys())) == set(
                MANDATORY_FIELDS
            ):
                raise Exception(
                    f"The mandatory fields {', '.join(MANDATORY_FIELDS)} were not provided."
                )
        else:
            field_names = list(set(field_names).union(MANDATORY_FIELDS))
        super().__init__(Particle, *field_names, **fields)

    def __str__(self):
        string = "\npipapo particles set\n"
        string += f"  with {len(self)} particles\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def volume_sum(self):
        return np.sum([p.volume() for p in self])

    def surface_sum(self):
        return np.sum([p.surface() for p in self])

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
    def from_vtk(cls, file_path, radius_keyword="radius", diameter_keyword=None):
        dictionary = dictionary_from_vtk_file(file_path)
        if not diameter_keyword:
            if radius_keyword in dictionary.keys():
                dictionary["radius"] = dictionary[radius_keyword]
            else:
                raise KeyError(
                    f"Field '{radius_keyword}' not found in {Path(file_path).resolve()}! Available fields are {', '.join(list(dictionary.keys()))}"
                )
        else:
            if diameter_keyword in dictionary.keys():
                dictionary["radius"] = dictionary[diameter_keyword] / 2
            else:
                raise KeyError(
                    f"Field '{diameter_keyword}' not found in {Path(file_path).resolve()}! Available fields are {', '.join(list(dictionary.keys()))}"
                )

        return cls(**dictionary)

    @classmethod
    def from_csv(
        cls,
        file_path,
        radius_keyword="radius",
        diameter_keyword=None,
        position_keywords=["x", "y", "z"],
    ):
        dictionary = import_csv(file_path)
        if not diameter_keyword:
            if radius_keyword in dictionary.keys():
                dictionary["radius"] = dictionary[radius_keyword]
            else:
                raise KeyError(
                    f"Field '{radius_keyword}' not found in {Path(file_path).resolve()}! Available fields are {', '.join(list(dictionary.keys()))}"
                )
        else:
            if diameter_keyword in dictionary.keys():
                dictionary["radius"] = dictionary[diameter_keyword] / 2
            else:
                raise KeyError(
                    f"Field '{diameter_keyword}' not found in {Path(file_path).resolve()}! Available fields are {', '.join(list(dictionary.keys()))}"
                )
        position = np.column_stack([dictionary[k] for k in position_keywords])
        dictionary["position"] = position

        for k in position_keywords:
            dictionary.pop(k)
        return cls(**dictionary)

    def particle_center_in_box(self, box_dimensions, box_center=None, index_only=False):
        def componentwise_in_box(
            compoment, component_box_dimension, compoment_box_center
        ):
            left_side = compoment >= (
                compoment_box_center - 0.5 * component_box_dimension
            )
            right_side = compoment <= (
                compoment_box_center + 0.5 * component_box_dimension
            )
            return np.logical_and(left_side, right_side)

        conditions = True
        if box_center is None:
            box_center = np.zeros(len(box_dimensions))
        for particles_component, box_dimension, box_center_component in zip(
            self.position.T, box_dimensions, box_center
        ):
            conditions = np.logical_and(
                componentwise_in_box(
                    particles_component, box_dimension, box_center_component
                ),
                conditions,
            )

        return self.where(conditions, index_only)

    def bounding_box(self, by_position=False):
        if by_position:
            position = self.position
        else:
            position = self.position - self.radius
            position = np.row_stack((position, self.position + self.radius))
        return get_bounding_box(position)

    def update_particle(self, particle, id=None):
        if not id:
            try:
                id = getattr(particle, "id")
            except AttributeError as exc:
                raise AttributeError(
                    "id was not provided and no id could be found in the provided particle object"
                ) from exc

        super().update_by_id(particle, id)

    def remove_particle_by_id(self, id, reset_ids=False):
        super().remove_element_by_id(id, reset_ids=reset_ids)

    def add_particle(self, new_particles):
        super().add_element(new_particles)

    def remove_field(self, field_name):
        if field_name in (fields := MANDATORY_FIELDS + ["id"]):
            raise Exception(
                f"Mandatory fields: {', '.join(fields)} can not be deleted!"
            )
        super().remove_field(field_name)
