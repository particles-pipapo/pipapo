"""Particle classes."""
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv

from pipapo.utils.binning import get_bounding_box
from pipapo.utils.csv import export_csv, import_csv
from pipapo.utils.dataclass import Container, NumpyContainer, has_len
from pipapo.utils.vtk import export_vtk, import_vtk


class Particle:
    """Particle object."""

    def __init__(self, position, radius, **fields):
        """Initialise particle object.

        Args:
            position (np.array): Particle center position
            radius (float): Particle radius
        """
        self.position = position
        self.radius = float(radius)
        for key, value in fields.items():
            if has_len(value) and len(value) == 1:
                value = value[0]

            setattr(self, key, value)

    def evaluate_function(self, fun, add_as_field=False, field_name=None):
        """Evaluate function on particle data.

        Args:
            fun (fun): function to be evaluated on the particle data.
            add_as_field (bool, optional): True if new field is to be added. Defaults to False.
            field_name (str, optional): Name of the new field. Defaults to None.
        Returns:
            function evaluation result
        """
        result = fun(self)
        if add_as_field:
            if not field_name:
                field_name = fun.__name__
                if field_name == "<lambda>":
                    raise NameError(
                        "You are using a lambda function. Please add a name to the field you want"
                        "to add with the keyword 'field_name'"
                    )
            setattr(self, field_name, result)

        return result

    def items(self):
        """Particle items.

        Returns:
            dict_items: Name and field values
        """
        return self.__dict__.items()

    def volume(self, add_as_field=False):
        """Volume of the particle.

        Args:
            add_as_field (bool, optional): True if the field is to be added. Defaults to False.

        Returns:
            float: particle volume
        """
        # pylint: disable=C3001
        volume_sphere = lambda p: 4 * np.pi / 3 * p.radius * p.radius * p.radius
        # pylint: enable=C3001
        return self.evaluate_function(volume_sphere, add_as_field, field_name="volume")

    def surface(self, add_as_field=False):
        """Surface of the particle.

        Args:
            add_as_field (bool, optional): True if the field is to be added. Defaults to False.

        Returns:
            float: particle surface
        """
        surface_sphere = (
            lambda p: 4 * np.pi * p.radius * p.radius  # pylint: disable=C3001
        )
        return self.evaluate_function(
            surface_sphere, add_as_field=add_as_field, field_name="surface"
        )

    def __str__(self):
        """Particle description."""
        string = "\npipapo particle\n"
        string += "\n".join([f"  {key}: {value}" for key, value in self.items()])
        return string


# Id is handled by the dataclass
MANDATORY_FIELDS = ["position", "radius"]


class ParticleContainer(NumpyContainer):
    """Particles container."""

    def __init__(self, *field_names, **fields):
        """Initialise particles container.

        Args:
            field_names (list): Field names list
            fields (dict): Dictionary with field names and values.
        """
        self.position = None
        self.radius = None
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
        """Particle container descriptions."""
        string = "\npipapo particles set\n"
        string += f"  with {len(self)} particles\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def volume_sum(self):
        """Sum of all particles volume.

        Returns:
            float: Volume
        """
        return np.sum([p.volume() for p in self])

    def surface_sum(self):
        """Sum of all particles surface.

        Returns:
            float: Surface
        """
        return np.sum([p.surface() for p in self])

    def export(self, file_path):
        """Export particles.

        Data type is selected based on the ending of file_path.

        Args:
            file_path (str): Path were to store the files
        """
        if self:
            file_path = Path(file_path)
            if file_path.suffix in [".vtk", ".vtp"]:
                export_vtk(self.to_dict(), file_path)
            elif file_path.suffix == ".csv":
                export_csv(self.to_dict(), file_path)
            else:
                raise TypeError(
                    f"Filetype {file_path.suffix} unknown. Supported export file types are vtk or "
                    "csv."
                )
        else:
            warnings.warn("Empty particles set, nothing was exported.")

    @classmethod
    def from_vtk(cls, file_path, radius_keyword="radius", diameter_keyword=None):
        """Create particle container from vtk.

        Args:
            file_path (str): Path to vtk file
            radius_keyword (str, optional): Radius name in the file. Defaults to "radius"
            diameter_keyword (str, optional): Diameter name in the file. Defaults to None

        Returns:
            ParticleContainer: particles container based on the vtk file.
        """
        dictionary = import_vtk(file_path)
        if not diameter_keyword:
            if radius_keyword in dictionary:
                dictionary["radius"] = dictionary[radius_keyword]
            else:
                raise KeyError(
                    f"Field '{radius_keyword}' not found in {Path(file_path).resolve()}! Available"
                    f" fields are {', '.join(list(dictionary.keys()))}"
                )
        else:
            if diameter_keyword in dictionary:
                dictionary["radius"] = dictionary[diameter_keyword] / 2
            else:
                raise KeyError(
                    f"Field '{diameter_keyword}' not found in {Path(file_path).resolve()}! "
                    f"Available fields are {', '.join(list(dictionary.keys()))}"
                )

        return cls(**dictionary)

    @classmethod
    def from_csv(
        cls,
        file_path,
        radius_keyword="radius",
        diameter_keyword=None,
        position_keywords=None,
    ):
        """Create particle container from csv.

        Args:
            file_path (str): Path to csv file
            radius_keyword (str, optional): Radius name in the file. Defaults to "radius"
            diameter_keyword (str, optional): Diameter name in the file. Defaults to None

        Returns:
            ParticleContainer: particles container based on the csv file.
        """
        if position_keywords is None:
            position_keywords = ["x", "y", "z"]
        dictionary = import_csv(file_path)
        if not diameter_keyword:
            if radius_keyword in dictionary:
                dictionary["radius"] = dictionary[radius_keyword]
            else:
                raise KeyError(
                    f"Field '{radius_keyword}' not found in {Path(file_path).resolve()}! Available"
                    f"fields are {', '.join(list(dictionary.keys()))}"
                )
        else:
            if diameter_keyword in dictionary:
                dictionary["radius"] = dictionary[diameter_keyword] / 2
            else:
                raise KeyError(
                    f"Field '{diameter_keyword}' not found in {Path(file_path).resolve()}! "
                    f"Available fields are {', '.join(list(dictionary.keys()))}"
                )
        position = np.column_stack([dictionary[k] for k in position_keywords])
        dictionary["position"] = position

        for k in position_keywords:
            dictionary.pop(k)
        return cls(**dictionary)

    def particle_center_in_box(self, box_dimensions, box_center=None, index_only=False):
        """Check if particles are with the box.

        Args:
            box_dimensions (iterable): Dimensions of the box
            box_center (iterable, optional): Box center coordinates. Defaults to None.
            index_only (bool, optional): Only return the indices. Defaults to False.
        """

        def componentwise_in_box(
            component, component_box_dimension, component_box_center
        ):
            left_side = component >= (
                component_box_center - 0.5 * component_box_dimension
            )
            right_side = component <= (
                component_box_center + 0.5 * component_box_dimension
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
        """Get bounding box.

        Outer bounding box can be selected with the keyword `by_position`. Otherwise only the
        particle centers are checked.

        Args:
            by_position (bool, optional): Outer bounding box. Defaults to False.

        Returns:
            center: Center of the bounding box
            lengths: Dimension of the bounding box
        """
        if by_position:
            position = self.position
        else:
            position = self.position - self.radius
            position = np.row_stack((position, self.position + self.radius))
        return get_bounding_box(position)

    def update_particle(self, particle, particle_id=None):
        """Update particle.

        Args:
            particle (Particle): New particle data
            particle_id (int, optional): Id for which particle to update. Defaults to None.
        """
        if not particle_id:
            try:
                particle_id = getattr(particle, "id")
            except AttributeError as exc:
                raise AttributeError(
                    "id was not provided and no id could be found in the provided particle object"
                ) from exc

        super().update_by_id(particle, particle_id)

    def remove_particle_by_id(self, particle_id, reset_ids=False):
        """Remove particle from container.

        Args:
            particle_id (int): Particle index to be deleted
            reset_ids (bool, optional): Rest the ids in sequential fashion. Defaults to False.
        """
        super().remove_element_by_id(particle_id, reset_ids=reset_ids)

    def add_particle(self, new_particles):
        """Add particle(s)

        Args:
            new_particles (Particle, ParticleContainer): New data to be added
        """
        super().add_element(new_particles)

    def remove_field(self, field_name):
        """Remove field from the container.

        Args:
            field_name (str): Name of the field to be removed
        """
        fields = MANDATORY_FIELDS + ["id"]
        if field_name in fields:
            raise Exception(
                f"Mandatory fields: {', '.join(fields)} can not be deleted!"
            )
        super().remove_field(field_name)

    def get_contacts(self):
        """Get contact partners.

        Returns:
            ContactContainer: Container with contact data
        """
        coordination_number = [0] * len(self)
        contact_partners_ids = [[] for i in range(len(self))]
        gaps = [[] for i in range(len(self))]

        for i, (radius_i, position_i) in enumerate(zip(self.radius, self.position)):
            for j, (radius_j, position_j) in enumerate(zip(self.radius, self.position)):
                if j <= i:
                    continue
                radius_sum = radius_i + radius_j
                distance = np.sqrt(np.sum((position_i - position_j) ** 2))
                gap = distance - radius_sum
                if gap < 0:
                    coordination_number[i] += 1
                    contact_partners_ids[i].append(j)
                    gaps[i].append(gap)
                    coordination_number[j] += 1
                    contact_partners_ids[j].append(i)
                    gaps[j].append(gap)
        return ContactContainer(
            coordination_number=coordination_number,
            contact_partners_ids=contact_partners_ids,
            gaps=gaps,
        )

    def _update_data_type(self):
        """Needs to be overwritten."""

    def get_pv_spheres(self):
        """Get pyvista spheres.

        Returns:
            list: List of all the pyvista spheres
        """
        spheres = []
        for i in range(len(self)):
            spheres.append(pv.Sphere(self.radius[i], self.position[i])) # pylint: disable=E1136
        return spheres

    def plot(self, color="green", pv_plotter=None, show=True):
        """Plot spheres.

        Args:
            color (str, optional): Color of the particles. Defaults to "green".
            pv_plotter (pv.Plotter, optional): Plotter object to plot. Defaults to None.
            show (bool, optional): Open the plotting window. Defaults to True.

        Returns:
            pv.Plotter: Plotter object
        """
        if not pv_plotter:
            pv_plotter = pv.Plotter()

        for sphere in self.get_pv_spheres():
            pv_plotter.add_mesh(sphere, color=color)

        if show:
            pv_plotter.show()
        return pv_plotter


class ContactContainer(Container):
    """Container for contact data."""

    def __init__(self, **fields):
        """Initialise contact container.

        Args:
            coordination_number (list, optional): Coordination number of contact pairs. Defaults to
                                                  None.
            contact_partners_ids (list, optional): List of contact partners. Defaults to None.
            gaps (list, optional): Gaps for each contact pair. Defaults to None.
        """
        super().__init__(self, **fields)

    def get_isolated_particles_id(self):
        """Array with isolated particles."""
        return self.where(self.coordination_number == 0)  # pylint: disable=E1101
