"""Particle object."""

import numpy as np

from pipapo.utils import sphere_intersection, sphere_plane_intersection
from pipapo.utils.dataclass import has_len


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

    def interface_area_to_particle(self, particle_j):
        """Get the interface area to another particle.

        Args:
            particle_j (Particle): Contact partner to compute the interface area

        Returns:
            interface_area (float): Interface area
        """
        center_to_center = sphere_intersection.get_center_to_center(
            self.position, particle_j.position
        )
        center_to_center = np.sqrt(np.sum(center_to_center**2))
        gap = sphere_intersection.get_gap(
            center_to_center, self.radius, particle_j.radius
        )

        if gap < 0:
            return sphere_intersection.get_interface_area(
                center_to_center,
                self.radius,
                particle_j.radius,
            )

        raise ValueError("Could not compute interface area as there is not contact.")

    def interface_area_to_wall(self, wall_point, wall_normal):
        """Get the interface area to wall.

        Args:
            particle_j (Particle): Contact partner to compute the interface area
            wall_point (np.ndarray): Point on the wall to define the plane
            wall_normal (np.ndarray): Wall normal POINTING INWARDS

        Returns:
            interface_area (float): Interface area
        """
        center_to_wall_point = sphere_plane_intersection.get_center_to_plane_point(
            self.position, wall_point
        )
        center_to_wall_distance = (
            sphere_plane_intersection.get_center_to_plane_distance(
                center_to_wall_point, wall_normal
            )
        )

        gap = sphere_plane_intersection.get_gap(center_to_wall_distance, self.radius)

        if gap < 0:
            return sphere_plane_intersection.get_interface_area(
                center_to_wall_distance, self.radius
            )

        raise ValueError("Could not compute interface area as there is not contact.")
