"""Geometry utils."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from pipapo.utils import sphere_plane_intersection
from pipapo.utils.interfaces import InterfaceContainer
from pipapo.utils.type_hinting import float_np_array

if TYPE_CHECKING:
    from pipapo.particles import ParticleContainer


# pylint: disable=too-few-public-methods
class Domain(Protocol):
    """Domain geometry."""

    def get_particles_in_domain(
        self,
        particles: ParticleContainer,
    ) -> ParticleContainer:
        """Get particles inside the domain.

        Args:
            particles: Particles to check for inclusion

        Returns:
            Particles inside the domain
        """


@dataclass
class Box(Domain):
    """Axis-aligned box.

    Attributes:
        center: Center of the box
        lengths: Lengths of the box edges
    """

    center: float_np_array
    lengths: float_np_array

    def get_particles_in_domain(
        self, particles: ParticleContainer[Any]
    ) -> ParticleContainer[Any]:
        """Get particles inside the box.

        Args:
            particles: Particles to check for inclusion

        Returns:
            Particles inside the box
        """
        in_box_mask = np.all(
            np.abs(particles.position - self.center) <= 0.5 * self.lengths,
            axis=1,
        )

        return particles[np.arange(len(particles))[in_box_mask]]


# Boundaries
class Boundary(Protocol):
    """Boundary geometry."""

    def get_interfaces(self, particles: ParticleContainer) -> InterfaceContainer:
        """Get particle boundary pairs.

        Args:
            particles: Particles to check for contact

        Returns:
            particle boundary pairs
        """


@dataclass
class Plane(Boundary):
    """Infinite plane.

    Attributes:
        point: Point on the plane
        normal: Normal of the plane
    """

    point: float_np_array
    normal: float_np_array

    def __post_init__(self) -> None:
        """Normalize the normal vector."""
        self.normal = self.normal / np.linalg.norm(self.normal)

    def get_interfaces(self, particles: ParticleContainer) -> InterfaceContainer:
        """Get particle boundary pairs from the plane.

        Args:
            particles: Particles to check if in contact

        Returns:
            Particle boundary pairs
        """
        particle_indices = []
        radius = []
        position = []
        normal = []
        distance_to_wall = []

        for i, (radius_i, position_i) in enumerate(
            zip(particles.radius, particles.position)
        ):
            signed_center_to_wall_distance = (
                sphere_plane_intersection.get_center_to_plane_signed_distance(
                    position_i, self.point, self.normal
                )
            )
            interface_gap = sphere_plane_intersection.get_gap(
                np.abs(signed_center_to_wall_distance), radius_i
            )

            if interface_gap < 0:
                interface_radius, interface_normal, interface_center = (
                    sphere_plane_intersection.get_sphere_plane_intersection_data(
                        position_i,
                        radius_i,
                        signed_center_to_wall_distance,
                        self.normal,
                    )
                )
                particle_indices.append(i)
                radius.append(interface_radius)
                normal.append(interface_normal)
                position.append(interface_center)
                distance_to_wall.append(signed_center_to_wall_distance)

        return InterfaceContainer(
            position=np.array(position),
            radius=np.array(radius),
            normal=np.array(normal),
            distance=np.array(distance_to_wall),
            particle_indices=particle_indices,
        )


# pylint: enable=too-few-public-methods
