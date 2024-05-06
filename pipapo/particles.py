"""Particle classes."""

import warnings
from pathlib import Path

import numpy as np
import pyvista as pv

from pipapo.particle import Particle
from pipapo.utils import sphere_plane_intersection
from pipapo.utils.binning import get_bounding_box
from pipapo.utils.contacts import (ParticlePairContainer,
                                   ParticleWallPairContainer)
from pipapo.utils.csv import import_csv
from pipapo.utils.dataclass import NumpyContainer
from pipapo.utils.io import export
from pipapo.utils.voxels import VoxelContainer
from pipapo.utils.vtk import import_vtk

pv.set_plot_theme("document")


class ParticleContainer(NumpyContainer):
    """Particles container."""

    # Id is handled by the dataclass
    _MANDATORY_FIELDS = ["position", "radius"]

    def __init__(self, *field_names, **fields):
        """Initialise particles container.

        Args:
            field_names (list): Field names list
            fields (dict): Dictionary with field names and values.
        """
        self.position = None
        self.radius = None
        if fields:
            if not set(ParticleContainer._MANDATORY_FIELDS).intersection(
                list(fields.keys())
            ) == set(ParticleContainer._MANDATORY_FIELDS):
                raise TypeError(
                    f"The mandatory fields {', '.join(ParticleContainer._MANDATORY_FIELDS)} were "
                    "not provided."
                )
        else:
            field_names = list(
                set(field_names).union(ParticleContainer._MANDATORY_FIELDS)
            )
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
            export(self.to_dict(), file_path)
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
        """Add particle(s).

        Args:
            new_particles (Particle, ParticleContainer): New data to be added
        """
        super().add_element(new_particles)

    def remove_field(self, field_name):
        """Remove field from the container.

        Args:
            field_name (str): Name of the field to be removed
        """
        fields = ParticleContainer._MANDATORY_FIELDS + ["id"]
        if field_name in fields:
            raise ValueError(
                f"Mandatory fields: {', '.join(fields)} can not be deleted!"
            )
        super().remove_field(field_name)

    def get_particle_pairs(self):
        """Get particle pairs.

        Returns:
            ParticlePairContainer: Pairs from particle set
        """
        return ParticlePairContainer.from_particles(self)

    def get_particle_wall_pairs(self, wall_point, wall_normal):
        """Get particle wall pairs.

        Args:
            wall_point  (np.ndarray): Point on the wall to define the plane
            wall_normal (np.ndarray): Wall normal POINTING INWARDS

        Returns:
            ParticleWallPairContainer: Pairs between wall and particle
        """
        return ParticleWallPairContainer.from_particles_and_wall(
            self, wall_point, wall_normal
        )

    def _update_data_type(self):
        """Needs to be overwritten."""

    def get_pv_spheres(self, field_name=None):
        """Get pyvista spheres.

        Returns:
            list: List of all the pyvista spheres
        """
        spheres = []
        for i in range(len(self)):
            sphere = pv.Sphere(
                self.radius[i], self.position[i]
            )  # pylint: disable=E1136
            if field_name:
                sphere["Data"] = (
                    np.ones(len(sphere.points)) * getattr(self, field_name)[i]
                )
            spheres.append(sphere)
        return spheres

    def plot(self, pv_plotter=None, show=True, field_name=None, **kwargs):
        """Plot spheres.

        Args:
            pv_plotter (pv.Plotter, optional): Plotter object to plot. Defaults to None.
            show (bool, optional): Open the plotting window. Defaults to True.
            field_name (str,optional): Field to plot. Defaults to None
            kwargs (dict): additional keyword arguments for add_mesh

        Returns:
            pv.Plotter: Plotter object
        """
        if not pv_plotter:
            pv_plotter = pv.Plotter()

        if not "color" in kwargs and not field_name:
            kwargs["color"] = "green"

        for sphere in self.get_pv_spheres(field_name):
            pv_plotter.add_mesh(sphere, scalar_bar_args={"title": field_name}, **kwargs)

        if show:
            pv_plotter.show()

        return pv_plotter

    def get_porosity(
        self,
        center=None,
        lengths=None,
        voxel_size=None,
        n_voxels_dim=None,
        n_threads=1,
        return_voxels=False,
    ):
        """Compute porosity.

        Args:
            center (np.ndarray): center of outer domain. Defaults to the center of the bounding box
                                 of the particle container
            lengths (np.ndarray): lengths of outer domain. Defaults to the lengths of the bounding
                                  box of the particle container
            voxel_size (float): voxel size. defaults to a quarter of the smallest radius of the
                                particles
            n_voxels_dim (np.ndarray): voxels per dimension. Defaults to the number of voxels that
                                       fit in a length of the bounding box of the container
            n_threads (int): threads to parallelize the calculation of the voxels
            return_voxels (bool): Return voxels container. Defaults to False.

        Returns:
            porosity or porosity and voxel container
        """
        voxels = VoxelContainer.from_particles(
            self,
            center=center,
            lengths=lengths,
            voxel_size=voxel_size,
            n_voxels_dim=n_voxels_dim,
            n_threads=n_threads,
        )
        porosity = 1 - voxels.volume_of_voxels() / voxels.volume_of_outer_domain()
        if return_voxels:
            return porosity, voxels

        return porosity

    def get_wall_contacts(self, wall_point, wall_normal, index_only=True):
        """Get the interface area to a wall.

        Args:
            wall_point  (np.ndarray): Point on the wall to define the plane
            wall_normal (np.ndarray): Wall normal POINTING INWARDS
            index_only (bool): Return only indices

        Returns:
            indices (np.array): List of indices (if index_only=True)
            particles (ParticleContainer): Particles in contact with the wall
        """
        center_to_wall_point = sphere_plane_intersection.get_center_to_plane_point(
            self.position, wall_point
        )
        center_to_wall_distance = (
            sphere_plane_intersection.get_center_to_plane_distance(
                center_to_wall_point, wall_normal
            )
        )
        indices = np.argwhere(
            np.abs(center_to_wall_distance) < self.radius.flatten()
        ).flatten()

        if index_only:
            return indices

        return self[indices]
