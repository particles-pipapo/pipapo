"""Particle classes."""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, Generic, Self, TypedDict

import numpy as np
import pyvista as pv

from pipapo.utils import sphere_intersection
from pipapo.utils.container import Container, D
from pipapo.utils.csv import import_csv
from pipapo.utils.domain import Box
from pipapo.utils.interfaces import InterfaceContainer
from pipapo.utils.io import export
from pipapo.utils.type_hinting import float_np_array, int_np_array
from pipapo.utils.voxels import VoxelContainer
from pipapo.utils.vtk import import_vtk

pv.set_plot_theme("document")  # type: ignore[no-untyped-call]


class ParticleProperties(TypedDict):
    """Particle properties."""

    position: float_np_array
    radius: float_np_array


class ParticleContainer(Generic[D], Container[ParticleProperties, D]):
    """Particle container."""

    def __init__(
        self, position: float_np_array, radius: float_np_array, data: D | None = None
    ):
        """Particle container.

        Args:
            position: Positions of the particles
            radius: Radii of the particles
            data: Particle data
        """
        if data is None:
            data: D = {}  # type: ignore[no-redef]

        super().__init__(ParticleProperties(position=position, radius=radius), data)

    @classmethod
    def create_new_instance(
        cls, properties: ParticleProperties, data: D, additional_properties: dict
    ) -> Self:
        """Create new particle instance.

        Args:
            properties: Particle properties
            data: Data
            additional_properties: Additional properties which might be constant

        Returns:
            new instance
        """
        return cls(**properties, data=data, **additional_properties)

    @property
    def position(self) -> float_np_array:
        """Position."""
        self._check_lens()
        return self._properties["position"]

    @position.setter
    def position(self, new_position: float_np_array) -> None:
        """Position."""
        self._properties["position"] = new_position
        self._check_lens()

    @property
    def radius(self) -> float_np_array:
        """Radius."""
        self._check_lens()
        return self._properties["radius"]

    @radius.setter
    def radius(self, new_radius: float_np_array) -> None:
        """Radius."""
        self._properties["radius"] = new_radius
        self._check_lens()

    def get_volume_sum(self) -> float:
        """Sum of all particles volume.

        Returns:
            float: Volume
        """
        return 4 / 3 * np.pi * np.sum(self.radius**3)

    def export(self, file_path: pathlib.Path | str) -> None:
        """Export particles.

        Data type is selected based on the ending of file_path.

        Args:
            file_path: were to store the files
        """
        if self:
            export(self.to_dict(), file_path)
        else:
            warnings.warn("Empty particles set, nothing was exported.")

    @classmethod
    def from_vtk(
        cls,
        file_path: pathlib.Path | str,
        radius_keyword: str = "radius",
        diameter_keyword: str | None = None,
    ) -> ParticleContainer:
        """Create particle container from vtk.

        Args:
            file_path: to vtk file
            radius_keyword: Radius name in the file. Defaults to "radius"
            diameter_keyword: Diameter name in the file. Defaults to None

        Returns:
            ParticleContainer: particles container based on the vtk file.
        """
        dictionary = import_vtk(file_path)
        if not diameter_keyword:
            if radius_keyword in dictionary:
                dictionary["radius"] = dictionary[radius_keyword]
            else:
                raise KeyError(
                    f"Field '{radius_keyword}' not found in {pathlib.Path(file_path).resolve()}!"
                    f" Available fields are {', '.join(list(dictionary.keys()))}"
                )
        else:
            if diameter_keyword in dictionary:
                dictionary["radius"] = dictionary[diameter_keyword] / 2
            else:
                raise KeyError(
                    f"Field '{diameter_keyword}' not found in {pathlib.Path(file_path).resolve()}! "
                    f"Available fields are {', '.join(list(dictionary.keys()))}"
                )

        position = dictionary.pop("position")
        radius = dictionary.pop("radius")
        return cls(position, radius, dictionary)  # type: ignore[arg-type]

    @classmethod
    def from_csv(
        cls,
        file_path: pathlib.Path | str,
        radius_keyword: str = "radius",
        diameter_keyword: str | None = None,
        position_keywords: list[str] | None = None,
    ) -> ParticleContainer:
        """Create particle container from csv.

        Args:
            file_path: to csv file
            radius_keyword: Radius name in the file. Defaults to "radius"
            diameter_keyword: Diameter name in the file. Defaults to None

        Returns:
            ParticleContainer: particles container based on the csv file.
        """
        if position_keywords is None:
            position_keywords = ["x", "y", "z"]
        dictionary = import_csv(str(file_path))
        if not diameter_keyword:
            if radius_keyword in dictionary:
                dictionary["radius"] = dictionary[radius_keyword]
            else:
                raise KeyError(
                    f"Field '{radius_keyword}' not found in {pathlib.Path(file_path).resolve()}! "
                    f"Available fields are {', '.join(list(dictionary.keys()))}"
                )
        else:
            if diameter_keyword in dictionary:
                dictionary["radius"] = dictionary[diameter_keyword] / 2
            else:
                raise KeyError(
                    f"Field '{diameter_keyword}' not found in {pathlib.Path(file_path).resolve()}! "
                    f"Available fields are {', '.join(list(dictionary.keys()))}"
                )
        position = np.column_stack([dictionary.pop(k) for k in position_keywords])
        radius = dictionary.pop("radius")
        return cls(position, radius, data=dictionary)  # type: ignore[arg-type]

    def get_interfaces(self) -> InterfaceContainer:
        """Get interfaces between particle pairs.

        Returns:
            Particle interfaces
        """
        particle_indices = []
        radius = []
        position = []
        normal = []
        distance_between_centers = []

        for i, (radius_i, position_i) in enumerate(zip(self.radius, self.position)):
            for j, (radius_j, position_j) in enumerate(zip(self.radius, self.position)):
                if j <= i:
                    continue
                center_to_center = sphere_intersection.get_center_to_center(
                    position_i, position_j
                )
                distance_between_centers_pair = np.sqrt(np.sum(center_to_center**2))
                interface_gap = sphere_intersection.get_gap(
                    distance_between_centers_pair, radius_i, radius_j
                )

                if interface_gap < 0:
                    interface_area = sphere_intersection.get_interface_area(
                        distance_between_centers_pair, radius_i, radius_j
                    )
                    interface_radius, interface_normal, interface_center = (
                        sphere_intersection.get_interface_position_and_size(
                            center_to_center, interface_area, position_i, radius_i
                        )
                    )
                    particle_indices.append((i, j))
                    radius.append(interface_radius)
                    normal.append(interface_normal)
                    position.append(interface_center)
                    distance_between_centers.append(distance_between_centers_pair)

        return InterfaceContainer(
            position=np.array(position),
            radius=np.array(radius),
            normal=np.array(normal),
            distance=np.array(distance_between_centers),
            particle_indices=particle_indices,
        )

    def get_voxels(
        self,
        box: Box | None = None,
        voxel_size: float | None = None,
        n_voxels_dim: int_np_array | None = None,
    ) -> VoxelContainer:
        """Get voxelized particles.

        Args:
            box: Box to compute the porosity.
            voxel_size: Voxel size
            n_voxels_dim: Voxels per dim

        Returns:
            Voxel container
        """
        if box is None:
            box = self.get_bounding_box(outer_bounding_box=True)
        voxels = VoxelContainer.from_particles(
            self,
            box=box,
            voxel_size=voxel_size,
            n_voxels_dim=n_voxels_dim,
        )
        return voxels

    def get_porosity(
        self,
        box: Box | None = None,
        voxel_size: float | None = None,
        n_voxels_dim: int_np_array | None = None,
    ) -> float:
        """Get porosity of the particle system.

        Args:
            box: Box to compute the porosity.
            voxel_size: Voxel size
            n_voxels_dim: Voxels per dim

        Returns:
            The porosity
        """
        return self.get_voxels(box, voxel_size, n_voxels_dim).get_porosity()

    def get_bounding_box(self, outer_bounding_box: bool = True) -> Box:
        """Get bounding box from particles.

        Args:
            outer_bounding_box: Use the entire span of the particles

        Returns:
            center and lengths of bounding box
        """
        if outer_bounding_box:
            offset = self.radius.reshape(-1, 1)
            mins = np.min(self.position - offset, axis=0)
            maxs = np.max(self.position + offset, axis=0)
        else:
            mins = np.min(self.position, axis=0)
            maxs = np.max(self.position, axis=0)
        center = 0.5 * (mins + maxs)
        lengths = maxs - mins
        return Box(center=center, lengths=lengths)

    def plot(
        self,
        pv_plotter: pv.Plotter | None = None,
        show: bool = True,
        field_name: str | None = None,
        **kwargs: Any,
    ) -> pv.Plotter:
        """Plot spheres.

        Args:
            pv_plotter:Plotter object to plot. Defaults to None.
            show:Open the plotting window. Defaults to True.
            field_name:Field to plot. Defaults to None
            kwargs :additional keyword arguments for add_mesh

        Returns:
            pv.Plotter: Plotter object
        """
        if not pv_plotter:
            pv_plotter = pv.Plotter()  # type: ignore[no-untyped-call]

        particles = pv.PolyData(self.position)
        particles.point_data["diameter"] = 2 * self.radius

        if "color" not in kwargs and not field_name:
            kwargs["color"] = "green"
        else:
            particles.point_data[field_name] = self._field_data[field_name]  # type: ignore[index]

        sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)

        particles_glyph = particles.glyph(scale="diameter", geom=sphere)

        pv_plotter.add_mesh(particles_glyph, scalars=field_name)

        if show:
            pv_plotter.show()  # type: ignore[no-untyped-call]

        return pv_plotter
