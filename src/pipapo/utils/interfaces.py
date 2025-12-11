"""Classes related to contacts and  contact pairs."""

from __future__ import annotations

import pathlib
import warnings
from itertools import chain
from typing import Any, Generator, Generic, Self, TypedDict

import numpy as np
import pyvista as pv
from pyvista.core.utilities import translate

from pipapo.utils.container import Container, D
from pipapo.utils.io import export
from pipapo.utils.type_hinting import float_np_array


class InterfaceProperties(TypedDict):
    """Interface properties."""

    position: float_np_array
    radius: float_np_array
    normal: float_np_array
    distance: float_np_array
    particle_indices: list[tuple[int, int]] | list[int]


class InterfaceContainer(Generic[D], Container[InterfaceProperties, D]):
    """Container for contact data per particle pair."""

    def __init__(
        self,
        position: float_np_array,
        radius: float_np_array,
        normal: float_np_array,
        distance: float_np_array,
        particle_indices: list[tuple[int, int]] | list[int],
        data: D | None = None,
    ):
        """Interface container.

        Args:
            position: Position of the interface
            radius: Radius of the interface
            normal: Normal of the interface
            distance: Distance to the interface or particle centers
            particle_indices: Particle indices
            data: Inteface data
        """
        properties = InterfaceProperties(
            position=position,
            radius=radius,
            normal=normal,
            distance=distance,
            particle_indices=particle_indices,
        )

        super().__init__(properties, data)

    @property
    def particle_indices(self) -> list[tuple[int, int]] | list[int]:
        """Particle indices."""
        self._check_lens()
        return self._properties["particle_indices"]

    @particle_indices.setter
    def particle_indices(self, new_indices: list[tuple[int, int]] | list[int]) -> None:
        """Particle indices."""
        self._properties["particle_indices"] = new_indices
        self._check_lens()

    @property
    def radius(self) -> float_np_array:
        """Interface radius."""
        self._check_lens()
        return self._properties["radius"]

    @radius.setter
    def radius(self, new_radius: float_np_array) -> None:
        """Interface radius."""
        self._properties["radius"] = new_radius
        self._check_lens()

    @property
    def position(self) -> float_np_array:
        """Interface position."""
        self._check_lens()
        return self._properties["position"]

    @position.setter
    def position(self, new_position: float_np_array) -> None:
        """Interface position."""
        self._properties["position"] = new_position
        self._check_lens()

    @property
    def normal(self) -> float_np_array:
        """Interface normal."""
        self._check_lens()
        return self._properties["normal"]

    @normal.setter
    def normal(self, new_normal: float_np_array) -> None:
        """Interface normal."""
        self._properties["normal"] = new_normal
        self._check_lens()

    @property
    def distance(self) -> float_np_array:
        """Distance to interface."""
        self._check_lens()
        return self._properties["distance"]

    @distance.setter
    def distance(self, new_distance: float_np_array) -> None:
        """Distance to interface."""
        self._properties["distance"] = new_distance
        self._check_lens()

    def get_contacts(self, n_particles: int) -> dict[int, set[int]]:
        """Get contacts of the interface container.

        Only works in particle index is two dimensional.

        Args:
            n_particles: Number of total particles

        Returns:
            Contact partners per particle
        """
        if not isinstance(self.particle_indices[0], tuple):
            raise ValueError(
                "Contacts can only be derived for particle-particle contacts."
            )

        contacts: dict[int, set] = {
            particle_index: set() for particle_index in range(n_particles)
        }

        for i, j in self.particle_indices:  # type: ignore[misc]
            if j <= i:
                continue
            contacts[i].update({j})
            contacts[j].update({i})

        return contacts

    def get_isolated_particle_indices(self, n_particles: int) -> list[int]:
        """Get isolated particles.

        Args:
            n_particles: Number of total particles

        Returns:
            Isolated particles
        """
        if not isinstance(self.particle_indices[0], tuple):
            raise ValueError(
                "Isolated particles can only be derived for particle-particle contacts."
            )
        particles = set(range(n_particles))
        particles -= set(i for pair in self.particle_indices for i in pair)  # type: ignore
        return list(particles)

    def get_connected_clusters(self, n_particles: int) -> list[list[int]]:
        """Get the connected clusters.

        I.e. particle sets which a in contact with each other.

        This solution is inspired from
        https://stackoverflow.com/a/13837045

        Args:
            n_particles: Number of total particles

        Returns:
            Connected clusters
        """

        def connected_components(
            neighbors: dict[int, set],
        ) -> Generator[Generator[int]]:
            seen = set()

            def component(node: int) -> Generator[int]:
                nodes: set[int] = set([node])
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    nodes |= neighbors[node] - seen
                    yield node

            for node in neighbors:
                if node not in seen:
                    yield component(node=node)

        return [list(c) for c in connected_components(self.get_contacts(n_particles))]

    def export(self, file_path: pathlib.Path | str) -> None:
        """Export interfaces.

        Args:
            file_path: to be exported
        """
        file_path = pathlib.Path(file_path)
        if file_path.suffix == ".vtu":
            self._export_vtu(file_path)
        else:
            export(
                self.to_dict(),
                file_path,
            )

    def _export_vtu(self, file_path: pathlib.Path | str) -> None:
        """Create polyhedrons and export as vtu.

        Note: slow

        Args:
            file_path: to file.
        """
        if circles := self.get_pv_circles():
            multiblock = pv.MultiBlock(circles)
            multiblock.combine().save(file_path)
        else:
            warnings.warn("Nothing to export.")

    def get_pv_circles(self) -> list[pv.PolyData]:
        """Get pyvista circles.

        Returns:
            list: List of all the pyvista circles.
        """
        circles = []
        for i in range(len(self)):
            circle = pv.Circle(radius=self.radius[i])

            # Modify the circle in order to orient it by its normal vector
            # From https://github.com/pyvista/pyvista/discussions/4187#discussioncomment-5553953
            circle.rotate_x(90, inplace=True)
            circle.rotate_z(90, inplace=True)
            translate(circle, center=self.position[i], direction=self.normal[i])

            for field_name, field in chain(self.data.items(), self._properties.items()):
                if field_name == "position":
                    continue
                circle.cell_data[field_name] = np.array([field[i]])  # type: ignore[index]
            circles.append(circle)
        return circles

    def plot(
        self,
        pv_plotter: pv.Plotter | None = None,
        show: bool = True,
        field_name: str | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> pv.Plotter:
        """Plot interfaces.

        Args:
            pv_plotter:Plotter object to plot. Defaults to None.
            show:Open the plotting window. Defaults to True.
            field_name:Field to plot. Defaults to None
            force:Force plotting also for large sets. Defaults to None
            kwargs :additional keyword arguments for add_mesh

        Returns:
            pv.Plotter: Plotter object
        """
        if not force:
            if len(self) > 2000:
                raise ValueError(
                    "You are trying to plot a large voxel set. If you really want to do this add"
                    " the kwarg force=True."
                )

        if not pv_plotter:
            pv_plotter = pv.Plotter()

        if "color" not in kwargs and not field_name:
            kwargs["color"] = kwargs.get("color", "purple")

        kwargs["show_edges"] = kwargs.get("show_edges", True)

        for voxel in self.get_pv_circles():
            pv_plotter.add_mesh(
                voxel,
                scalar_bar_args={"title": field_name},  # type: ignore[arg-type]
                **kwargs,
            )

        if show:
            pv_plotter.show()

        return pv_plotter

    @classmethod
    def create_new_instance(
        cls, properties: InterfaceProperties, data: D, additional_properties: dict
    ) -> Self:
        """Create new interface properties instance.

        Args:
            properties: Interface properties
            data: Data
            additional_properties: Additional properties which might be constant

        Returns:
            new instance
        """
        return cls(**properties, data=data, **additional_properties)

    def _additional_properties(self) -> dict:
        """Return additional properties.

        Returns:
            Additional data
        """
        return {}
