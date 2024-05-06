"""Interface container.

This container is mainly used for plotting or exporting.

Use the PairContainers instead.
"""

import warnings

import numpy as np
import pyvista as pv
from pyvista.utilities.geometric_objects import translate

from pipapo.utils.dataclass import NumpyContainer
from pipapo.utils.io import export, pathify

pv.set_plot_theme("document")


class InterfaceContainer(NumpyContainer):
    """Interface container.

    In pipapo slang, an interface is described by a disk using a radius, its position and its not
    vector.
    """

    def __init__(self, *field_names, **fields):
        """Initialise interface container."""
        self.position = []
        self.radius = []
        self.normal = []
        super().__init__(*field_names, **fields, datatype=None)

    def __str__(self):
        """Interface container description."""
        string = "\npipapo interface set\n"
        string += f"  with {len(self)} interfaces\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def export(self, file_path):
        """Export interfaces.

        Args:
            file_path (pathlib.Path): Path to be exported
        """
        file_path = pathify(file_path)
        if file_path.suffix == ".vtu":
            self._export_vtu(file_path)
        else:
            export(
                self.to_dict(),
                file_path,
            )

    def _export_vtu(self, file_path):
        """Create polyhedrons and export as vtu.

        Note: slow

        Args:
            file_path (pathlib.Path): Path to file.
        """
        if circles := self.get_pv_circles():
            multiblock = pv.MultiBlock(circles)
            for k, v in self._items():
                for i, circle in enumerate(circles):
                    circle.cell_data[k] = [v[i]]
            multiblock.combine().save(file_path)
        else:
            warnings.warn("Nothing to export.")

    def get_pv_circles(self):
        """Get pyvista circles.

        Returns:
            list: List of all the pyvista circles.
        """
        circles = []
        for i in range(len(self)):
            circle = pv.Circle(radius=self.radius[i])
            circle.rotate_x(90, inplace=True)
            circle.rotate_z(90, inplace=True)
            translate(circle, center=self.position[i], direction=self.normal[i])

            for field_name in self.field_names:
                circle.cell_data[field_name] = [getattr(self, field_name)[i]]

            circles.append(circle)
        return circles

    def plot(self, pv_plotter=None, show=True, field_name=None, force=False, **kwargs):
        """Plot interfaces.

        Args:
            pv_plotter (pv.Plotter, optional): Plotter object to plot. Defaults to None.
            show (bool, optional): Open the plotting window. Defaults to True.
            field_name (str,optional): Field to plot. Defaults to None
            force (bool,optional): Force plotting also for large sets. Defaults to None
            kwargs (dict): additional keyword arguments for add_mesh

        Returns:
            pv.Plotter: Plotter object
        """
        if not force:
            if len(self) > 2000:
                warnings.warn(
                    "You are trying to plot a large voxel set. If you really want to do this add"
                    " the kwarg force=True."
                )
                return

        if not pv_plotter:
            pv_plotter = pv.Plotter()

        if not "color" in kwargs and not field_name:
            kwargs["color"] = kwargs.get("color", "purple")

        kwargs["show_edges"] = kwargs.get("show_edges", True)

        for voxel in self.get_pv_circles():
            pv_plotter.add_mesh(voxel, scalar_bar_args={"title": field_name}, **kwargs)

        if show:
            pv_plotter.show()

        return pv_plotter
