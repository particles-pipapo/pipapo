"""Utils related to shpere-shpere intersection."""

import numpy as np

from pipapo.utils.type_hinting import float_np_array


def get_center_to_center(
    center_i: float_np_array, center_j: float_np_array
) -> float_np_array:
    """Get distance vector from particle i to j.

    Args:
        center_i: Center of particle i
        center_j: Center of particle j

    Returns:
        vector between centers
    """
    return center_j - center_i


def get_interface_area(
    distance_between_centers: float, radius_i: float, radius_j: float
) -> float:
    """Compute the interface area.

    Args:
        distance_between_centers: Distance between centers
        radius_i: Radius of particle i
        radius_j: Radius of particle j

    Returns:
        float: interface area
    """
    squared_interface_radius = -(
        0.25
        * (
            (distance_between_centers - radius_i - radius_j)
            * (distance_between_centers + radius_i - radius_j)
            * (distance_between_centers - radius_i + radius_j)
            * (distance_between_centers + radius_i + radius_j)
        )
        / (distance_between_centers * distance_between_centers)
    )
    return np.pi * squared_interface_radius


def get_interface_position_and_size(
    center_to_center: float_np_array,
    interface_area: float,
    center_i: float_np_array,
    radius_i: float_np_array,
) -> tuple[float, float_np_array, float_np_array]:
    """Compute the position, normal and radius of the interface.

    Args:
        center_to_center: Vector between centers
        interface_area: Area of the interface
        center_i: Center of particle i
        radius_i: Radius of particle i

    Returns:
        interface_radius: Radius of the interface
        interface_normal: Interface normal
        interface_center: Interface center
    """
    interface_radius = np.sqrt(interface_area / np.pi)
    interface_normal = center_to_center / np.sqrt(np.sum(center_to_center**2))
    interface_center = center_i + interface_normal * np.sqrt(
        radius_i**2 - interface_radius**2
    )
    return interface_radius, interface_normal, interface_center


def get_gap(distance_between_centers: float, radius_i: float, radius_j: float) -> float:
    """Compute gap of sphere sphere intersection.

    Args:
        distance_between_centers: Distance between centers
        radius_i: Radius of particle i
        radius_j: Radius of particle j

    Returns:
        gap
    """
    radius_sum = radius_i + radius_j
    gap = distance_between_centers - radius_sum
    return gap
