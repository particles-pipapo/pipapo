"""Utils related to sphere plane intersection."""

import numpy as np

from pipapo.utils.type_hinting import float_np_array


def get_sphere_plane_intersection_data(
    center_i: float_np_array,
    radius_i: float,
    center_to_plane_signed_distance: float,
    plane_normal: float_np_array,
) -> tuple[float, float_np_array, float_np_array]:
    """Get all the data relevant for sphere-plane intersection.

    Args:
        center_i: Center of particle i
        radius_i: Radius of particle i
        center_to_plane_signed_distance: Distance from center to plane
        plane_normal: Unit plane normal

    Returns:
        interface_radius: Interface radius
        interface_normal: Interface normal
        interface_center: Interface center
    """
    interface_normal = np.sign(center_to_plane_signed_distance) * plane_normal

    interface_radius = get_interface_radius(center_to_plane_signed_distance, radius_i)

    interface_center = (
        center_i + np.abs(center_to_plane_signed_distance) * interface_normal
    )

    return interface_radius, interface_normal, interface_center


def get_center_to_plane_point(
    center_i: float_np_array, plane_point: float_np_array
) -> float_np_array:
    """Get vector from particle to plane point.

    Args:
        center_i: Center of particle i
        plane_point: Point on the plane to define the wall

    Returns:
        Vector from center to plane
    """
    return plane_point - center_i


def get_center_to_plane(
    center_to_plane_distance: float, plane_normal: float_np_array
) -> float_np_array:
    """Get the vector from center to plane.

    Args:
        center_to_plane_distance: Shortest distance from center to wall
        plane_normal: Plane normal POINTING OUTWARDS to the domain

    Returns:
        Vector from center to interface center
    """
    return center_to_plane_distance * plane_normal


def get_center_to_plane_signed_distance(
    center_i: float_np_array, plane_point: float_np_array, plane_normal: float_np_array
) -> float:
    """Signed distance from center to plane.

    A positive sign indicates the same direction as the plane normal.

    Args:
        center_to_plane_point: Vector from center to plane point
        plane_normal: Plane normal POINTING INWARDS to the domain

    Returns:
        float: Shortest distance between center and plane
    """
    center_to_plane_point = get_center_to_plane_point(center_i, plane_point)
    return np.sum(center_to_plane_point * plane_normal)


def get_interface_radius(center_to_plane_distance: float, radius_i: float) -> float:
    """Get interface radius.

    Args:
        center_to_plane_distance: Distance from center to plane
        radius_i: Radius of particle i

    Returns:
        float: area of the interface
    """
    return np.sqrt(
        radius_i * radius_i - center_to_plane_distance * center_to_plane_distance
    )


def get_gap(signed_center_to_plane_distance: float, radius_i: float) -> float:
    """Get the gap between sphere and plane.

    Args:
        signed_center_to_plane_distance: Distance from center to plane
        radius_i: Radius of particle i

    Returns:
        float: gap
    """
    gap = -radius_i + np.abs(signed_center_to_plane_distance)
    return gap
