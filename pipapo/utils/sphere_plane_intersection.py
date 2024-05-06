"""Utils related to sphere plane intersection."""

import numpy as np


def get_sphere_plane_intersection_data(
    center_i, radius_i, plane_point, plane_normal, return_area=False
):
    """Get all the data relevant for sphere-plane intersection.

    Args:
        center_i (np.ndarray): Center of particle i
        radius_i (np.ndarray): Radius of particle i
        plane_point (np.ndarray): Point on the plane to define the wall
        plane_normal (np.ndarray): Plane normal POINTING INWARDS to the domain
        return_area (bool, optional): Return interface area

    Returns:
        interface_radius (float): Interface radius
        interface_normal (np.ndarray): Interface normal
        interface_center (np.ndarray): Interface center
        interface_area (float): Interface area if return area is True
    """
    center_to_plane_point = get_center_to_plane_point(center_i, plane_point)
    center_to_plane_distance = get_center_to_plane_distance(
        center_to_plane_point, plane_normal
    )

    interface_area = get_interface_area(center_to_plane_distance, radius_i)
    center_to_plane = get_center_to_plane(center_to_plane_distance, plane_normal)

    interface_radius, interface_center = get_interface_position_and_size(
        center_to_plane, interface_area, center_i
    )

    interface_normal = -plane_normal
    if return_area:
        return interface_radius, interface_normal, interface_center, interface_area

    return interface_radius, interface_normal, interface_center


def get_center_to_plane_point(center_i, plane_point):
    """Get vector from particle to plane point.

    Args:
        center_i (np.ndarray): Center of particle i
        plane_point (np.ndarray): Point on the plane to define the wall

    Returns:
        np.ndarray: Vector from center to plane
    """
    return plane_point - center_i


def get_center_to_plane(center_to_plane_distance, plane_normal):
    """Get the vector from center to plane.

    Args:
        center_to_plane_distance (float): Shortest distance from center to wall
        plane_normal (np.ndarray): Plane normal POINTING INWARDS to the domain

    Returns:
        np.ndarray: Vector from center to interface center
    """
    return -center_to_plane_distance * plane_normal


def get_center_to_plane_distance(center_to_plane_point, plane_normal):
    """Distance from center to plane.

    Args:
        center_to_plane_point (np.ndarray): Vector from center to plane point
        plane_normal (np.ndarray): Plane normal POINTING INWARDS to the domain

    Returns:
        float: Shortest distance between center and plane
    """
    return np.abs(center_to_plane_point @ plane_normal)


def get_interface_area(center_to_plane_distance, radius_i):
    """Get interface area.

    Args:
        center_to_plane_distance (float): Distance from center to plane
        radius_i (np.ndarray): Radius of particle i

    Returns:
        float: area of the interface
    """
    unnormalized_interface_area = radius_i**2 - center_to_plane_distance**2
    return np.pi * unnormalized_interface_area


def get_interface_position_and_size(center_to_plane, interface_area, center_i):
    """Compute the position, normal and radius of the interface.

    Args:
        center_to_plane (np.ndarray): Vector from center to interface center
        interface_area (float): Area of the interface
        center_i (np.ndarray): Center of particle i

    Returns:
        interface_radius (float): Radius of the interface
        interface_normal (np.ndarray): Interface normal
        interface_center (np.ndarray): Interface center
    """
    interface_radius = np.sqrt(interface_area / np.pi)
    interface_center = center_i + center_to_plane
    return interface_radius, interface_center


def get_gap(center_to_plane_distance, radius_i):
    """Get the gap between sphere and plane.

    Args:
        center_to_plane_distance (float): Distance from center to plane
        radius_i (np.ndarray): Radius of particle i

    Returns:
        float: gap
    """
    gap = -radius_i + center_to_plane_distance
    return gap
