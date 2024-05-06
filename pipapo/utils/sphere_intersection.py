"""Utils related to shpere-shpere intersection."""

import numpy as np


def get_sphere_intersection_data(
    center_i, radius_i, center_j, radius_j, return_area=False
):
    """Get all the data relevant for sphere-sphere intersection.

    Args:
        center_i (np.ndarray): Center of particle i
        radius_i (np.ndarray): Radius of particle i
        center_j (np.ndarray): Center of particle j
        radius_j (np.ndarray): Radius of particle j
        return_area (bool, optional): Return interface area

    Returns:
        interface_radius (float): Interface radius
        interface_normal (np.ndarray): Interface normal
        interface_center (np.ndarray): Interface center
        interface_area (float): Interface area if return area is True
    """
    center_to_center = get_center_to_center(center_i, center_j)
    distance_between_centers = np.sqrt(np.sum(center_to_center**2))

    interface_area = get_interface_area(distance_between_centers, radius_i, radius_j)
    interface_radius, interface_normal, interface_center = (
        get_interface_position_and_size(
            center_to_center, interface_area, center_i, radius_i
        )
    )
    if return_area:
        return interface_radius, interface_normal, interface_center, interface_area

    return interface_radius, interface_normal, interface_center


def get_center_to_center(center_i, center_j):
    """Get distance vector from particle i to j.

    Args:
        center_i (np.ndarray): Center of particle i
        center_j (np.ndarray): Center of particle j

    Returns:
        np.ndarray: vector between centers
    """
    return center_j - center_i


def get_interface_area(distance_between_centers, radius_i, radius_j):
    """Compute the interface area.

    Args:
        distance_between_centers (float): Distance between centers
        radius_i (np.ndarray): Radius of particle i
        radius_j (np.ndarray): Radius of particle j

    Returns:
        float: interface area
    """
    unnormalized_interface_area = -(
        0.25
        * (
            (distance_between_centers - radius_i - radius_j)
            * (distance_between_centers + radius_i - radius_j)
            * (distance_between_centers - radius_i + radius_j)
            * (distance_between_centers + radius_i + radius_j)
        )
        / (distance_between_centers * distance_between_centers)
    )
    return np.pi * unnormalized_interface_area


def get_interface_position_and_size(
    center_to_center, interface_area, center_i, radius_i
):
    """Compute the position, normal and radius of the interface.

    Args:
        center_to_center (np.ndarray): Vector between centers
        interface_area (float): Area of the interface
        center_i (np.ndarray): Center of particle i
        radius_i (np.ndarray): Radius of particle i

    Returns:
        interface_radius (float): Radius of the interface
        interface_normal (np.ndarray): Interface normal
        interface_center (np.ndarray): Interface center
    """
    interface_radius = np.sqrt(interface_area / np.pi)
    interface_normal = center_to_center / np.sqrt(np.sum(center_to_center**2))
    interface_center = center_i + interface_normal * np.sqrt(
        radius_i**2 - interface_radius**2
    )
    return interface_radius, interface_normal, interface_center


def get_gap(distance_between_centers, radius_i, radius_j):
    """Compute gap of sphere sphere intersection.

    Args:
        distance_between_centers (float): Distance between centers
        radius_i (np.ndarray): Radius of particle i
        radius_j (np.ndarray): Radius of particle j

    Returns:
        _type_: _description_
    """
    radius_sum = radius_i + radius_j
    gap = distance_between_centers - radius_sum
    return gap
