"""Tests for conatiner."""

from typing import TypedDict

import numpy as np
import pytest
from pipapo_testing_utils.asserting_utils import assert_equal, indexer

from pipapo.utils.container import Container
from pipapo.utils.type_hinting import float_np_array


class PropertiesDict(TypedDict):
    """Example properties dict."""

    np_array_1d: float_np_array
    np_array_2d: float_np_array
    list_1d_array: list[int]
    list_2d_array: list[list[int]]


class DummyContainer(Container[PropertiesDict, PropertiesDict]):
    """Dummy container."""


# pylint: disable=protected-access


@pytest.fixture(name="reference_data")
def fixture_reference_data():
    """Container classes fixture."""
    np_array_1d = np.arange(10)
    np_array_2d = np.arange(30).reshape(10, 3)
    list_1d_array = np_array_1d.tolist()
    list_2d_array = np_array_2d.tolist()
    return {
        "num_array_1d": np_array_1d.copy(),
        "num_array_2d": np_array_2d.copy(),
        "list_1d_array": list_1d_array.copy(),
        "list_2d_array": list_2d_array.copy(),
    }


@pytest.fixture(name="container")
def fixture_container(reference_data):
    """Container instance fixture."""
    container = DummyContainer(
        reference_data,
        {k: v[::-1] for k, v in reference_data.items()},
    )
    return container


def test_data_dicts(container, reference_data):
    """Test if data dicts are correctly assigned."""
    for key in reference_data:
        assert_equal(container._properties[key], reference_data[key])
        assert_equal(container.data[key], reference_data[key][::-1])


def test_len(container):
    """Test len method."""
    assert len(container) == 10


def test_add_data_field(container, reference_data):
    """Test add field method."""
    field_name = "new_field"
    field_data = np.arange(10) * 2
    container.data[field_name] = field_data
    assert set(container.data.keys()) == set(reference_data.keys()).union({field_name})
    assert_equal(container.data[field_name], field_data)


def test_remove_data_field(container):
    """Test remove field."""
    container = container.deepcopy()
    container.data.pop("num_array_1d")
    assert "num_array_1d" not in container.data.keys()


def test_getitem_single(container, reference_data):
    """Test getitem method for single index."""
    obtained_item = container[3]
    for key in reference_data:
        assert_equal(obtained_item._properties[key], indexer(reference_data[key], 3))
        assert_equal(obtained_item.data[key], indexer(reference_data[key][::-1], 3))


def test_getitem_slice(container, reference_data):
    """Test getitem method for slice index."""
    obtained_item = container[2:8]
    for key in reference_data:
        assert_equal(obtained_item._properties[key], reference_data[key][2:8])
        assert_equal(obtained_item.data[key], reference_data[key][::-1][2:8])


def test_getitem_list(container, reference_data):
    """Test getitem method for list index."""
    idx = [3, 5, 7]
    obtained_item = container[idx]
    for key in reference_data:
        assert_equal(obtained_item._properties[key], indexer(reference_data[key], idx))
        assert_equal(
            obtained_item.data[key],
            indexer(reference_data[key][::-1], idx),
        )


@pytest.mark.parametrize("index", [10, -11, "not valid"])
def test_getitem_failure(container, index):
    """Test index failure."""
    with pytest.raises(IndexError):
        container[index]  # pylint: disable=pointless-statement


def test_append_containers(container):
    """Test appending two containers."""
    container_2 = container.deepcopy()
    container.append(container_2)
    assert len(container) == 20
    for key in container._properties.keys():
        assert_equal(
            container._properties[key],
            np.concatenate(
                (container_2._properties[key], container_2._properties[key]),
                axis=0,
            ),
        )
        assert_equal(
            container._field_data[key],
            np.concatenate(
                (container_2._field_data[key], container_2._field_data[key]),
                axis=0,
            ),
        )


def test_remove_container_item(container, reference_data):
    """Test removing an item from the container."""
    container = container.deepcopy()
    container.remove(4)
    assert len(container) == 9
    for key in reference_data:
        expected_property = np.delete(reference_data[key], 4, axis=0)
        expected_data = np.delete(reference_data[key][::-1], 4, axis=0)
        assert_equal(container._properties[key], expected_property)
        assert_equal(container._field_data[key], expected_data)


def test_new_instance(container):
    """Test new instance."""
    new_instance: Container = container.create_new_instance(
        container._properties, container._field_data, container._additional_properties()
    )

    assert new_instance._properties == container._properties
    assert new_instance._field_data == container._field_data
    assert new_instance._additional_properties() == container._additional_properties()


def test_failure_wrong_size(container):
    """Test failure."""
    container.data["some_data"] = np.ones(42)
    with pytest.raises(ValueError, match="Data and Properties"):
        container.data  # pylint: disable=pointless-statement


def test_wrong_size_container(container):
    """Test, no failure."""
    Container._safe = False
    container.data["some_data"] = np.ones(42)
    assert container.data
    Container._safe = True


def test_wrong_size_container_instance(container):
    """Test no failure."""
    container._safe = False
    container.data["some_data"] = np.ones(42)
    assert container.data


# pylint: enable=protected-access
