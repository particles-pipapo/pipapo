import numpy as np
import pytest
from pipapo.utils.dataclass import Container

from .testing_utils import assert_equal


@pytest.fixture(name="container_1")
def fixture_containter_1():
    """Container fixture."""
    np.random.seed(42)
    n_elements = 10
    field_1 = np.arange(n_elements).tolist()
    field_2 = np.random.rand(n_elements, 3).tolist()
    ids = np.arange(n_elements).tolist()
    np.random.shuffle(ids)
    container = Container([], field_1=field_1, field_2=field_2, id=ids)
    return container, (ids, field_1, field_2, n_elements)


def test_intialization_empty():
    """Test if container is initialized."""
    container = Container([])
    assert container.field_names == ["id"]
    assert container._current_idx == 0


def test_initialization_from_fields(container_1):
    """Test initialization from fields."""
    container, (id, field_1, field_2, _) = container_1
    assert assert_equal(id, container.id)
    assert assert_equal(field_1, container.field_1)
    assert assert_equal(field_2, container.field_2)


def test_intialization_fail():
    """Test if container is not initialized."""
    with pytest.raises(
        Exception,
        match="You provided field_names and fields. You can only provide the one or the other!",
    ):
        Container([], "field_1", field_2=[])


def test_container_intialization_id_creation():
    """Assert if ids are initialized."""
    container = Container([], field_1=np.ones(10).tolist())
    assert assert_equal(container.id, np.arange(10).tolist())


def test_initialization_mismatching_lists():
    """Test initialization for mismatched list lengths."""
    with pytest.raises(
        ValueError,
        match="Dimension mismatch between fields! field_1: 10, field_2: 5",
    ):
        Container([], field_1=np.arange(10).tolist(), field_2=np.arange(5).tolist())


def test_add_field_name():
    """Test if field name is added"""
    container = Container([])
    new_field_name = "new_field"
    container._add_field_name(new_field_name)
    assert new_field_name in container.field_names


def test_values(container_1):
    """Test _values method."""
    container, (id, field_1, field_2, _) = container_1
    for obtained_valued, reference_value in zip(
        container._values(), [field_1, field_2, id]
    ):
        assert obtained_valued == reference_value


def test_items(container_1):
    """Test _items method."""
    container, (id, field_1, field_2, _) = container_1
    reference_dictionary = dict(
        zip(["field_1", "field_2", "id"], [field_1, field_2, id])
    )
    for field_name, field_value in container._items():
        assert field_value == reference_dictionary[field_name]


def test_len(container_1):
    """Test len method"""
    container, (id, field_1, field_2, n_elements) = container_1
    assert len(container) == n_elements == len(id) == len(field_1) == len(field_2)


def test_len_failure(container_1):
    """Test len method failure."""
    container, (_, _, _, n_elements) = container_1
    container.id = np.arange(n_elements + 1).tolist()
    with pytest.raises(
        ValueError,
        match="Different lengths between fields: field_1: 10, field_2: 10, id: 11",
    ):
        len(container)


def test_bool_false():
    """Test bool method for False."""
    container = Container([])
    assert not bool(container)


def test_bool_true(container_1):
    """Test bool method for True."""
    container, _ = container_1
    assert bool(container)


def test_copy(container_1):
    """Test copy object."""
    container_reference, _ = container_1
    container_copy = container_reference.copy()
    assert not id(container_reference) == id(container_copy)
    for field_name in set(container_reference.field_names).union(
        container_copy.field_names
    ):
        assert getattr(container_copy, field_name) == getattr(
            container_reference, field_name
        )


def test_to_dict(container_1):
    """Test to dict method."""
    container, (id, field_1, field_2, _) = container_1
    reference_dictionary = dict(
        zip(["field_1", "field_2", "id"], [field_1, field_2, id])
    )
    assert container.to_dict() == reference_dictionary


def test_add_field(container_1):
    """Test add field method."""
    container, (_, _, _, n_elements) = container_1
    field_3 = np.random.rand(n_elements).tolist()
    container.add_field("field_3", field_3)
    assert "field_3" in container.field_names
    assert hasattr(container, "field_3")
    assert assert_equal(container.field_3, field_3)
    # check that field is copied!
    assert not id(container.field_3) == id(field_3)


def test_add_field_dimension_mismatch(container_1):
    """Test add field method with different dimensions."""
    container, (_, _, _, n_elements) = container_1
    field_3 = np.random.rand(n_elements + 1).tolist()
    with pytest.raises(Exception, match="Dimension mismatch while adding new field."):
        container.add_field("field_3", field_3)


def test_remove_field(container_1):
    """Test remove field."""
    container, _ = container_1
    field_names_original = container.field_names.copy()
    field_name_to_be_removed = "field_2"
    container.remove_field(field_name_to_be_removed)

    # Check if the field name was removed
    assert set(container.field_names).union([field_name_to_be_removed]) == set(
        field_names_original
    )

    # Check if attribute was removed
    assert not hasattr(container, field_name_to_be_removed)


def test_remove_field_warning(container_1):
    """Warn if field to be removed does not exist."""
    container, _ = container_1
    field_name_to_be_removed = "field_3"
    with pytest.warns():
        container.remove_field(field_name_to_be_removed)


def test_evaluate_function(container_1):
    """Test evaluate function."""
    container, (_, field_1, _, _) = container_1

    reference_value = (np.array(field_1) ** 2).tolist()

    def eval_function_example(x):
        return x["field_1"] ** 2

    assert container.evaluate_function(eval_function_example) == reference_value
    assert not hasattr(container, "eval_function_example")


def test_evaluate_function_add_as_field(container_1):
    """Test evaluate function and add as field."""
    container, (_, field_1, _, _) = container_1

    reference_value = (np.array(field_1) ** 2).tolist()

    def eval_function_example(x):
        return x["field_1"] ** 2

    container.evaluate_function(eval_function_example, add_as_field=True)
    assert container.eval_function_example == reference_value
    assert hasattr(container, "eval_function_example")

    container.evaluate_function(
        eval_function_example, add_as_field=True, field_name="new_name"
    )
    assert container.eval_function_example == reference_value
    assert hasattr(container, "new_name")


def test_evaluate_function_add_as_field_lambda(container_1):
    """Test evaluate lambda function and add as field."""
    container, _ = container_1

    eval_function_example = lambda x: x["field_1"] ** 2

    with pytest.raises(
        NameError,
        match="You are using a lambda function. Please add a name to the field you want to add with the keyword 'field_name'",
    ):
        container.evaluate_function(eval_function_example, add_as_field=True)
