import numpy as np
import pytest
from pipapo import ParticleContainer
from pipapo.utils.dataclass import Container, NumpyContainer
from pipapo.utils.exceptions import ContainerError, LenError

from .testing_utils import assert_equal, indexer


@pytest.fixture(name="container_1")
def fixture_container_1():
    """Container fixture."""
    np.random.seed(42)
    n_elements = 10
    radius = np.arange(n_elements).tolist()
    position = np.random.rand(n_elements, 3).tolist()
    ids = np.arange(n_elements).tolist()
    np.random.shuffle(ids)
    container = Container(radius=radius, position=position, id=ids)
    return container, (ids, radius, position, n_elements)


@pytest.fixture(name="container_2")
def fixture_container_2():
    """Numpy container fixture."""
    np.random.seed(42)
    n_elements = 10
    radius = np.arange(n_elements).reshape(-1, 1)
    position = np.random.rand(n_elements, 3)
    ids = np.arange(n_elements).reshape(-1, 1)
    np.random.shuffle(ids)
    container = NumpyContainer(radius=radius, position=position, id=ids)
    return container, (ids, radius, position, n_elements)


@pytest.fixture(name="container_3")
def fixture_container_3():
    """Particle container fixture."""
    np.random.seed(42)
    n_elements = 10
    radius = np.arange(n_elements).reshape(-1, 1)
    position = np.random.rand(n_elements, 3)
    ids = np.arange(n_elements).reshape(-1, 1)
    np.random.shuffle(ids)
    container = ParticleContainer(radius=radius, position=position, id=ids)
    return container, (ids, radius, position, n_elements)


@pytest.fixture(name="containers", params=["container_1", "container_2", "container_3"])
def fixture_containers(request, container_1, container_2, container_3):
    if request.param == "container_1":
        return container_1
    elif request.param == "container_2":
        return container_2
    else:
        return container_3


# dataclass only
@pytest.mark.parametrize("method", ["mean_of_field", "standard_deviation_of_field"])
def test_method_of_field_failure_due_to_shape(container_1, method):
    """Raise exception in case dimension is not consistent."""
    container, _ = container_1
    container.radius[0] = [1, 2, 3]
    method = getattr(container, method)
    with pytest.raises(
        NotImplementedError,
        match="computation is currently only available for",
    ):
        method("radius")


@pytest.mark.parametrize(
    "field",
    [
        ("radius", True),
        ("position", True),
        ("invalid_field", False, [1, 12, [1, 2, 3]]),
    ],
)
def test_if_equal_length(container_1, field):
    """Test if lengths of field are equal."""
    container, _ = container_1
    if len(field) == 2:
        field_name, reference = field
    else:
        field_name, reference, mod_field = field
        field = setattr(container, field_name, mod_field)
    assert container._check_if_equal_length(field_name) == reference


def test_initialization_empty():
    """Test if container is initialized."""
    container = Container()
    assert container.field_names == ["id"]
    assert container._current_idx == 0


def test_initialization_fail():
    """Test if container is not initialized."""
    with pytest.raises(
        ContainerError,
        match="You provided field_names and fields. You can only provide the one or the other!",
    ):
        Container(None, None, "radius", position=[])


def test_container_initialization_id_creation():
    """Assert if ids are initialized."""
    container = Container(radius=np.ones(10).tolist())
    assert assert_equal(container.id, np.arange(10).tolist())


def test_initialization_mismatching_lists():
    """Test initialization for mismatched list lengths."""
    with pytest.raises(
        ValueError,
        match="Dimension mismatch between fields! radius: 10, position: 5",
    ):
        Container(radius=np.arange(10).tolist(), position=np.arange(5).tolist())


# other tests
def test_bool_false(containers):
    """Test bool method for False."""
    container, _ = containers
    container = type(container)()
    assert not bool(container)


def test_initialization_from_fields(containers):
    """Test initialization from fields."""
    container, (id, radius, position, _) = containers
    assert assert_equal(id, container.id)
    assert assert_equal(radius, container.radius)
    assert assert_equal(position, container.position)


def test_add_field_name(containers):
    """Test if field name is added"""
    container, _ = containers
    new_field_name = "new_field"
    container._add_field_name(new_field_name)
    assert new_field_name in container.field_names


def test_values(containers):
    """Test _values method."""
    container, (id, radius, position, _) = containers
    for obtained_valued, reference_value in zip(
        container._values(), [id, position, radius]
    ):
        assert assert_equal(obtained_valued, reference_value)


def test_items(containers):
    """Test _items method."""
    container, (id, radius, position, _) = containers
    reference_dictionary = dict(
        zip(["radius", "position", "id"], [radius, position, id])
    )
    for field_name, field_value in container._items():
        assert assert_equal(field_value, reference_dictionary[field_name])


def test_len(containers):
    """Test len method"""
    container, (id, radius, position, n_elements) = containers
    assert len(container) == n_elements == len(id) == len(radius) == len(position)


def test_len_failure(containers):
    """Test len method failure."""
    container, _ = containers
    container.id = np.arange(11)
    with pytest.raises(
        LenError,
        match="Different lengths between fields: id: 11, position: 10, radius: 10",
    ):
        len(container)


def test_bool_true(containers):
    """Test bool method for True."""
    container, _ = containers
    assert bool(container)


def test_copy(containers):
    """Test copy object."""
    container_reference, _ = containers
    container_copy = container_reference.copy()
    assert not id(container_reference) == id(container_copy)
    for field_name in set(container_reference.field_names).union(
        container_copy.field_names
    ):
        assert assert_equal(
            getattr(container_copy, field_name),
            getattr(container_reference, field_name),
        )


def test_to_dict(containers):
    """Test to dict method."""
    container, (id, radius, position, _) = containers
    reference_dictionary = dict(
        zip(["radius", "position", "id"], [radius, position, id])
    )
    for key, value in container.to_dict().items():
        assert assert_equal(reference_dictionary[key], value)


def test_add_field(containers):
    """Test add field method."""
    container, (_, field_1, _, _) = containers
    original_datatype = container.datatype

    field_3 = field_1[::1]
    container.add_field("field_3", field_3)
    new_datatype = container.datatype

    assert "field_3" in container.field_names
    assert hasattr(container, "field_3")
    assert assert_equal(container.field_3, field_3)

    # check that field is copied!
    assert id(container.field_3) != id(field_3)

    # For particle containers the type does not change
    particle_factor = isinstance(container, ParticleContainer)

    # check that datatype changed!
    assert (new_datatype != original_datatype) ^ particle_factor


def test_add_field_dimension_mismatch(containers):
    """Test add field method with different dimensions."""
    container, (_, _, _, n_elements) = containers
    field_3 = np.random.rand(n_elements + 1)
    with pytest.raises(Exception, match="Dimension mismatch while adding new field."):
        container.add_field("field_3", field_3)


def test_remove_field(containers):
    """Test remove field."""
    container, _ = containers
    field_name_to_be_removed = "new_field"

    # Add a new field to be removed
    container.add_field(field_name_to_be_removed, container.id.copy())
    original_datatype = container.datatype
    field_names_original = container.field_names.copy()

    container.remove_field(field_name_to_be_removed)
    new_datatype = container.datatype

    # Check if the field name was removed
    assert set(container.field_names).union([field_name_to_be_removed]) == set(
        field_names_original
    )

    # Check if attribute was removed
    assert not hasattr(container, field_name_to_be_removed)

    # For particle containers the type does not change
    particle_factor = isinstance(container, ParticleContainer)

    # check that datatype changed!
    assert (new_datatype != original_datatype) ^ particle_factor


def test_remove_field_warning(containers):
    """Warn if field to be removed does not exist."""
    container, _ = containers
    field_name_to_be_removed = "field_3"
    with pytest.warns():
        container.remove_field(field_name_to_be_removed)


def test_evaluate_function(containers):
    """Test evaluate function."""
    container, (_, radius, _, _) = containers

    reference_value = np.array(radius) ** 2

    def eval_function_example(x):
        return x.radius**2

    assert assert_equal(
        container.evaluate_function(eval_function_example), reference_value
    )
    assert not hasattr(container, "eval_function_example")


def test_evaluate_function_add_as_field(containers):
    """Test evaluate function and add as field."""
    container, (_, radius, _, _) = containers

    reference_value = np.array(radius) ** 2

    def eval_function_example(x):
        return x.radius**2

    container.evaluate_function(eval_function_example, add_as_field=True)
    assert assert_equal(reference_value, container.eval_function_example)
    assert hasattr(container, "eval_function_example")

    container.evaluate_function(
        eval_function_example, add_as_field=True, field_name="new_name"
    )
    assert assert_equal(reference_value, container.eval_function_example)
    assert hasattr(container, "new_name")


def test_evaluate_function_add_as_field_lambda(containers):
    """Test evaluate lambda function and add as field."""
    container, _ = containers

    eval_function_example = lambda x: x.radius**2

    with pytest.raises(
        NameError,
        match="You are using a lambda function. Please add a name to the field you want to add with the keyword 'field_name'",
    ):
        container.evaluate_function(eval_function_example, add_as_field=True)


@pytest.mark.parametrize("idx", [3, slice(2, 10), [3, 5]])
def test_getitem(containers, idx):
    """Test getitem method for different types of indexing."""
    container, (ids, radius, position, _) = containers

    obtained_item = container[idx]
    assert assert_equal(
        obtained_item.radius, indexer(radius, idx, reshape=not isinstance(idx, int))
    )
    assert assert_equal(
        obtained_item.position, indexer(position, idx, reshape=not isinstance(idx, int))
    )
    assert assert_equal(
        obtained_item.id, indexer(ids, idx, reshape=not isinstance(idx, int))
    )


def test_getitem_int_failure(containers):
    """Test index too large."""
    container, (_, _, _, n_elements) = containers
    with pytest.raises(
        IndexError, match=f"Index {n_elements+1} out of range for size {n_elements}"
    ):
        container[n_elements + 1]


def test_map_to_index(containers):
    """Test if ids are correctly mapped."""
    container, (ids, _, _, _) = containers
    for idx, i in enumerate(ids):
        assert container._map_id_to_index(i) == idx


def test_map_to_index_not_found(containers):
    """Test map_to_index for unknown id."""
    container, (ids, _, _, _) = containers
    with pytest.raises(
        IndexError, match=f"No element with id {int(max(ids) + 1)} found!"
    ):
        container._map_id_to_index(max(ids) + 1)


def test_map_to_index_duplicate(containers):
    """Test map_to_index for duplicate index."""
    container, _ = containers
    container.id[-2] = 42
    container.id[-1] = 42
    with pytest.raises(IndexError, match="More than one element found with id 42."):
        container._map_id_to_index(42)


def test_reset_ids(containers):
    """Test if ids are reset correctly."""
    container, (_, _, _, n_elements) = containers
    container.reset_ids()
    reset_ids = np.arange(n_elements)
    if isinstance(container.id, list):
        reset_ids = reset_ids.tolist()
    else:
        reset_ids = reset_ids.reshape(-1, 1)
    assert assert_equal(container.id, reset_ids)


def test_update_by_id(containers):
    """Test update by id."""
    container, _ = containers
    new_entry = {"radius": 5, "position": [1, 2, 3]}
    container.update_by_id(new_entry, 2)
    new_entry["id"] = 2

    reference = container.get_by_id(2)
    obtained = container.datatype(**new_entry)

    for key in new_entry:
        assert assert_equal(getattr(reference, key), getattr(obtained, key))


@pytest.mark.parametrize("idx", [3, [3, 5], (5, 6)])
def test_get_by_id(containers, idx):
    """Test get by id."""
    container, (ids, radius, position, _) = containers
    obtained_item = container.get_by_id(indexer(ids, idx))
    assert assert_equal(obtained_item.radius, indexer(radius, idx))
    assert assert_equal(obtained_item.position, indexer(position, idx))


def test_get_by_id_failure(containers):
    """Test get by id if non-iterable"""
    container, _ = containers
    with pytest.raises(
        TypeError,
        match="Ids need to be iterable",
    ):
        # Non iterable and non int
        container.get_by_id(5.0)


def test_list(containers):
    """Test __list__ method."""
    container, (ids, radius, position, _) = containers
    obtained = list(container)
    reference = [
        container.datatype(**{"radius": f1, "position": f2, "id": i})
        for i, f1, f2 in zip(ids, radius, position)
    ]
    assert isinstance(obtained, list)
    for o, r in zip(obtained, reference):
        for key in container.field_names:
            assert assert_equal(getattr(r, key), getattr(o, key))


def test_mean_of_field(containers):
    """Test mean of field method."""
    container, (_, radius, position, _) = containers
    assert container.mean_of_field("radius") == np.mean(radius)
    assert assert_equal(
        container.mean_of_field("position", axis=0), np.mean(position, axis=0)
    )


def test_standard_deviation_of_field(containers):
    """Test standard deviation of field method."""
    container, (_, radius, position, _) = containers
    assert container.standard_deviation_of_field("radius") == np.std(radius)
    assert assert_equal(
        container.standard_deviation_of_field("position", axis=0),
        np.std(position, axis=0),
    )


def test_histogram_of_field(containers):
    """Test histogram of field method."""
    container, (_, radius, position, _) = containers

    hist_ref, bins_ref = np.histogram(radius)
    hist_container, bins_container = container.histogram_of_field("radius")
    assert assert_equal(hist_ref, hist_container)
    assert assert_equal(bins_ref, bins_container)

    hist_ref, bins_ref = np.histogram(radius, bins=2)
    hist_container, bins_container = container.histogram_of_field("radius", bins=2)
    assert assert_equal(hist_ref, hist_container)
    assert assert_equal(bins_ref, bins_container)


def test_histogram_of_field_failure_due_to_shape(containers):
    """Raise exception in case of multidimensional array."""
    container, _ = containers
    with pytest.raises(
        NotImplementedError,
        match="Histogram is currently only available for",
    ):
        container.histogram_of_field("position")


@pytest.mark.parametrize("field", [("radius", False), ("position", True)])
def test_if_field_is_nested(containers, field):
    """Test if field is nested."""
    container, _ = containers
    field_name, reference = field
    assert container._check_if_field_is_nested(field_name) == reference


def test_add_element_container(containers):
    """Test if elements are added correctly."""
    container, (ids, radius, position, n_elements) = containers
    container.add_element(container)

    # Extend the reference solution
    if isinstance(ids, list):
        radius.extend(radius)
        position.extend(position)
        ids.extend(np.arange(mid := max(ids) + 1, mid + n_elements).tolist())
    else:
        radius = np.row_stack((radius, radius))
        position = np.row_stack((position, position))
        ids = np.row_stack(
            (ids, np.arange(mid := max(ids) + 1, mid + n_elements).reshape(-1, 1))
        )

    assert len(container) == 2 * n_elements
    assert assert_equal(container.radius, radius)
    assert assert_equal(container.position, position)
    assert assert_equal(container.id, ids)


def test_add_element_single_element(containers):
    """Test if element is added correctly."""
    container, (ids, radius, position, n_elements) = containers
    container.add_element(container[0])

    # Extend the reference solution
    if isinstance(ids, list):
        radius.append(radius[0])
        position.append(position[0])
        ids.append(max(ids) + 1)
    else:
        radius = np.row_stack((radius, radius[0]))
        position = np.row_stack((position, position[0]))
        ids = np.row_stack((ids, np.arange(max(ids) + 1, max(ids) + 2).reshape(-1, 1)))
    assert len(container) == n_elements + 1
    assert assert_equal(container.radius, radius)
    assert assert_equal(container.position, position)
    assert assert_equal(container.id, ids)
    assert container[0].id != container[-1].id
    assert assert_equal(container[0].radius, container[-1].radius)
    assert assert_equal(container[0].position, container[-1].position)


def test_remove_from_field_by_index(containers):
    """Test if item is remove by index."""
    container, (ids, _, _, _) = containers
    idx = 5
    obtained = container._remove_from_field_by_index(ids, idx)
    if isinstance(ids, list):
        reference = ids[:idx] + ids[idx + 1 :]
    else:
        reference = np.delete(ids, idx, 0)
    assert_equal(obtained, reference)


def test_remove_element_by_id(containers):
    """Test if element is removed from the container"""
    container, (ids, _, _, n_elements) = containers
    container.remove_element_by_id(ids[0])
    assert len(container) == n_elements - 1
    assert not ids[0] in container.id


def test_where(containers):
    """Test where method."""
    container, (_, radius, _, n_elements) = containers
    condition = np.array(radius) > 5
    obtained_indexes = container.where(condition, index_only=True)
    assert_equal(np.arange(6, n_elements), obtained_indexes)
