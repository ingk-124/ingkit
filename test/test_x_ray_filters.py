from copy import copy

import numpy as np
import pytest

from ingkit.physics.X_ray import AbsorptionFilter, DoubleFilter, FilterLike, FilterSet


@pytest.fixture(scope="module")
def filters():
    aluminium = AbsorptionFilter("Al", thickness=0.03)
    aluminium_backing = AbsorptionFilter("Al", thickness=1.0)
    aluminium_set = FilterSet([aluminium, aluminium_backing])
    return aluminium, aluminium_backing, aluminium_set


def test_filter_set_implements_filter_like(filters):
    aluminium, _, aluminium_set = filters

    assert isinstance(aluminium, FilterLike)
    assert isinstance(aluminium_set, FilterLike)
    np.testing.assert_array_equal(aluminium_set.E_ph, aluminium.E_ph)


def test_filter_set_supports_unambiguous_legacy_positional_order(filters):
    _, _, aluminium_set = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])

    with pytest.deprecated_call(match="transmission.*is deprecated"):
        legacy = aluminium_set.transmission([0.03, 1.0], E_ph)

    current = aluminium_set.transmission(E_ph, [0.03, 1.0])
    np.testing.assert_allclose(legacy, current)


def test_filter_set_normalizes_numpy_scalar_thickness(filters):
    _, _, aluminium_set = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])

    scalar = aluminium_set.transmission_angle(
        E_ph=E_ph, angle=np.float64(0.2), thickness=np.float64(0.5)
    )
    explicit = aluminium_set.transmission_angle(
        E_ph=E_ph, angle=0.2, thickness=[0.5, 0.5]
    )

    np.testing.assert_allclose(scalar, explicit)


def test_absorption_filter_and_filter_set_intensity(filters):
    aluminium, _, aluminium_set = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([100.0, 300.0])
    angle = np.array([0.0, 0.2])

    single_intensity = aluminium.intensity(Te=Te, E_ph=E_ph, angle=angle)
    set_intensity = aluminium_set.intensity(Te=Te, E_ph=E_ph, angle=angle)

    assert single_intensity.shape == (2, 2)
    assert set_intensity.shape == (2, 2)


@pytest.mark.parametrize("filter_index", [0, 2])
def test_intensity_lookup_matches_exact_on_temperature_grid(filters, filter_index):
    filter_ = filters[filter_index]
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([100.0, 300.0])
    angle = np.array([0.0, 0.2])

    exact = filter_.intensity(Te=Te, E_ph=E_ph, angle=angle)
    lookup = filter_.intensity_lookup(Te=Te, E_ph=E_ph, angle=angle, Te_grid=Te)

    np.testing.assert_allclose(lookup, exact)


@pytest.mark.parametrize("filter_index", [0, 2])
def test_intensity_points_uses_pointwise_angle_shape(filters, filter_index):
    filter_ = filters[filter_index]
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([[100.0, 300.0], [500.0, 700.0]])
    angle = np.array([[0.0, 0.1], [0.2, 0.3]])

    intensity = filter_.intensity_points(Te=Te, E_ph=E_ph, angle=angle)

    assert intensity.shape == Te.shape


@pytest.mark.parametrize("filter_index", [0, 2])
def test_precomputed_intensity_response_matches_exact_on_grid(filters, filter_index):
    filter_ = filters[filter_index]
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([[100.0, 300.0], [100.0, 300.0]])
    angle = np.array([[0.0, 0.0], [0.2, 0.2]])

    exact = filter_.intensity_points(Te=Te, E_ph=E_ph, angle=angle)
    filter_.set_intensity_response(
        Te=np.array([100.0, 300.0]), E_ph=E_ph, angle=np.array([0.0, 0.2])
    )
    response = filter_.intensity_points_from_response(Te=Te, angle=angle)

    np.testing.assert_allclose(response, exact)


def test_precomputed_intensity_response_rejects_out_of_grid_values(filters):
    aluminium, _, _ = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])

    aluminium.set_intensity_response(
        Te=np.array([100.0, 300.0]), E_ph=E_ph, angle=np.array([0.0, 0.2])
    )

    with pytest.raises(ValueError, match="Te is outside"):
        aluminium.intensity_points_from_response(Te=np.array([50.0]), angle=0.1)
    with pytest.raises(ValueError, match="angle is outside"):
        aluminium.intensity_points_from_response(Te=np.array([200.0]), angle=0.3)


@pytest.mark.parametrize("filter_index", [0, 2])
def test_temperature_response_is_logarithmic_gradient(filters, filter_index):
    filter_ = filters[filter_index]
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([300.0, 600.0, 1_200.0])
    angle = np.array([0.0, 0.2])
    intensity = filter_.intensity(Te=Te, E_ph=E_ph, angle=angle)

    response = filter_.temperature_response(Te=Te, E_ph=E_ph, angle=angle)
    expected = np.gradient(
        np.log(intensity), np.log(Te), axis=0, edge_order=2
    )

    np.testing.assert_allclose(response, expected)


def test_temperature_response_requires_positive_temperature_grid(filters):
    aluminium, _, _ = filters

    with pytest.raises(ValueError, match="positive"):
        aluminium.temperature_response(Te=np.array([0.0, 100.0]))


def test_double_filter_temperature_response(filters):
    aluminium, _, aluminium_set = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([300.0, 600.0, 1_200.0])
    double_filter = DoubleFilter(aluminium_set, aluminium, E_ph=E_ph)

    response1, response2 = double_filter.temperature_response(Te=Te, E_ph=E_ph)

    np.testing.assert_allclose(
        response1, aluminium_set.temperature_response(Te=Te, E_ph=E_ph)
    )
    np.testing.assert_allclose(
        response2, aluminium.temperature_response(Te=Te, E_ph=E_ph)
    )


def test_double_filter_precomputed_pointwise_ratios(filters):
    aluminium, _, aluminium_set = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([[100.0, 300.0], [100.0, 300.0]])
    angle = np.array([[0.0, 0.0], [0.2, 0.2]])
    double_filter = DoubleFilter(aluminium_set, aluminium, E_ph=E_ph)

    double_filter.set_intensity_response(
        Te=np.array([100.0, 300.0]), E_ph=E_ph, angle=np.array([0.0, 0.2])
    )
    ratio_12, ratio_21 = double_filter.intensity_ratios_points_from_response(
        Te=Te, angle=angle
    )
    intensity1 = aluminium_set.intensity_points(Te=Te, E_ph=E_ph, angle=angle)
    intensity2 = aluminium.intensity_points(Te=Te, E_ph=E_ph, angle=angle)

    np.testing.assert_allclose(ratio_12, intensity1 / intensity2)
    np.testing.assert_allclose(ratio_21, intensity2 / intensity1)


def test_double_filter_Te_from_ratio_names_match_ratio_direction(filters):
    aluminium, _, aluminium_set = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([100.0, 300.0, 600.0])
    double_filter = DoubleFilter(aluminium_set, aluminium, E_ph=E_ph)

    double_filter.set_Te_from_ratio(Te=Te, E_ph=E_ph, angle=0.0)
    ratio_12, ratio_21 = double_filter.intensity_ratios_points_from_response(
        Te=Te, angle=0.0
    )

    np.testing.assert_allclose(double_filter.Te_from_1over2(ratio_12, angle=0.0), Te)
    np.testing.assert_allclose(double_filter.Te_from_2over1(ratio_21, angle=0.0), Te)


def test_double_filter_intensities_delegate_to_filters(filters, monkeypatch):
    aluminium, _, aluminium_set = filters
    E_ph = np.array([1_000.0, 2_000.0, 3_000.0])
    Te = np.array([100.0, 300.0])
    angle = np.array([0.0, 0.2])
    double_filter = DoubleFilter(aluminium_set, aluminium, E_ph=E_ph)
    expected1 = aluminium_set.intensity(Te=Te, E_ph=E_ph, angle=angle)
    expected2 = aluminium.intensity(Te=Te, E_ph=E_ph, angle=angle)
    calls = []

    def first_intensity(**kwargs):
        calls.append("filter1")
        return expected1

    def second_intensity(**kwargs):
        calls.append("filter2")
        return expected2

    monkeypatch.setattr(aluminium_set, "intensity", first_intensity)
    monkeypatch.setattr(aluminium, "intensity", second_intensity)

    intensity1, intensity2 = double_filter.intensities(Te=Te, E_ph=E_ph, angle=angle)

    assert calls == ["filter1", "filter2"]
    np.testing.assert_allclose(intensity1, expected1)
    np.testing.assert_allclose(intensity2, expected2)


@pytest.mark.parametrize("combination", ["single-single", "set-single", "set-set"])
def test_double_filter_accepts_filter_like_combinations(filters, combination):
    aluminium, aluminium_backing, aluminium_set = filters
    pairs = {
        "single-single": (aluminium, aluminium_backing),
        "set-single": (aluminium_set, aluminium),
        "set-set": (aluminium_set, aluminium_set),
    }
    double_filter = DoubleFilter(*pairs[combination], E_ph=np.array([1_000.0, 2_000.0]))

    transmission1, transmission2 = double_filter.transmissions(
        E_ph=np.array([1_000.0, 2_000.0]), angle=np.array([0.0, 0.2]), squeeze=False
    )

    assert transmission1.shape == (2, 2)
    assert transmission2.shape == (2, 2)


def test_filter_set_rejects_mismatched_energy_grids(filters):
    aluminium, aluminium_backing, _ = filters
    aluminium_backing = copy(aluminium_backing)
    aluminium_backing._E_ph = aluminium_backing.E_ph + 1.0

    with pytest.raises(ValueError, match="component E_ph arrays do not match"):
        _ = FilterSet([aluminium, aluminium_backing]).E_ph
