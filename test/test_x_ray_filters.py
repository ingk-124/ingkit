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
