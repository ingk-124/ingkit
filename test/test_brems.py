import numpy as np
import pytest

from ingkit.physics.plasma import brems


def test_bremsstrahlung_spectrum_preserves_photon_energy_axis():
    E_ph = np.array([100.0, 200.0, 300.0])

    scalar_temperature = brems.bremsstrahlung_spectrum(100.0, 1e18, E_ph=E_ph)
    temperature_vector = brems.bremsstrahlung_spectrum(
        np.array([100.0, 200.0]), 1e18, E_ph=E_ph
    )

    assert scalar_temperature.shape == (3,)
    assert temperature_vector.shape == (2, 3)


def test_integrate_spectrum_accepts_integer_transmission():
    E_ph = np.array([100.0, 200.0, 300.0])
    spectrum = np.ones(3)

    result = brems.integrate_spectrum(spectrum, E_ph, transmission=[1, 1, 1])
    expected = brems.integrate_spectrum(spectrum, E_ph)

    np.testing.assert_allclose(result, expected)


def test_integrate_spectrum_validates_last_axis():
    with pytest.raises(ValueError, match="last-axis"):
        brems.integrate_spectrum(np.ones(2), np.ones(3))
