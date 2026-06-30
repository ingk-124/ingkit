import numpy as np
from scipy.constants import e, m_p, mu_0

from ingkit.physics.plasma.core import Alfven_speed, alfven_speed, ion_sound_speed


def test_alfven_speed_uses_ion_mass():
    B = np.array([0.5, 1.0])
    ne = 1e19
    expected = B / np.sqrt(mu_0 * ne * m_p)

    np.testing.assert_allclose(alfven_speed(B, ne), expected)
    np.testing.assert_allclose(Alfven_speed(B, ne), expected)


def test_ion_sound_speed_uses_electron_volt_temperature():
    Te = np.array([10.0, 100.0])
    expected = np.sqrt(Te * e / m_p)

    np.testing.assert_allclose(ion_sound_speed(Te), expected)
