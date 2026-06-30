# src/ingkit/physics/plasma/core.py
# Core plasma physics functions.
from __future__ import annotations

import numpy as np
from scipy.constants import e, m_e, epsilon_0, h, mu_0, m_p


def _Te_J(Te: float | np.ndarray) -> float | np.ndarray:
    """
    Convert electron temperature from eV to Joules.

    Parameters
    ----------
    Te : float or np.ndarray
        Electron temperature (eV)

    Returns
    -------
    float or np.ndarray
        Electron temperature in Joules.
    """
    return Te * e


def _Te_eV(Te_J: float | np.ndarray) -> float | np.ndarray:
    """
    Convert electron temperature from Joules to eV.

    Parameters
    ----------
    Te_J : float or np.ndarray
        Electron temperature (Joules)

    Returns
    -------
    float or np.ndarray
        Electron temperature in eV.
    """
    return Te_J / e


# Frequencies

def plasma_frequency(ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the electron plasma frequency.

    Parameters
    ----------
    ne : float or np.ndarray
        Electron density (m^-3)

    Returns
    -------
    float or np.ndarray
        The electron plasma frequency (Hz).
    """
    return np.sqrt(ne * e ** 2 / (epsilon_0 * m_e)) / (2 * np.pi)


def ion_plasma_frequency(ni: float | np.ndarray, Z: int = 1, A: int = 1) -> float | np.ndarray:
    """
    Calculate the ion plasma frequency.

    Parameters
    ----------
    ni : float or np.ndarray
        Ion density (m^-3)
    Z : int, optional
        Ion charge state (default is 1 for singly ionized)
    A : int, optional
        Ion mass number (default is 1 for proton)

    Returns
    -------
    float or np.ndarray
        The ion plasma frequency (Hz).
    """
    m_i = A * m_p
    return np.sqrt(ni * (Z * e) ** 2 / (epsilon_0 * m_i)) / (2 * np.pi)


def cyclotron_frequency(B: float | np.ndarray, q: float = e, m: float = m_e) -> float | np.ndarray:
    """
    Calculate the cyclotron frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (Tesla)
    q : float, optional
        Charge of the particle (default is electron charge e)
    m : float, optional
        Mass of the particle (default is electron mass m_e)

    Returns
    -------
    float or np.ndarray
        The cyclotron frequency (Hz).

    Notes
    -----
    The cyclotron frequency is the frequency at which charged particles spiral around magnetic field lines.
    """
    return q * B / (2 * np.pi * m)


def electron_cyclotron_frequency(B: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the electron cyclotron frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (Tesla)

    Returns
    -------
    float or np.ndarray
        The electron cyclotron frequency (Hz).
    """
    return cyclotron_frequency(B, q=e, m=m_e)


def ion_cyclotron_frequency(B: float | np.ndarray, Z: int = 1, A: int = 1) -> float | np.ndarray:
    """
    Calculate the ion cyclotron frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (Tesla)
    Z : int, optional
        Ion charge state (default is 1 for singly ionized)
    A : int, optional
        Ion mass number (default is 1 for proton)

    Returns
    -------
    float or np.ndarray
        The ion cyclotron frequency (Hz).
    """
    m_i = A * m_p
    return cyclotron_frequency(B, q=Z * e, m=m_i)


def lower_hybrid_frequency(B: float | np.ndarray, ne: float | np.ndarray, Z: int = 1, A: int = 1
                           ) -> float | np.ndarray:
    """
    Calculate the lower hybrid frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (Tesla)
    ne : float or np.ndarray
        Electron density (m^-3)
    Z : int, optional
        Ion charge state (default is 1 for singly ionized)
    A : int, optional
        Ion mass number (default is 1 for proton)

    Returns
    -------
    float or np.ndarray
        The lower hybrid frequency (Hz).

    Notes
    -----
    The lower hybrid frequency is a characteristic frequency in magnetized plasmas, associated with the coupling of ion and electron motions.
    """
    _f_ce = electron_cyclotron_frequency(B)
    _f_ci = ion_cyclotron_frequency(B, Z, A)
    _f_pi = ion_plasma_frequency(ne, Z, A)
    omega_ce = 2 * np.pi * _f_ce
    omega_ci = 2 * np.pi * _f_ci
    omega_pi = 2 * np.pi * _f_pi
    omega = 1 / np.sqrt(omega_pi ** (-2) + 1 / (omega_ce * omega_ci))
    return omega / (2 * np.pi)


def upper_hybrid_frequency(B: float | np.ndarray, ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the upper hybrid frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (Tesla)
    ne : float or np.ndarray
        Electron density (m^-3)

    Returns
    -------
    float or np.ndarray
        The upper hybrid frequency (Hz).

    Notes
    -----
    The upper hybrid frequency is a characteristic frequency in magnetized plasmas, associated with the coupling of electron motion and plasma oscillations.
    """
    _f_ce = electron_cyclotron_frequency(B)
    _f_pe = plasma_frequency(ne)
    return np.sqrt(_f_ce ** 2 + _f_pe ** 2)


def alfven_speed(B: float | np.ndarray, ne: float | np.ndarray,
                 mi: float = m_p) -> float | np.ndarray:
    """
    Calculate the Alfvén speed.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (Tesla)
    ne : float or np.ndarray
        Electron density (m^-3)
    mi : float, optional
        Ion mass (default is proton mass m_p)

    Returns
    -------
    float or np.ndarray
        The Alfvén speed (m/s).

    Notes
    -----
    The Alfvén speed v_A is defined as:
    v_A = B / sqrt(mu_0 * n_e * m_i)
    where mu_0 is the vacuum permeability, n_e is the electron density, and m_i is the ion mass.
    It represents the speed at which Alfvén waves propagate in a magnetized plasma.
    """
    return B / np.sqrt(mu_0 * ne * mi)


def Alfven_speed(B: float | np.ndarray, ne: float | np.ndarray,
                 mi: float = m_p) -> float | np.ndarray:
    """Compatibility alias for :func:`alfven_speed`."""
    return alfven_speed(B, ne, mi=mi)


def ion_sound_speed(Te: float | np.ndarray, mi: float = m_p) -> float | np.ndarray:
    """
    Calculate the ion sound speed.

    Parameters
    ----------
    Te : float or np.ndarray
        Electron temperature (eV)
    mi : float, optional
        Ion mass (default is proton mass m_p)

    Returns
    -------
    float or np.ndarray
        The ion sound speed (m/s).

    Notes
    -----
    The ion sound speed c_s is defined as:
    c_s = sqrt(k_B * T_e / m_i)
    where k_B is the Boltzmann constant, T_e is the electron temperature in Joules, and m_i is the ion mass.
    It represents the speed at which ion acoustic waves propagate in a plasma.
    """
    Te_J = _Te_J(Te)  # Convert Te from eV to Joules
    return np.sqrt(Te_J / mi)


def o_mode_cutoff_f(ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the O-mode cutoff frequency.

    Parameters
    ----------
    ne : float or np.ndarray
        Electron density (m^-3)

    Returns
    -------
    float or np.ndarray
        The O-mode cutoff frequency (Hz).

    Notes
    -----
    The O-mode: Electric field oscillates parallel to the magnetic field.
    """
    return plasma_frequency(ne)


def x_mode_cutoff_f(ne: float | np.ndarray, B: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the X-mode cutoff frequency.

    Parameters
    ----------
    ne : float or np.ndarray
        Electron density (m^-3)
    B : float or np.ndarray
        Magnetic field strength (Tesla)

    Returns
    -------
    float or np.ndarray
        The X-mode cutoff frequency (Hz).

    Notes
    -----
    The X-mode: Electric field oscillates perpendicular to the magnetic field.
    """
    _f_ce = electron_cyclotron_frequency(B)
    _f_pe = plasma_frequency(ne)
    return 0.5 * (_f_ce + np.sqrt(_f_ce ** 2 + 4 * _f_pe ** 2))


# Dimensionless values

def coulomb_logarithm(Te: float | np.ndarray, ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the Coulomb logarithm.

    Parameters
    ----------
    Te : float or np.ndarray
        Electron temperature (eV)
    ne : float or np.ndarray
        Electron density (m^-3)

    Returns
    -------
    float or np.ndarray
        The Coulomb logarithm.

    Notes
    -----
    The Coulomb logarithm is a dimensionless quantity that arises in plasma physics, particularly in the context of collisional processes.
    """
    ne = np.maximum(np.nan_to_num(ne, nan=0), 1e10)  # Avoid zero or negative densities
    Te = np.maximum(np.nan_to_num(Te, nan=0.01), 0.01)  # Avoid zero or negative temperatures
    return np.log(np.sqrt(2 * epsilon_0 * m_e) / (e * h) * (_Te_J(Te)) / np.sqrt(ne))


def lundquist_number(B: float | np.ndarray, L: float | np.ndarray, ne: float | np.ndarray,
                     Te: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the Lundquist number.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (Tesla)
    L : float or np.ndarray
        Characteristic length scale (m)
    ne : float or np.ndarray
        Electron density (m^-3)
    Te : float or np.ndarray
        Electron temperature (eV)

    Returns
    -------
    float or np.ndarray
        The Lundquist number.

    Notes
    -----
    The Lundquist number is a dimensionless quantity that characterizes the relative importance of magnetic diffusion to advection in a plasma.
    """
    # Convert Te from eV to Joules
    Te_J = Te * e

    # Calculate the resistivity using Spitzer formula (simplified)
    ln_lambda = coulomb_logarithm(Te, ne)
    eta = (m_e * (Te_J) ** (3 / 2)) / (e ** 2 * ne * ln_lambda * np.sqrt(2 * epsilon_0 * m_e))

    # Calculate the Alfvén speed
    v_A = B / np.sqrt(mu_0 * ne * m_e)

    # Calculate the Lundquist number
    S = v_A * L / eta

    return S


#
def spitzer_resistivity(Te: float | np.ndarray, ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the Spitzer resistivity.

    Parameters
    ----------
    Te : float or np.ndarray
        Electron temperature (eV)
    ne : float or np.ndarray
        Electron density (m^-3)

    Returns
    -------
    float or np.ndarray
        The Spitzer resistivity (Ohm m).

    Notes
    -----
    The Spitzer resistivity is a measure of the electrical resistivity of a plasma due to electron-ion collisions.
    """
    # Convert Te from eV to Joules
    Te_J = Te * e

    # Calculate the Coulomb logarithm
    ln_lambda = coulomb_logarithm(Te, ne)

    # Calculate the Spitzer resistivity using the formula
    eta = (m_e * (Te_J) ** (3 / 2)) / (e ** 2 * ne * ln_lambda * np.sqrt(2 * epsilon_0 * m_e))

    return eta
