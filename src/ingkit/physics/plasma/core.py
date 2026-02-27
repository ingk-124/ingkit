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
        Electron temperature (unit: eV)

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
        Electron temperature (unit: Joules)

    Returns
    -------
    float or np.ndarray
        Electron temperature in eV.
    """
    return Te_J / e


# Frequencies

def f_pe(ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the electron plasma frequency.

    Parameters
    ----------
    ne : float or np.ndarray
        Electron density (unit: m^-3)

    Returns
    -------
    float or np.ndarray
        The electron plasma frequency (unit: Hz).
    """
    return np.sqrt(ne * e ** 2 / (epsilon_0 * m_e)) / (2 * np.pi)


def f_pi(ni: float | np.ndarray, Z: int = 1, A: int = 1) -> float | np.ndarray:
    """
    Calculate the ion plasma frequency.

    Parameters
    ----------
    ni : float or np.ndarray
        Ion density (unit: m^-3)
    Z : int, optional
        Ion charge state (default is 1 for singly ionized)
    A : int, optional
        Ion mass number (default is 1 for proton)

    Returns
    -------
    float or np.ndarray
        The ion plasma frequency (unit: Hz).
    """
    m_i = A * m_p
    return np.sqrt(ni * (Z * e) ** 2 / (epsilon_0 * m_i)) / (2 * np.pi)


def cyclotron_frequency(B: float | np.ndarray, q: float = e, m: float = m_e) -> float | np.ndarray:
    """
    Calculate the cyclotron frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (unit: Tesla)
    q : float, optional
        Charge of the particle (default is electron charge e)
    m : float, optional
        Mass of the particle (default is electron mass m_e)

    Returns
    -------
    float or np.ndarray
        The cyclotron frequency (unit: Hz).

    Notes
    -----
    The cyclotron frequency is the frequency at which charged particles spiral around magnetic field lines.
    """
    return q * B / (2 * np.pi * m)


def f_ce(B: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the electron cyclotron frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (unit: Tesla)

    Returns
    -------
    float or np.ndarray
        The electron cyclotron frequency (unit: Hz).
    """
    return cyclotron_frequency(B, q=e, m=m_e)


def f_ci(B: float | np.ndarray, Z: int = 1, A: int = 1) -> float | np.ndarray:
    """
    Calculate the ion cyclotron frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (unit: Tesla)
    Z : int, optional
        Ion charge state (default is 1 for singly ionized)
    A : int, optional
        Ion mass number (default is 1 for proton)

    Returns
    -------
    float or np.ndarray
        The ion cyclotron frequency (unit: Hz).
    """
    m_i = A * m_p
    return cyclotron_frequency(B, q=Z * e, m=m_i)


def f_lh(B: float | np.ndarray, ne: float | np.ndarray, Z: int = 1, A: int = 1
         ) -> float | np.ndarray:
    """
    Calculate the lower hybrid frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (unit: Tesla)
    ne : float or np.ndarray
        Electron density (unit: m^-3)
    Z : int, optional
        Ion charge state (default is 1 for singly ionized)
    A : int, optional
        Ion mass number (default is 1 for proton)

    Returns
    -------
    float or np.ndarray
        The lower hybrid frequency (unit: Hz).

    Notes
    -----
    The lower hybrid frequency is a characteristic frequency in magnetized plasmas, associated with the coupling of ion and electron motions.
    """
    _f_ce = f_ce(B)
    _f_ci = f_ci(B, Z, A)
    _f_pe = f_pe(ne)
    return np.sqrt(_f_ci * _f_ce / (1 + _f_pe ** 2 / _f_ce ** 2))


def f_uh(B: float | np.ndarray, ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the upper hybrid frequency.

    Parameters
    ----------
    B : float or np.ndarray
        Magnetic field strength (unit: Tesla)
    ne : float or np.ndarray
        Electron density (unit: m^-3)

    Returns
    -------
    float or np.ndarray
        The upper hybrid frequency (unit: Hz).

    Notes
    -----
    The upper hybrid frequency is a characteristic frequency in magnetized plasmas, associated with the coupling of electron motion and plasma oscillations.
    """
    _f_ce = f_ce(B)
    _f_pe = f_pe(ne)
    return np.sqrt(_f_ce ** 2 + _f_pe ** 2)


def o_mode_cutoff_f(ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the O-mode cutoff frequency.

    Parameters
    ----------
    ne : float or np.ndarray
        Electron density (unit: m^-3)

    Returns
    -------
    float or np.ndarray
        The O-mode cutoff frequency (unit: Hz).

    Notes
    -----
    The O-mode: Electric field oscillates parallel to the magnetic field.
    """
    return f_pe(ne)


def x_mode_cutoff_f(ne: float | np.ndarray, B: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the X-mode cutoff frequency.

    Parameters
    ----------
    ne : float or np.ndarray
        Electron density (unit: m^-3)
    B : float or np.ndarray
        Magnetic field strength (unit: Tesla)

    Returns
    -------
    float or np.ndarray
        The X-mode cutoff frequency (unit: Hz).

    Notes
    -----
    The X-mode: Electric field oscillates perpendicular to the magnetic field.
    """
    _f_ce = f_ce(B)
    _f_pe = f_pe(ne)
    return 0.5 * (_f_ce + np.sqrt(_f_ce ** 2 + 4 * _f_pe ** 2))


# Dimensionless values

def coulomb_logarithm(Te: float | np.ndarray, ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the Coulomb logarithm.

    Parameters
    ----------
    Te : float or np.ndarray
        Electron temperature (unit: eV)
    ne : float or np.ndarray
        Electron density (unit: m^-3)

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
        Magnetic field strength (unit: Tesla)
    L : float or np.ndarray
        Characteristic length scale (unit: m)
    ne : float or np.ndarray
        Electron density (unit: m^-3)
    Te : float or np.ndarray
        Electron temperature (unit: eV)

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
        Electron temperature (unit: eV)
    ne : float or np.ndarray
        Electron density (unit: m^-3)

    Returns
    -------
    float or np.ndarray
        The Spitzer resistivity (unit: Ohm m).

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
