# src/ingkit/pyhsics/X_ray/brems.py
# Bremsstrahlung radiation calculations for X-ray physics.

from __future__ import annotations

import numpy as np
from scipy.constants import e, m_e, epsilon_0, h, c

from ingkit.physics.plasma.core import coulomb_logarithm
from ingkit.tools import type_check

factor = (e ** 6 * np.sqrt(e)) / (3 * np.sqrt(6) * np.pi ** (3 / 2) * h * (c ** 3) * (epsilon_0 ** 3) * m_e ** (3 / 2))


def gaunt_factor(Te: float | np.ndarray, ne: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the Gaunt factor for bremsstrahlung radiation.

    Parameters
    ----------
    Te : float or np.ndarray
        Electron temperature (unit: eV)
    ne : float or np.ndarray
        Electron density (unit: m^-3)

    Returns
    -------
    float or np.ndarray
        The Gaunt factor (dimensionless).
    """
    return (np.sqrt(3) / np.pi) * coulomb_logarithm(Te, ne)


def bremsstrahlung_spectrum(Te: float | np.ndarray, ne: float | np.ndarray, Z_eff: float = 1.,
                            E_ph: float | np.ndarray = None) -> float | np.ndarray:
    """
    Calculate the bremsstrahlung spectrum.

    Parameters
    ----------
    Te : float or np.ndarray
        Electron temperature (unit: eV)
    ne : float or np.ndarray
        Electron density (unit: m^-3)
    Z_eff : float, optional
        Effective charge state of the plasma (default is 1 for hydrogenic plasma).
    E_ph : float or np.ndarray, optional
        Photon energy (unit: eV). If not provided, the function will return the spectrum as a function of photon energy.

    Returns
    -------
    float or np.ndarray
        The bremsstrahlung spectrum (unit: photons/s/m^3/eV).
    """

    if E_ph is None:
        Te_max = np.max(Te)
        E_ph = np.logspace(0, np.log10(Te_max * 10), 100)  # Generate photon energy array if not provided
    E_ph = np.atleast_1d(E_ph)

    # Te = np.nan_to_num(Te, nan=0.01)
    # ne = np.nan_to_num(ne, nan=0)
    Te = np.maximum(Te, 0.01)  # Avoid division by zero or negative temperatures
    ne = np.maximum(ne, 0)  # Avoid negative densities

    factor_arr = factor * ne ** 2 * Z_eff / np.sqrt(Te) * gaunt_factor(Te, ne)
    spectrum = factor_arr[..., None] * np.exp(-E_ph / Te[..., None])
    return spectrum.squeeze()


def integrate_spectrum(spectra: float | np.ndarray, E_ph: float | np.ndarray,
                       transmission: float | np.ndarray = None,
                       nan_to_zero: bool = True) -> float | np.ndarray:
    """
    Calculate the integrated bremsstrahlung power over a given photon energy range, optionally applying a transmission function.

    Parameters
    ----------
    spectra : float or np.ndarray
        The bremsstrahlung spectrum (unit: photons/s/m^3/eV) as a function of photon energy.
    E_ph : float or np.ndarray
        Photon energy array corresponding to the spectra (unit: eV).
    transmission : float or np.ndarray, optional
        Transmission function to apply to the spectrum (dimensionless, between 0 and 1). If not provided, no transmission is applied.
        This should be the same shape as E_ph or broadcastable to the shape of spectra.

    Returns
    -------

    """
    if transmission is None:
        transmission = np.ones_like(E_ph)
    else:
        transmission = type_check.ensure_array_like_of_float(transmission)
    if transmission.shape != E_ph.shape:
        transmission = np.broadcast_to(transmission, E_ph.shape)

    spectra = spectra * transmission[None, :]
    intensities = np.trapezoid(spectra * E_ph[None, :], np.log(E_ph), axis=-1)
    if nan_to_zero:
        intensities = np.nan_to_num(intensities, nan=0)
    return intensities


if __name__ == '__main__':
    from ingkit.myplot import pyplot as plt

    E_ph = np.logspace(1, 5, 200)  # in eV
    Te = np.array([200, 500, 1000, 2000])  # in eV
    n_e = 1e18  # in m^-3
    Z_eff = 1.

    spectrum = bremsstrahlung_spectrum(Te, n_e, Z_eff, E_ph)

    fig, ax = plt.subplots()
    for i, T in enumerate(Te):
        ax.plot(E_ph, spectrum[i], label=f'$T_e = {T}$ eV')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$E_{ph}$ (eV)')
    ax.set_ylabel(r'$dP/dE_{ph}$ (W / m$^3$ / eV)')
    ax.set_xlim(10, E_ph[-1])
    ax.set_ylim(1e-5, 1e-2)
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()

    Te_arr = np.linspace(10, 2000, 10)[:, None]
    ne_arr = (np.linspace(1, 5, 5) * 1e18)[None, :]
    spectrum_arr = bremsstrahlung_spectrum(Te_arr, ne_arr, Z_eff, E_ph)
    plt.figure()
    for i in range(spectrum_arr.shape[1]):
        plt.plot(E_ph, spectrum_arr[0, i],
                 label=f'$n_e = {ne_arr[0, i]:.1e}$ m$^{{-3}}$')

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$E_{ph}$ (eV)')
    plt.ylabel(r'$dP/dE_{ph}$ (W / m$^3$ / eV)')
    plt.xlim(10, E_ph[-1])
    # plt.ylim(1e-5, 1e-2)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
