from __future__ import annotations

import matplotlib.pyplot as plt
from typing import Callable
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d

from ingkit.physics.X_ray import AbsorptionFilter, FilterSet
from ingkit.physics.plasma import brems


class DoubleFilter:
    def __init__(self,
                 filter1: AbsorptionFilter | FilterSet,
                 filter2: AbsorptionFilter | FilterSet,
                 E_ph: np.ndarray = None) -> None:
        """
        Class to calculate the intensity ratios of two X-ray filters over a range of electron temperatures.

        Parameters
        ----------
        filter1: AbsorptionFilter | FilterSet
        filter2: AbsorptionFilter | FilterSet
        E_ph: np.ndarray
            Photon energy array in eV. If None, defaults to np.linspace(0, 3e5, 3000).
        """

        self.filter1 = filter1
        self.filter2 = filter2
        E_ph_default = np.logspace(0, np.log10(3e5), 3000) if E_ph is None else E_ph
        _E_ph = np.union1d(np.union1d(filter1.E_ph, filter2.E_ph), E_ph_default)  # in eV
        self.E_ph = np.unique(_E_ph)  # in eV, sorted unique photon energy array for both filters and the default range
        self._Te_from_ratio = None  # function to get Te from ratio

    @staticmethod
    def from_materials(material1: str, thickness1: float,
                       material2: str, thickness2: float,
                       photon_energy: np.ndarray = None) -> "DoubleFilter":
        """
        Create DoubleFilter from materials and thicknesses.

        Parameters
        ----------
        material1: str
            Material of filter1.
        thickness1: float
            Thickness of filter1 in micrometers.
        material2: str
            Material of filter2.
        thickness2: float
            Thickness of filter2 in micrometers.
        photon_energy: np.ndarray
            Photon energy array in eV. If None, defaults to np.linspace(0,
            3e4, 1000).

        Returns
        -------
        DoubleFilter
        """
        filter1 = AbsorptionFilter(material=material1, thickness=thickness1)
        filter2 = AbsorptionFilter(material=material2, thickness=thickness2)
        return DoubleFilter(filter1, filter2, E_ph=photon_energy)

    def transmissions(self, E_ph: np.ndarray, angle: float = 0.0, squeeze: bool = True                      ) -> np.ndarray:
        """
        Calculate the transmissions of both filters at given photon energies.

        Parameters
        ----------
        E_ph : np.ndarray
            Photon energy array in eV.
        angle : float
            Angle of incidence in radians. Default is 0 (normal incidence).
        squeeze : bool
            Whether to squeeze the output arrays. Default is True.

        Returns
        -------
        [trans1, trans2] : np.ndarray
            Transmissions of filter1 and filter2 at the specified photon energies.
        """
        trans1 = self.filter1.transmission_angle(E_ph=E_ph, angle=angle, squeeze=squeeze)
        trans2 = self.filter2.transmission_angle(E_ph=E_ph, angle=angle, squeeze=squeeze)
        return trans1, trans2

    def intensities(self,
                    Te: float | np.ndarray, ne: float = 5e18, Z_eff: float = 1.0,
                    E_ph: np.ndarray = None,
                    angle: float | np.ndarray = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the intensities of both filters over a range of electron temperatures.

        Parameters
        ----------
        Te : float or np.ndarray
            Electron temperature in eV.
        E_ph : np.ndarray
            Photon energy array in eV.
        angle : float | np.ndarray
            Angle of incidence in radians. Default is 0 (normal incidence).

        Returns
        -------
        intensity1 : np.ndarray
            Intensity through filter1.
        intensity2 : np.ndarray
            Intensity through filter2.
        """
        E_ph = self.E_ph if E_ph is None else E_ph  # in eV (n_Eph)
        angle = np.atleast_1d(angle)  # angle_shape = (n_angle, (m_angle, ...)) unknown shape
        angle_dim = angle.ndim
        spectrum = brems.bremsstrahlung_spectrum(Te=Te, ne=ne, Z_eff=Z_eff, E_ph=E_ph)  # (*TnZ_shape, n_Eph)
        spectrum_dim = spectrum.ndim
        trans1, trans2 = self.transmissions(E_ph=E_ph, angle=angle, squeeze=False)  # (*angle_shape, n_Eph)

        spectrum = np.expand_dims(spectrum, axis=tuple(range(-angle_dim-1, -1)))  # (*TnZ_shape, 1, ..., n_Eph)
        trans1 = np.expand_dims(trans1, axis=tuple(range(spectrum_dim-1)))  # (1, ..., *angle_shape, n_Eph)
        trans2 = np.expand_dims(trans2, axis=tuple(range(spectrum_dim-1)))  # (1, ..., *angle_shape, n_Eph)
        intensity1 = brems.integrate_spectrum(spectra=spectrum, E_ph=E_ph,
                                              transmission=trans1)  # (*TnZ_shape, *angle_shape)
        intensity2 = brems.integrate_spectrum(spectra=spectrum, E_ph=E_ph,
                                              transmission=trans2)  # (*TnZ_shape, *angle_shape)
        return intensity1.squeeze(), intensity2.squeeze()  # (*TnZ_shape, *angle_shape)

    def intensity_ratios(self, Te: float | np.ndarray,
                         E_ph: np.ndarray = None,
                         angle: float | np.ndarray = 0.0
                         ) -> tuple[np.ndarray, np.ndarray]:
        # """
        # Calculate the intensity ratios of the two filters over a range of electron temperatures.
        #
        # Parameters
        # ----------
        # Te : float or np.ndarray
        #     Electron temperature in eV.
        # E_ph : np.ndarray
        #     Photon energy array in eV.
        # angle : float | np.ndarray
        #     Angle of incidence in radians. Default is 0 (normal incidence).
        #
        # Returns
        # -------
        # ratio_12 : np.ndarray
        #     Intensity ratio of filter2 to filter1.
        # ratio_21 : np.ndarray
        #     Intensity ratio of filter1 to filter2.
        # """
        """
        Calculate the intensity ratios of the two filters over a range of electron temperatures.

        Parameters
        ----------
        Te : float or np.ndarray
            Electron temperature (eV).
        E_ph : np.ndarray, optional
            Photon energy array (eV). Default is None.
            If None, the function will use the photon energy array defined in the DoubleFilter instance.
        angleE_ph : np.ndarray, optional
            Photon energy array (eV). If None, the instance photon energy array is used. Default is None.
        angle : float or np.ndarray, optional
            Angle of incidence (radians). Default is 0.

        Returns
        -------
        ratio_12 : np.ndarray
            Intensity ratio of filter2 to filter1.
        ratio_21 : np.ndarray
            Intensity ratio of filter1 to filter2.
        """
        intensity1, intensity2 = self.intensities(Te=Te, E_ph=E_ph, angle=angle)  # (N_Te,) or (N_Te, angle)
        ratio_12 = intensity2 / intensity1
        ratio_21 = intensity1 / intensity2
        return ratio_12, ratio_21

    def plot_ratios(self, Te: float | np.ndarray = None, E_photon: np.ndarray = None, angle: float = 0.0
                    ) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        """
        Plot the intensity ratios of the two filters over a range of electron temperatures.

        Parameters
        ----------
        Te : float or np.ndarray
            Electron temperature in eV.
        E_photon : np.ndarray
            Photon energy array in eV.
        """
        Te = np.linspace(10, 300, 100) if Te is None else Te
        E_photon = self.E_ph if E_photon is None else E_photon
        intensity1, intensity2 = self.intensities(Te=Te, E_ph=E_photon, angle=angle)
        ratio_12, ratio_21 = self.intensity_ratios(Te=Te, E_ph=E_photon, angle=angle)

        fig, ax = plt.subplots()
        l1 = ax.plot(Te, intensity1, "C0-",
                     label=f"{self.filter1.material} ({self.filter1.thickness} um)")
        l2 = ax.plot(Te, intensity2, "C1-",
                     label=f"{self.filter2.material} ({self.filter2.thickness} um)")
        ax.set_xlim(Te[0], Te[-1])
        ax.set_xlabel("Electron Temperature [eV]")
        ax.set_ylabel("Intensity [arb. unit]")
        ax.set_ylim(0, None)

        ax2 = ax.twinx()
        l3 = ax2.plot(Te, ratio_12, "r--",
                      label=f"{self.filter2.material} ({self.filter2.thickness} um) / "
                            f"{self.filter1.material} ({self.filter1.thickness} um)")
        l4 = ax2.plot(Te, ratio_21, "k--",
                      label=f"{self.filter1.material} ({self.filter1.thickness} um) / "
                            f"{self.filter2.material} ({self.filter2.thickness} um)")
        ax2.set_ylabel("Intensity Ratio")
        ax2.set_ylim(0, 10)

        lines = l1 + l2 + l3 + l4
        labels = [line.get_label() for line in lines]
        ax2.legend(lines, labels, loc='upper left', fontsize=8)
        ax2.grid(True)
        fig.suptitle("Intensity Ratios of Two X-ray Filters")
        fig.tight_layout()
        return fig, ax, ax2

    @property
    def Te_from_ratio12(self) -> Callable[..., float | np.ndarray | None]:
        """
        Function to get electron temperature from intensity ratio (filter2 / filter1).

        Returns
        -------
        Te_from_ratio : function
            Function that takes intensity ratio as input and returns electron temperature.
            If no interpolation function is available, returns a function that always returns None.
            If only one angle was used to create the interpolation function, the angle parameter is ignored.
        """
        if self._Te_from_ratio is None:
            return lambda *args, **kwargs: None
        if isinstance(self._Te_from_ratio[0], interp1d):
            return lambda ratio, angle: self._Te_from_ratio[0](ratio)
        else:
            return lambda ratio, angle: self._Te_from_ratio[0](ratio, angle)

    @property
    def Te_from_ratio21(self) -> Callable[..., float | np.ndarray | None]:
        """
        Function to get electron temperature from intensity ratio (filter1 / filter2).

        Returns
        -------
        Te_from_ratio : function
            Function that takes intensity ratio as input and returns electron temperature.
            If no interpolation function is available, returns a function that always returns None.
            If only one angle was used to create the interpolation function, the angle parameter is ignored.
        """
        if self._Te_from_ratio is None:
            return lambda *args, **kwargs: None
        if isinstance(self._Te_from_ratio[1], interp1d):
            return lambda ratio, angle: self._Te_from_ratio[1](ratio)
        else:
            return lambda ratio, angle: self._Te_from_ratio[1](ratio, angle)

    def set_Te_from_ratio(self, Te: float | np.ndarray = None,
                          E_photon: np.ndarray = None,
                          angle: float | np.ndarray = 0.0) -> None:
        """
        Set the function to get electron temperature from intensity ratio.

        Parameters
        ----------
        Te: float or np.ndarray
            Electron temperature in eV.
        E_photon: np.ndarray
            Photon energy array in eV.
        angle: float | np.ndarray
            Angle of incidence in radians. Default is 0 (normal incidence).
        """
        Te = np.linspace(10, 1e3, 100) if Te is None else Te  # in eV
        E_photon = self.E_ph if E_photon is None else E_photon  # in eV

        if isinstance(angle, (int, float)):
            # only one angle, no need for 2D interpolation
            ratio_12, ratio_21 = self.intensity_ratios(Te=Te, E_ph=E_photon, angle=angle)
            if len(ratio_12) < 2 or len(ratio_21) < 2:
                self._Te_from_ratio = None
                return
            else:
                self._Te_from_ratio = (interp1d(ratio_12, Te, bounds_error=False, fill_value=np.nan),
                                       interp1d(ratio_21, Te, bounds_error=False, fill_value=np.nan))
                return
        elif isinstance(angle, (list, tuple, np.ndarray)):
            angle = np.asarray(angle)
            TT, AA = np.meshgrid(np.insert(Te, 0, 0), angle, indexing='ij')  # (N_Te, N_angle)
            ratio_12, ratio_21 = self.intensity_ratios(Te=Te, E_ph=E_photon, angle=angle)
            ratio_12 = ratio_12 if ratio_12.ndim == 2 else ratio_12[:, None]  # (N_Te, N_angle)
            ratio_21 = ratio_21 if ratio_21.ndim == 2 else ratio_21[:, None]  # (N_Te, N_angle)
            if len(ratio_12) < 2 or len(ratio_21) < 2:
                self._Te_from_ratio = None
                return
            points12 = np.column_stack((np.insert(ratio_12, 0, np.zeros(ratio_12.shape[1]), axis=0).ravel(),
                                        AA.ravel()))
            points21 = np.column_stack((np.insert(ratio_21, 0, np.zeros(ratio_21.shape[1]), axis=0).ravel(),
                                        AA.ravel()))

            self._Te_from_ratio = (LinearNDInterpolator(points12, TT.ravel(),
                                                        fill_value=np.nan, rescale=True),
                                   LinearNDInterpolator(points21, TT.ravel(),
                                                        fill_value=np.nan, rescale=True))
            return
        else:
            raise ValueError("angle must be a float, list, tuple, or np.ndarray.")


if __name__ == "__main__":
    # Example usage
    poly_25um_w_Al = FilterSet([AbsorptionFilter(material="polyimide", thickness=25.0),
                                AbsorptionFilter(material="Al", thickness=0.03)])  # 25 um polyimide + 30 nm Al
    poly_50um_w_Al = FilterSet([AbsorptionFilter(material="polyimide", thickness=50.0),
                                AbsorptionFilter(material="Al", thickness=0.03)])  # 50 um polyimide + 30 nm Al
    poly_75um_w_Al = FilterSet([AbsorptionFilter(material="polyimide", thickness=75.0),
                                AbsorptionFilter(material="Al", thickness=0.03)])  # 75 um polyimide + 30 nm Al
    poly_125um_w_Al = FilterSet([AbsorptionFilter(material="polyimide", thickness=125.0),
                                 AbsorptionFilter(material="Al", thickness=0.03)])  # 125 um polyimide + 30 nm Al

    double_filter = DoubleFilter(poly_25um_w_Al, poly_125um_w_Al, E_ph=None)
    Te = np.linspace(10, 2500, 100)  # in eV
    # fig, ax, ax2 = double_filter.plot_ratios(Te=Te, E_photon=None)
    # ax2.set_ylim(0, 5)
    # plt.show()

    # for filter_a, filter_b in combinations([poly_25um_w_Al, poly_50um_w_Al, poly_75um_w_Al, poly_125um_w_Al], 2):
    #     double_filter = DoubleFilter(filter_a, filter_b, E_ph=None)
    #     Te = np.linspace(10, 2500, 100)  # in eV
    #     fig, ax, ax2 = double_filter.plot_ratios(Te=Te, E_photon=None)
    #     ax2.set_ylim(0, 5)
    #
    #     fig.savefig(f"DoubleFilter_{filter_a.filters[0].material}_{filter_a.filters[0].thickness}um_"
    #                 f"{filter_b.filters[0].material}_{filter_b.filters[0].thickness}um.png",
    #                 dpi=300, bbox_inches='tight')
    #     ax2.set_ylim(0, 2)
    #     fig.savefig(f"DoubleFilter_{filter_a.filters[0].material}_{filter_a.filters[0].thickness}um_"
    #                 f"{filter_b.filters[0].material}_{filter_b.filters[0].thickness}um_2.png",
    #                 dpi=300, bbox_inches='tight')
    #     plt.show()
    #     plt.close(fig)
    fig, ax, ax2 = double_filter.plot_ratios(Te=Te, E_photon=None)
    ax2.set_ylim(0, 5)
    double_filter.set_Te_from_ratio(Te=Te, E_photon=None, angle=0)
    ratio_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    Te_from_ratio12 = double_filter.Te_from_ratio12(ratio_test, angle=0.0)
    Te_from_ratio21 = double_filter.Te_from_ratio21(ratio_test, angle=0.0)
    print("Ratio (filter2/filter1):", ratio_test)
    print("Te from ratio (filter2/filter1):", Te_from_ratio12)
    print("Te from ratio (filter1/filter2):", Te_from_ratio21)
    ratio12_from_Te = double_filter.intensity_ratios(Te=Te_from_ratio12, angle=0.0)[0]
    ratio21_from_Te = double_filter.intensity_ratios(Te=Te_from_ratio21, angle=0.0)[1]
    print("Ratio from Te_from_ratio12:", ratio12_from_Te)
    print("Ratio from Te_from_ratio21:", ratio21_from_Te)
