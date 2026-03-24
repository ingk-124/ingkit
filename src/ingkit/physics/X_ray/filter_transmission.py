# src/ingkit/physics/X_ray/filter_transmission.py
# calculate X-ray transmission
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import requests
from matplotlib import pyplot as plt
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from scipy import constants
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.2 Safari/605.1.15'
options = webdriver.ChromeOptions()
options.add_argument(ua)
options.add_argument('headless')
FILE_LOC = Path(__file__).parent


def _calc_att_length_from_sf(photon_energy: np.ndarray, f2: np.ndarray, num_density: float) -> np.ndarray:
    """
    Calculate X-ray attenuation length from atomic scattering factor (imaginary part)

    Parameters
    ----------
    photon_energy : np.ndarray (n,)
        photon energy [eV]
    f2 : np.ndarray (n,)
        atomic scattering factor (imaginary part)
    num_density : float
        number density [1/m^3]

    Returns
    -------
    att_len : np.ndarray (n, 2)
        attenuation length (photon energy [eV], attenuation length [um])

    Notes
    -----
    The attenuation length of X-ray is expressed as:
        mu_a = 2 * r_0 * Lambda * f2
        att_len = 1 / (N * mu_a)
    where r_0 is the classical electron radius, Lambda is the wavelength of X-ray,
    and N is the number density of the material.
    `E_ph` and `f2` should be the same length.
    """
    Lambda = constants.h * constants.c / (photon_energy * constants.e)  # [m]
    r_0 = constants.physical_constants["classical electron radius"][0]  # [m]
    mu_a = 2 * r_0 * Lambda * f2  # [m^-1]
    with np.errstate(divide='ignore'):
        att_len = 1 / (num_density * mu_a)  # [m]
    return np.array([photon_energy, att_len * 1e6]).T


def _calc_att_length_from_transmission(photon_energy: np.ndarray, thicknesses: list[float], transmissions: np.ndarray) -> np.ndarray:
    """
    Calculate X-ray attenuation length from transmission data

    Parameters
    ----------
    photon_energy : np.ndarray (n,)
        photon energy [eV]
    thicknesses : list of float
        thickness of the material [um]
    transmissions : np.ndarray (m, n)
        transmission data of the material

    Returns
    -------
    att_len : np.ndarray (n, 2)
        attenuation length (photon energy [eV], attenuation length [um])

    Notes
    -----
    The transmission of a thin film is expressed as:
        T = exp(-N * mu_a * d)
    where N is the number density of the material, mu_a is the attenuation length, and d is the thickness of the film.
    `att_len` is defined as 1 / (N * mu_a) and `att_len` is calculated as:
        att_len = -(d1 - d2) / ln(T1 / T2)
    where d1 and d2 are the thickness of the material, and T1 and T2 are the transmission of the material, respectively.
    The attenuation length is calculated for all combinations of thicknesses and the median value is returned.
    """
    thicknesses = np.array(thicknesses)
    transmissions = np.array(transmissions)
    if transmissions.shape[0] != len(thicknesses):
        raise ValueError("transmissions.shape[0] must be equal to len(thicknesses)")
    att_len_list = []
    for i, j in combinations(range(len(thicknesses)), 2):
        with np.errstate(divide='ignore'):
            att_len = -(thicknesses[i] - thicknesses[j]) / np.log(transmissions[i] / transmissions[j])
        att_len_list.append(att_len)
    arr = np.array(att_len_list)  # (m, n)
    arr[~np.isfinite(arr) | (arr <= 0)] = np.nan
    att_len = np.nanmedian(arr, axis=0)  # (n,)
    return np.array([photon_energy, att_len]).T


def _get_transmission_data_from_CXRO(material: str, thickness: float, density: float = -1,
                                     ph_min: float = 10, ph_max: float = 30000, n_pts: int = 1000,
                                     ph_scale: Literal["log", "lin"] = "log",
                                     save: bool = True, save_dir: Union[str, Path] = None, force: bool = False
                                     ) -> tuple[np.ndarray, dict[str, str | float]]:
    """
    Get transmission data from CXRO (https://henke.lbl.gov/optical_constants/filter2.html)

    Parameters
    ----------
    material : str
        material name or formula (e.g., "Al", "C22H10N2O5")
    thickness : float
        thickness of the material [um]
    density : float, optional
        material density [g/cm^3] Default is -1.
    ph_min : float, optional
        minimum photon energy [eV] (larger than 10) Default is 10.
    ph_max : float, optional
        maximum photon energy [eV] (smaller than 30000) Default is 30000.
    n_pts : int, optional
        number of points (maximum: 1000) Default is 1000.
    ph_scale : str, optional
        photon energy scale ("log" or "lin") Default is "log".
    save : bool, optional
        save transmission data Default is True.
    save_dir : str, optional
        save directory Default is "./transmission".

    Returns
    -------
    data : np.ndarray
        transmission data (photon energy [eV], transmission)
    """
    save_dir = Path("./transmission") if save_dir is None else Path(save_dir)

    thickness = str(thickness)
    density = str(density)
    file_name = save_dir / f"{material}_{thickness.replace('.', '_')}.dat"
    if save_dir.exists() and file_name.exists() and not force:
        data = np.loadtxt(file_name, skiprows=2, dtype=float)
        info = dict(material=material, density=density)
    else:
        driver = webdriver.Chrome(options=options)
        url = "https://henke.lbl.gov/optical_constants/filter2.html"
        driver.get(url)
        pulldown = driver.find_element(By.NAME, "Material")
        opt = [_.get_attribute('value') for _ in Select(pulldown).options]
        time.sleep(0.5)

        def change_value(name: str, value: str) -> None:
            driver.execute_script(f'document.getElementsByName("{name}")[0].value = "{value}";')
            return

        if material in opt:
            Select(pulldown).select_by_visible_text(material)
        else:
            try:
                _ = Composition(material)
            except ValueError:
                driver.quit()
                raise ValueError(f"{material} is not a valid material name or formula")

            change_value("Formula", material)
        change_value("Density", density)
        change_value("Thickness", thickness)
        change_value("Min", str(ph_min))
        change_value("Max", str(ph_max))
        change_value("Npts", str(n_pts))

        if ph_scale == "log":
            Select(driver.find_element(By.NAME, "Plot")).select_by_visible_text("LogLin")
        elif ph_scale == "lin":
            Select(driver.find_element(By.NAME, "Plot")).select_by_visible_text("LinLin")

        time.sleep(0.5)  # wait for 0.5 sec
        driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit Request']").click()
        driver.switch_to.window(driver.window_handles[-1])
        # time.sleep(0.5)  # wait for 0.5 sec
        time.sleep(5)  # wait for 0.5 sec
        try:
            dat_url = driver.find_element(By.XPATH, "//a").get_attribute("href")
        except Exception as e:
            print(e)
            driver.quit()
            return
        response = requests.get(dat_url)
        driver.quit()
        data = np.array([_.strip().split() for _ in response.text.split('\n') if _][2:], dtype=float)
        density = float(re.findall(r"Density=([\d.]+)", response.text)[0])  # [g/cm^3]

        info = dict(material=material, density=density)

        if save:
            save_dir.mkdir(exist_ok=True, parents=True)
            with open(file_name, "w") as f:
                f.write(response.text)

        info = dict(material=info.get("material"),
                    density=float(info.get("density")))
    return data, info


def _attenuation_length(material: str, density: float = -1,
                        thicknesses: tuple[float, ...] = (1e-2, 1e-1, 1, 10),
                        save: bool = True, save_dir: str | Path | None = None,
                        CXRO_kw: dict[str, Any] | None = None, force: bool = False
                        ) -> tuple[np.ndarray, dict[str, str | float]]:
    """
    Calculate X-ray transmission

    Parameters
    ----------
    material : str
        material name or formula (e.g., "Al", "C22H10N2O5")
    density : float, optional
        material density [g/cm^3] Default is -1.
    d1d2 : tuple, optional
        thickness of the material [um] Default is (1e-2, 5e-2.)
    save : bool, optional
        save transmission data Default is True.
    save_dir : str, optional
        save directory Default is "./att_len".
    CXRO_kw : dict, optional
        keyword arguments for _get_transmission_data_from_CXRO

    Returns
    -------
    data : np.ndarray (n, 2)
        attenuation length (photon energy [eV], attenuation length [um])
    """
    # if material is included in sf sf_database -> load f2 and calculate attenuation length
    # if material is not included in sf sf_database -> load attenuation length from CXRO
    sf_database = FILE_LOC / "sf"
    save_dir = Path("./att_len") if save_dir is None else Path(save_dir)
    file_path = save_dir / f"{material}_({density}).dat"
    sf_path = sf_database / f"{material.lower()}.nff"
    if not force and ((save_dir / f"{material}_({density}).dat").exists() or (
            FILE_LOC / f"att_len/{material}_({density}).dat").exists()):
        if (save_dir / f"{material}_({density}).dat").exists():
            file = save_dir / f"{material}_({density}).dat"
        else:
            file = FILE_LOC / f"att_len/{material}_({density}).dat"
        with open(file, "r") as f:
            info = f.readline()[1:].strip().split()
        info = dict([_.split("=") for _ in info])
        data = np.loadtxt(file, skiprows=2, dtype=float)
    else:
        if sf_path.exists():
            E, f1, f2 = np.loadtxt(sf_database / f"{material.lower()}.nff", skiprows=1, dtype=float).T
            m = Element(material).atomic_mass * constants.atomic_mass  # [kg]
            if density < 0:
                density_info = Element(material).data["Density of solid"]
                if "no data" in density_info:
                    raise ValueError(f"no data for {material}")
                else:
                    density_kg_m3 = float(re.findall(r"(\d+.\d+) kg m<sup>-3</sup>",
                                                     Element(material).data["Density of solid"])[0])  # [kg/m^3]
                    density = density_kg_m3 / 1e3  # [g/cm^3]
            else:
                density_kg_m3 = density * 1e3  # [kg/m^3]
            num_density = density_kg_m3 / m  # [1/m^3]
            data = _calc_att_length_from_sf(E, f2, num_density)
            info = dict(material=material, density=density)
        else:
            CXRO_kw = {} if CXRO_kw is None else CXRO_kw
            CXRO_kw.setdefault("force", force)
            CXRO_kw.setdefault("save", save)
            res = [_get_transmission_data_from_CXRO(material, thickness=d, density=density, **CXRO_kw) for d in
                   thicknesses]
            photon_energy = res[0][0][:, 0]  # (n,)
            transmission_data = np.array([_[0][:, 1] for _ in res])  # (len(thicknesses), n)
            info = res[0][1]
            data = _calc_att_length_from_transmission(photon_energy, np.array(thicknesses), transmission_data)
        if save:
            save_dir.mkdir(exist_ok=True, parents=True)
            header = (f"material={material} density={density}\n"
                      f"Photon Energy [eV]  Attenuation Length [um]")
            np.savetxt(file_path, data, fmt="%.6e", header=header)
    info = dict(material=info.get("material"),
                density=float(info.get("density")))
    return data, info


class AbsorptionFilter:
    def __init__(self, material: str, thickness: float = 1, density: float = -1,
                 force: bool = False, attn_kw: dict[str, Any] | None = None,
                 CXRO_kw: dict[str, Any] | None = None) -> None:
        """
        Calculate X-ray transmission

        Parameters
        ----------
        material : str
            material name or formula (e.g., "Al", "C22H10N2O5")
        thickness : float
            thickness of the material [um]
        density : float, optional
            material density [g/cm^3]. If density is negative, the density of solid is used Default is -1.
        force : bool, optional
            force to calculate attenuation length Default is False.
        attn_kw : dict, optional
            keyword arguments for _attenuation_length
            - save : bool, optional
                save transmission data Default is True.
            - save_dir : str, optional
                save directory Default is "./att_len".
            - CXRO_kw : dict, optional
                keyword arguments for _get_transmission_data_from_CXRO
        CXRO_kw : dict, optional
            keyword arguments for _get_transmission_data_from_CXRO
            - save : bool, optional
                save transmission data Default is True.
            - save_dir : str, optional
                save directory Default is "./transmission".
            - ph_min : float, optional
                minimum photon energy [eV] (larger than 10) Default is 10.
            - ph_max : float, optional
                maximum photon energy [eV] (smaller than 30000) Default is 30000.
            - n_pts : int, optional
                number of points (maximum: 1000) Default is 1000.
            - ph_scale : str, optional
                photon energy scale ("log" or "lin") Default is "log".
            - force : bool, optional
                force to calculate transmission data Default is False.

        Notes
        -----
        The attenuation length of X-ray is expressed as:
            mu_a = 2 * r_0 * Lambda * f2
            att_len = 1 / (N * mu_a)
        where r_0 is the classical electron radius, Lambda is the wavelength of X-ray,
        and N is the number density of the material.
        The transmission of a thin film is expressed as:
            T = exp(-N * mu_a * d)
        where N is the number density of the material, mu_a is the attenuation length, and d is the thickness of the film.
        Cite: https://henke.lbl.gov/optical_constants/intro.html
        """
        attn_kw = {} if attn_kw is None else attn_kw
        CXRO_kw = {} if CXRO_kw is None else CXRO_kw
        self._thickness = thickness
        _atten_len_data, info = _attenuation_length(material, density=density,
                                                    **attn_kw, CXRO_kw=CXRO_kw, force=force)
        self._info = info
        self._E_ph = _atten_len_data[:, 0].ravel()
        self._attn_len = _atten_len_data[:, 1].ravel()

    def __repr__(self) -> str:
        return f"AbsorptionFilter(material={self.material}, thickness={self.thickness}, density={self.density})"

    @property
    def material(self) -> str:
        return self._info["material"]

    @property
    def density(self) -> float:
        return self._info["density"]

    @property
    def E_ph(self) -> np.ndarray:
        return self._E_ph

    @property
    def attn_len(self) -> np.ndarray:
        return self._attn_len

    @property
    def thickness(self) -> float:
        return self._thickness

    @thickness.setter
    def thickness(self, thickness: float) -> None:
        self._thickness = thickness

    def interpolate(self, E_ph: np.ndarray) -> np.ndarray:
        """
        Interpolate attenuation length

        Parameters
        ----------
        E_ph : np.ndarray (n,)
            photon energy [eV]

        Returns
        -------
        att_len : np.ndarray (n,)
            attenuation length [um]
        """
        return np.interp(E_ph, self._E_ph, self._attn_len)

    def transmission(self, E_ph: np.ndarray | None = None, thickness: float | None = None) -> np.ndarray:
        """
        Calculate X-ray transmission of the material

        Parameters
        ----------
        E_ph : np.ndarray, optional
            photon energy [eV]
        thickness : float
            thickness of the material [um]

        Returns
        -------
        transmission : np.ndarray
            X-ray transmission
            If `E_ph` is None, the transmission is calculated for the photon energy of the attenuation length data.
        """
        return self.transmission_angle(E_ph=E_ph, angle=0, thickness=thickness)

    def transmission_angle(self, E_ph: np.ndarray | None = None, angle: np.ndarray | float = 0,
                           thickness: float | None = None, squeeze: bool = True) -> np.ndarray:
        """
        Calculate transmission with angle

        Parameters
        ----------
        E_ph : np.ndarray (n,)
            photon energy [eV]
        thickness : float
            thickness of the material [um]
        angle : float or np.ndarray, optional
            angle [rad]
        squeeze : bool, optional
            if True, squeeze the output array Default is True.

        Returns
        -------
        transmission : np.ndarray (n, ) or (m, n)
            X-ray transmission
            If `E_ph` is None, the transmission is calculated for the photon energy of the attenuation length data.

        Notes
        -----
        The thickness of the material is corrected by the cosine of the angle.
        At angles close to zero, the transmittance deviates from the actual value because refraction cannot be ignored.
        """
        angle = np.clip(np.abs(np.atleast_1d(angle)), 0, np.pi / 2 - 1e-3)  # avoid angles close to pi/2
        angle_dim = angle.ndim

        thickness = self._thickness if thickness is None else thickness
        E_ph = self._E_ph if E_ph is None else E_ph

        att_len = self.interpolate(E_ph)  # [um] (n_Eph,)
        thickness_eff = thickness / np.cos(angle)  # [um] (m, ...)

        att_len = np.expand_dims(att_len, axis=tuple(range(angle_dim)))  # (m, ..., n_Eph)
        thickness_eff = np.expand_dims(thickness_eff, axis=-1)  # (m, ..., 1)
        transmission = np.exp(-thickness_eff / att_len)  # (m, ..., n_Eph)
        transmission = transmission.squeeze() if squeeze else transmission
        return np.clip(transmission, 0, 1)  # just in case of numerical errors

    def plot_transmission(self, E_ph: np.ndarray | None = None, thickness: float | None = None,
                          ax: plt.Axes | None = None, **kwargs: Any) -> plt.Axes:
        """
        Plot X-ray transmission

        Parameters
        ----------
        E_ph : np.ndarray, optional
            photon energy [eV]
        thickness : float
            thickness of the material [um]
        ax : plt.Axes, optional
            plt.Axes object
        kwargs : dict, optional
            keyword arguments for plt.plot
        """
        if ax is None:
            fig, ax = plt.subplots()

        thickness = self._thickness if thickness is None else thickness
        transmission = self.transmission(E_ph, thickness)
        E_ph = self._E_ph if E_ph is None else E_ph

        ax.plot(E_ph, transmission, **kwargs, label=f"{self.material} ({thickness:.2f}um)")
        ax.set_xlabel("Photon Energy [eV]")
        ax.set_ylabel("Transmission")
        return ax


class FilterSet:
    def __init__(self, filters: list[AbsorptionFilter]) -> None:
        """
        Set of absorption filters

        Parameters
        ----------
        filters : list of AbsorptionFilter
            list of AbsorptionFilter objects

        Notes
        -----
        The transmission of the filters set is calculated as the product of the transmission of each filters.
        """
        self._filters = filters

    def __repr__(self) -> str:
        return f"FilterSet(filters={self.filters})"

    @staticmethod
    def from_materials(materials: list[str], thicknesses: list[float], densities: list[float] | None = None,
                       force: bool = False, attn_kw: dict[str, Any] | None = None,
                       CXRO_kw: dict[str, Any] | None = None) -> "FilterSet":
        """
        Create FilterSet from materials, thicknesses, and densities

        Parameters
        ----------
        materials : list of str
            list of material names or formulas (e.g., "Al", "C22H10N2O5", "polyimide")
        thicknesses : list of float
            list of thicknesses of the materials [um]
        densities : list of float, optional
            list of material densities [g/cm^3]. If density is negative, the density of solid is used Default is None.
        force : bool, optional
            force to calculate attenuation length Default is False.
        attn_kw : dict, optional
            keyword arguments for _attenuation_length
            - save : bool, optional
                save transmission data Default is True.
            - save_dir : str, optional
                save directory Default is "./att_len".
            - CXRO_kw : dict, optional
                keyword arguments for _get_transmission_data_from_CXRO
        CXRO_kw : dict, optional
            keyword arguments for _get_transmission_data_from_CXRO

        Returns
        -------
        FilterSet
            FilterSet object
        """
        if densities is None:
            densities = [-1] * len(materials)
        if len(materials) != len(thicknesses) or len(materials) != len(densities):
            raise ValueError("materials, thicknesses, and densities must be the same length")
        filters = [AbsorptionFilter(material, thickness, density, force=force, attn_kw=attn_kw, CXRO_kw=CXRO_kw)
                   for material, thickness, density in zip(materials, thicknesses, densities)]
        return FilterSet(filters)


    @property
    def filters(self) -> list[AbsorptionFilter]:
        return self._filters

    @property
    def material(self) -> str:
        return " + ".join([f.material for f in self._filters])

    @property
    def thickness(self) -> list[float]:
        return [f.thickness for f in self._filters]

    def transmission(self, thickness: list[float] | None = None, E_ph: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate X-ray transmission of the filters set

        Parameters
        ----------
        E_ph : np.ndarray, optional
            photon energy [eV]

        Returns
        -------
        transmission : np.ndarray
            X-ray transmission
            If `E_ph` is None, the transmission is calculated for the photon energy of the first filters.
        """
        return self.transmission_angle(E_ph=E_ph, angle=0, thickness=thickness).ravel()

    def transmission_angle(self, E_ph: np.ndarray | None = None, angle: float = 0,
                           thickness: list[float] | None = None) -> np.ndarray:
        """
        Calculate transmission with angle

        Parameters
        ----------
        E_ph : np.ndarray (n,)
            photon energy [eV]
        angle : float
            angle [rad]
        thickness : list of float, optional
            thickness of each filters [um]

        Returns
        -------
        transmission : np.ndarray (n, m)
            transmission

        Notes
        -----
        The thickness of the material is corrected by the cosine of the angle.
        At angles close to zero, the transmittance deviates from the actual value because refraction cannot be ignored.
        """
        if thickness is None:
            thickness = [f.thickness for f in self._filters]
        elif isinstance(thickness, (int, float)):
            thickness = [thickness] * len(self._filters)
        E_ph = self._filters[0].E_ph if E_ph is None else E_ph
        transmissions = [f.transmission_angle(E_ph, angle, t) for f, t in zip(self._filters, thickness)]
        return np.prod(transmissions, axis=0)

    def plot_transmission(self, E_ph: np.ndarray | None = None, thickness: list[float] | None = None,
                          ax: plt.Axes | None = None, **kwargs: Any) -> plt.Axes:
        """
        Plot X-ray transmission

        Parameters
        ----------
        E_ph : np.ndarray, optional
            photon energy [eV]
        thickness : list of float, optional
            thickness of each filters [um]
        ax : plt.Axes, optional
            plt.Axes object
        kwargs : dict, optional
            keyword arguments for plt.plot
        """
        if ax is None:
            fig, ax = plt.subplots()

        E_ph = self._filters[0].E_ph if E_ph is None else E_ph
        thickness = [f.thickness for f in self._filters] if thickness is None else thickness
        if len(thickness) != len(self._filters):
            raise ValueError("thickness must be a list of the same length as the number of filters")
        transmission = self.transmission(thickness=thickness, E_ph=E_ph)
        for f, t in zip(self._filters, thickness):
            f.plot_transmission(E_ph=E_ph,
                                thickness=t, ax=ax, linewidth=1, linestyle="--", **kwargs)
        ax.plot(E_ph, transmission, color="k", label="Filter Set", **kwargs)
        ax.set_xlabel("Photon Energy [eV]")
        ax.set_ylabel("Transmission")

        return ax


if __name__ == '__main__':
    # files = (FILE_LOC / "sf").glob("*.nff")
    # for file in files:
    #     material = file.stem
    #     material = material[0].upper() + material[1:]
    #     print(material)
    #     try:
    #         Filter = AbsorptionFilter(material, force=True)
    #     except ValueError:
    #         print(f"no data for {material}")
    #         continue

    # filters = FilterSet([AbsorptionFilter('Al', 1.0),
    #                      AbsorptionFilter('polyimide', 0.5)])
    filters = FilterSet.from_materials(["Al", "polyimide"], thicknesses=[0.03, 125])
    ax = filters.plot_transmission()
    ax.set_xscale('log')
    plt.legend()
    plt.show()
