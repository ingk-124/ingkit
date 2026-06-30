# src/ingkit/io/read_vmec.py
# Read VMEC output files (wout~~~.nc or wout~~~.txt) and return a dictionary of the data.

from __future__ import annotations

from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np


class VMECData:
    """
    A class to hold VMEC data read from a file.
    """

    def __init__(self, file: str | Path):
        self.file = Path(file)
        self.data: dict[str, Any] = {}
        self._read_file()
        self.ns = int(self.data['ns'])
        self.xm = np.asarray(self.data['xm'])  # (mn,) array of poloidal mode numbers
        self.xn = np.asarray(self.data['xn'])  # (mn,) array of toroidal mode numbers
        self.xm_nyq = np.asarray(self.data['xm_nyq'])  # (mn_nyq,) Nyquist poloidal modes
        self.xn_nyq = np.asarray(self.data['xn_nyq'])  # (mn_nyq,) Nyquist toroidal modes
        self.nfp = int(self.data['nfp'])
        self.lasym = int(self.data['lasym__logical__'])
        self.s_arr = np.linspace(0, 1, self.ns, endpoint=True)

    def _read_nc(self):
        with nc.Dataset(self.file, 'r') as ds:
            for var_name in ds.variables:
                self.data[var_name] = ds.variables[var_name][:]

    def _read_file(self):
        if self.file.suffix == '.nc':
            self._read_nc()
        else:
            raise ValueError(f"Unsupported file format: {self.file.suffix}")

    @staticmethod
    def _validate_angles(u_arr: np.ndarray, v_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_arr = np.asarray(u_arr, dtype=float)
        v_arr = np.asarray(v_arr, dtype=float)
        if u_arr.ndim != 1 or v_arr.ndim != 1:
            raise ValueError("u_arr and v_arr must be one-dimensional arrays")
        return u_arr, v_arr

    def get_derivatives(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the surface coordinates (R, Z) and their derivatives with respect to s, u, v.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        x : tuple of np.ndarray
            (R, Z, Zeta) arrays of shape (ns, nu, nv).
        dx_ds : tuple of np.ndarray
            (dR/ds, dZ/ds) arrays of shape (ns, nu, nv).
        dx_du : tuple of np.ndarray
            (dR/du, dZ/du) arrays of shape (ns, nu, nv).
        dx_dv : tuple of np.ndarray
            (dR/dv, dZ/dv) arrays of shape (ns, nu, nv).
        """

        u_arr, v_arr = self._validate_angles(u_arr, v_arr)

        R_mn_c = self.data.get('rmnc', np.zeros((self.ns, len(self.xm))))  # (ns, mn)
        R_mn_s = self.data.get('rmns', np.zeros((self.ns, len(self.xm))))  # (ns, mn)
        Z_mn_c = self.data.get('zmnc', np.zeros((self.ns, len(self.xm))))  # (ns, mn)
        Z_mn_s = self.data.get('zmns', np.zeros((self.ns, len(self.xm))))  # (ns, mn)

        dR_mn_c_ds = np.gradient(R_mn_c, self.s_arr, axis=0)  # (ns, mn)
        dR_mn_s_ds = np.gradient(R_mn_s, self.s_arr, axis=0)  # (ns, mn)
        dZ_mn_c_ds = np.gradient(Z_mn_c, self.s_arr, axis=0)  # (ns, mn)
        dZ_mn_s_ds = np.gradient(Z_mn_s, self.s_arr, axis=0)  # (ns, mn)

        angles = self.xm[:, None, None] * u_arr[None, :, None] \
                 - self.xn[:, None, None] * v_arr[None, None, :]  # (mn, nu, nv)
        cosines = np.cos(angles)  # (mn, nu, nv)
        sines = np.sin(angles)  # (mn, nu, nv)

        R = np.tensordot(R_mn_c, cosines, axes=([1], [0])) \
            + np.tensordot(R_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        Z = np.tensordot(Z_mn_c, cosines, axes=([1], [0])) \
            + np.tensordot(Z_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        Zeta = np.broadcast_to(
            v_arr[None, None, :], (self.ns, u_arr.size, v_arr.size)
        )  # (ns, nu, nv)

        dR_ds = np.tensordot(dR_mn_c_ds, cosines, axes=([1], [0])) \
                + np.tensordot(dR_mn_s_ds, sines, axes=([1], [0]))  # (ns, nu, nv)
        dZ_ds = np.tensordot(dZ_mn_c_ds, cosines, axes=([1], [0])) \
                + np.tensordot(dZ_mn_s_ds, sines, axes=([1], [0]))  # (ns, nu, nv)

        dR_du = np.tensordot(R_mn_c, self.xm[:, None, None] * (-sines), axes=([1], [0])) \
                + np.tensordot(R_mn_s, self.xm[:, None, None] * cosines, axes=([1], [0]))  # (ns, nu, nv)
        dZ_du = np.tensordot(Z_mn_c, self.xm[:, None, None] * (-sines), axes=([1], [0])) \
                + np.tensordot(Z_mn_s, self.xm[:, None, None] * cosines, axes=([1], [0]))  # (ns, nu, nv)

        dR_dv = np.tensordot(R_mn_c, (-self.xn[:, None, None]) * (-sines), axes=([1], [0])) \
                + np.tensordot(R_mn_s, (-self.xn[:, None, None]) * cosines, axes=([1], [0]))  # (ns, nu, nv)
        dZ_dv = np.tensordot(Z_mn_c, (-self.xn[:, None, None]) * (-sines), axes=([1], [0])) \
                + np.tensordot(Z_mn_s, (-self.xn[:, None, None]) * cosines, axes=([1], [0]))  # (ns, nu, nv)

        return (R, Z, Zeta), (dR_ds, dZ_ds), (dR_du, dZ_du), (dR_dv, dZ_dv)

    def get_B_field_contravariant(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the contravariant components of the magnetic field (B^u, B^v) in the (u, v) coordinate system.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        B_sup : tuple of np.ndarray
            (B^u, B^v) arrays of shape (ns, nu, nv).
        """

        u_arr, v_arr = self._validate_angles(u_arr, v_arr)
        B_sup_u_mn_c = self.data.get('bsupumnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        B_sup_u_mn_s = self.data.get('bsupumns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        B_sup_v_mn_c = self.data.get('bsupvmnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        B_sup_v_mn_s = self.data.get('bsupvmns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)

        angles = self.xm_nyq[:, None, None] * u_arr[None, :, None] \
                 - self.xn_nyq[:, None, None] * v_arr[None, None, :]  # (mn_nyq, nu, nv)
        cosines = np.cos(angles)  # (mn_nyq, nu, nv)
        sines = np.sin(angles)  # (mn_nyq, nu, nv)

        B_sup_u = np.tensordot(B_sup_u_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(B_sup_u_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        B_sup_v = np.tensordot(B_sup_v_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(B_sup_v_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)

        return (B_sup_u, B_sup_v)

    def get_current_contravariant(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """Calculate contravariant current components ``(J^u, J^v)``."""
        u_arr, v_arr = self._validate_angles(u_arr, v_arr)
        J_sup_u_mn_c = self.data.get('currumnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        J_sup_u_mn_s = self.data.get('currumns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        J_sup_v_mn_c = self.data.get('currvmnc', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)
        J_sup_v_mn_s = self.data.get('currvmns', np.zeros((self.ns, len(self.xm_nyq))))  # (ns, mn_nyq)

        angles = self.xm_nyq[:, None, None] * u_arr[None, :, None] \
                 - self.xn_nyq[:, None, None] * v_arr[None, None, :]  # (mn_nyq, nu, nv)
        cosines = np.cos(angles)  # (mn_nyq, nu, nv)
        sines = np.sin(angles)  # (mn_nyq, nu, nv)

        # \sum_mn J^u_mnc * cos(angles) + \sum_mn J^u_mns * sin(angles)
        J_sup_u = np.tensordot(J_sup_u_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(J_sup_u_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)
        J_sup_v = np.tensordot(J_sup_v_mn_c, cosines, axes=([1], [0])) \
                  + np.tensordot(J_sup_v_mn_s, sines, axes=([1], [0]))  # (ns, nu, nv)

        return (J_sup_u, J_sup_v)

    def get_current_contraveriant(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """Deprecated misspelling of :meth:`get_current_contravariant`."""
        return self.get_current_contravariant(u_arr, v_arr)


    def get_B_field_cylindrical(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the magnetic field components (B_R, B_Z, B_phi) in cylindrical coordinates.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        B : tuple of np.ndarray
            (B_R, B_Z, B_Zeta) arrays of shape (ns, nu, nv).
        """
        (R, Z, Zeta), (dR_ds, dZ_ds), (dR_du, dZ_du), (dR_dv, dZ_dv) = self.get_derivatives(u_arr, v_arr)
        (B_sup_u, B_sup_v) = self.get_B_field_contravariant(u_arr, v_arr)

        # B_R = B^u * dR/du + B^v * dR/dv
        # B_Zeta = B^v * R
        # B_Z = B^u * dZ/du + B^v * dZ/dv
        B_R = B_sup_u * dR_du + B_sup_v * dR_dv  # (ns, nu, nv)
        B_Z = B_sup_u * dZ_du + B_sup_v * dZ_dv  # (ns, nu, nv)
        B_Zeta = B_sup_v * R  # (ns, nu, nv)

        return (R, Z, Zeta), (B_R, B_Z, B_Zeta)

    def get_current_cylindrical(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate current components (J_R, J_Z, J_phi) in cylindrical coordinates.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        J : tuple of np.ndarray
            (J_R, J_Z, J_Zeta) arrays of shape (ns, nu, nv).
        """
        (R, Z, Zeta), (dR_ds, dZ_ds), (dR_du, dZ_du), (dR_dv, dZ_dv) = self.get_derivatives(u_arr, v_arr)
        (J_sup_u, J_sup_v) = self.get_current_contravariant(u_arr, v_arr)

        # J_R = J^u * dR/du + J^v * dR/dv
        # J_Zeta = J^v * R
        # J_Z = J^u * dZ/du + J^v * dZ/dv
        J_R = J_sup_u * dR_du + J_sup_v * dR_dv  # (ns, nu, nv)
        J_Z = J_sup_u * dZ_du + J_sup_v * dZ_dv  # (ns, nu, nv)
        J_Zeta = J_sup_v * R  # (ns, nu, nv)

        return (R, Z, Zeta), (J_R, J_Z, J_Zeta)

    def get_B_field_cartesian(self, u_arr: np.ndarray, v_arr: np.ndarray):
        """
        Calculate the magnetic field components (B_x, B_y, B_z) in Cartesian coordinates.

        Parameters
        ----------
        u_arr : np.ndarray
            Array of poloidal angles (u) of shape (nu,).
        v_arr : np.ndarray
            Array of toroidal angles (v) of shape (nv,).

        Returns
        -------
        B : tuple of np.ndarray
            (B_x, B_y, B_z) arrays of shape (ns, nu, nv).
        """
        (R, Z, Zeta), (B_R, B_Z, B_phi) = self.get_B_field_cylindrical(u_arr, v_arr)

        # Convert cylindrical to Cartesian coordinates
        X = R * np.cos(Zeta)
        Y = R * np.sin(Zeta)

        # Convert magnetic field components from cylindrical to Cartesian coordinates
        B_X = B_R * np.cos(Zeta) - B_phi * np.sin(Zeta)
        B_Y = B_R * np.sin(Zeta) + B_phi * np.cos(Zeta)

        return (X, Y, Z), (B_X, B_Y, B_Z)
