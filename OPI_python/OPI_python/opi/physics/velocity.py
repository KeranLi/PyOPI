"""
Velocity Perturbation Calculations

Calculates horizontal velocity perturbations from Fourier solution.
Matches MATLAB's uPrime.m and streamline.m
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class VelocityCalculator:
    """
    Calculator for velocity perturbations and streamlines.
    
    Maintains persistent state for efficient repeated calculations
    with the same base parameters.
    """
    
    def __init__(self):
        self.s = None
        self.t = None
        self.n_s = 0
        self.n_t = 0
        self.k_s = None
        self.k_t = None
        self.k_z = None
        self.h_hat = None
        self.s_wind = None
        self.t_wind = None
        self.z_s0_prev = None
        self.s_s_wind = None
        self.t_s_wind = None
        self.z_s_wind = None
    
    def compute_fourier_solution(self, x, y, h_grid, U, azimuth, NM, f_c, h_rho):
        """
        Compute and store Fourier solution.
        
        Parameters
        ----------
        x, y : ndarray
            Grid vectors (m)
        h_grid : ndarray
            Topography grid (m)
        U : float
            Wind speed (m/s)
        azimuth : float
            Wind direction (degrees)
        NM : float
            Buoyancy frequency (rad/s)
        f_c : float
            Coriolis frequency (rad/s)
        h_rho : float
            Density scale height (m)
        """
        from .fourier import fourier_solution, wind_grid
        
        # Compute Fourier solution
        fourier_result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        self.s = fourier_result['s']
        self.t = fourier_result['t']
        self.n_s = len(self.s)
        self.n_t = len(self.t)
        self.k_s = fourier_result['k_s']
        self.k_t = fourier_result['k_t']
        self.k_z = fourier_result['k_z']
        self.h_hat = fourier_result['h_hat']
        
        # Store coordinate transformation grids
        Sxy = fourier_result['Sxy']
        Txy = fourier_result['Txy']
        self.s_wind = Sxy
        self.t_wind = Txy
        
        self.z_s0_prev = None  # Reset stream surface cache
    
    def calculate_u_prime(self, z_bar: float, U: float, f_c: float, 
                         h_rho: float, is_first: bool = True) -> np.ndarray:
        """
        Calculate horizontal perturbation velocity u' at height z_bar.
        
        Matches MATLAB's uPrime function.
        
        Parameters
        ----------
        z_bar : float
            Base-state height (m)
        U : float
            Wind speed (m/s)
        f_c : float
            Coriolis frequency (rad/s)
        h_rho : float
            Density scale height (m)
        is_first : bool
            If True, compute Fourier solution first
        
        Returns
        -------
        u_prime_grid : ndarray
            u' grid in geographic coordinates (n_y x n_x)
        """
        if is_first or self.s is None:
            raise RuntimeError("Must call compute_fourier_solution first")
        
        # Create wavenumber grids
        K_S, K_T = np.meshgrid(self.k_s, self.k_t, indexing='ij')
        K_Z = self.k_z
        
        # Calculate u' in Fourier space
        # u'hat = (i*k_s + f_c*k_t/(U*k_s))/(k_s^2 + k_t^2) * (i*k_z + 1/(2*h_rho)) * i*k_s*U*h_hat * exp((i*k_z + 1/(2*h_rho))*z_bar)
        numerator = 1j * K_S + f_c * K_T / (U * K_S)
        denominator = K_S**2 + K_T**2
        vertical_factor = 1j * K_Z + 1 / (2 * h_rho)
        exponential = np.exp((1j * K_Z + 1 / (2 * h_rho)) * z_bar)
        
        u_prime_hat = (numerator / denominator) * vertical_factor * 1j * K_S * U * self.h_hat * exponential
        
        # Set DC term to zero (singularity at k_s=0)
        u_prime_hat[0, :] = 0
        
        # Transform to space domain
        u_prime_wind = np.fft.ifft2(u_prime_hat, s=(self.n_s, self.n_t))
        u_prime_wind = np.real(u_prime_wind)
        u_prime_wind = u_prime_wind[:self.n_s, :self.n_t]
        
        # Transform to geographic coordinates
        # Create interpolator for wind grid
        interp_s = RegularGridInterpolator(
            (self.s, self.t), 
            u_prime_wind,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        # Query at geographic coordinates
        points = np.column_stack([self.s_wind.ravel(), self.t_wind.ravel()])
        u_prime_grid = interp_s(points).reshape(self.s_wind.shape)
        
        return u_prime_grid
    
    def calculate_stream_surface(self, z_l0: float, U: float, f_c: float, 
                                  h_rho: float, is_first: bool = True) -> None:
        """
        Calculate stream surface at height z_l0.
        
        Parameters
        ----------
        z_l0 : float
            Stream surface height (m)
        U : float
            Wind speed (m/s)
        f_c : float
            Coriolis frequency (rad/s)
        h_rho : float
            Density scale height (m)
        is_first : bool
            If True, recompute Fourier solution
        """
        if is_first or self.s is None:
            raise RuntimeError("Must call compute_fourier_solution first")
        
        # Check if we need to recompute
        if not is_first and self.z_s0_prev == z_l0 and self.z_s_wind is not None:
            return
        
        self.z_s0_prev = z_l0
        
        K_S, K_T = np.meshgrid(self.k_s, self.k_t, indexing='ij')
        K_Z = self.k_z
        
        # Calculate z' at stream surface
        z_prime_hat = self.h_hat * np.exp((1j * K_Z + 1 / (2 * h_rho)) * z_l0)
        
        # Calculate s' and t' displacements
        # s'hat = (i*k_s + f_c*k_t/(U*k_s))/(k_s^2 + k_t^2) * (i*k_z + 1/(2*h_rho)) * z_prime_hat
        numerator_s = 1j * K_S + f_c * K_T / (U * K_S)
        denominator = K_S**2 + K_T**2
        vertical_factor = 1j * K_Z + 1 / (2 * h_rho)
        
        s_prime_hat = (numerator_s / denominator) * vertical_factor * z_prime_hat
        s_prime_hat[0, :] = 0  # Set DC term to zero
        
        # t'hat = (i*k_t - f_c/U)/(k_s^2 + k_t^2) * (i*k_z + 1/(2*h_rho)) * z_prime_hat
        numerator_t = 1j * K_T - f_c / U
        t_prime_hat = (numerator_t / denominator) * vertical_factor * z_prime_hat
        t_prime_hat[0, :] = 0  # Set DC term to zero
        
        # Transform to space domain
        z_s_wind_full = z_l0 + np.fft.ifft2(z_prime_hat, s=(self.n_s, self.n_t))
        self.z_s_wind = np.real(z_s_wind_full)[:self.n_s, :self.n_t]
        
        s_s_wind_full = np.fft.ifft2(s_prime_hat, s=(self.n_s, self.n_t))
        S, T = np.meshgrid(self.s, self.t, indexing='ij')
        self.s_s_wind = S + np.real(s_s_wind_full)[:self.n_s, :self.n_t]
        
        t_s_wind_full = np.fft.ifft2(t_prime_hat, s=(self.n_s, self.n_t))
        self.t_s_wind = T + np.real(t_s_wind_full)[:self.n_s, :self.n_t]
    
    def calculate_streamline(self, x_l0: float, y_l0: float, z_l0: float,
                            azimuth: float, x: np.ndarray, y: np.ndarray,
                            U: float, f_c: float, h_rho: float,
                            is_first: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate single streamline from starting point.
        
        Matches MATLAB's streamline function.
        
        Parameters
        ----------
        x_l0, y_l0, z_l0 : float
            Starting coordinates (m)
        azimuth : float
            Wind direction (degrees)
        x, y : ndarray
            Grid vectors
        U, f_c, h_rho : float
            Physical parameters
        is_first : bool
            If True, compute stream surface
        
        Returns
        -------
        x_l, y_l, z_l : ndarray
            Streamline coordinates
        s_l : ndarray
            Horizontal distance along streamline
        """
        from ..optimization.wind_path import wind_path
        
        # Calculate stream surface
        self.calculate_stream_surface(z_l0, U, f_c, h_rho, is_first)
        
        # Calculate path in x-y plane using wind_path
        x_bar_path, y_bar_path = wind_path(x_l0, y_l0, azimuth, x, y)
        
        # Convert to s-t coordinates
        azimuth_rad = np.deg2rad(azimuth)
        s_bar_path = x_bar_path * np.sin(azimuth_rad) + y_bar_path * np.cos(azimuth_rad)
        t_bar_path = -x_bar_path * np.cos(azimuth_rad) + y_bar_path * np.sin(azimuth_rad)
        
        # Interpolate to get streamline coordinates
        # Create interpolators for stream surface
        interp_s = RegularGridInterpolator(
            (self.t, self.s), self.s_s_wind.T,
            method='linear', bounds_error=False, fill_value=None
        )
        interp_t = RegularGridInterpolator(
            (self.t, self.s), self.t_s_wind.T,
            method='linear', bounds_error=False, fill_value=None
        )
        interp_z = RegularGridInterpolator(
            (self.t, self.s), self.z_s_wind.T,
            method='linear', bounds_error=False, fill_value=None
        )
        
        # Query at path points
        points = np.column_stack([t_bar_path, s_bar_path])
        s_l = interp_s(points)
        t_l = interp_t(points)
        z_l = interp_z(points)
        
        # Convert back to x-y coordinates
        x_l = s_l * np.sin(azimuth_rad) - t_l * np.cos(azimuth_rad)
        y_l = s_l * np.cos(azimuth_rad) + t_l * np.sin(azimuth_rad)
        
        # Calculate horizontal distance along path
        ds = np.sqrt(np.diff(x_l)**2 + np.diff(y_l)**2)
        s_l_cumsum = np.concatenate([[0], np.cumsum(ds)])
        
        return x_l, y_l, z_l, s_l_cumsum


def calculate_u_prime(x, y, h_grid, z_bar, U, azimuth, NM, f_c, h_rho,
                     is_first: bool = True) -> np.ndarray:
    """
    Convenience function to calculate u' perturbation velocity.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors (m)
    h_grid : ndarray
        Topography grid (m)
    z_bar : float
        Height for calculation (m)
    U : float
        Wind speed (m/s)
    azimuth : float
        Wind direction (degrees)
    NM : float
        Buoyancy frequency (rad/s)
    f_c : float
        Coriolis frequency (rad/s)
    h_rho : float
        Density scale height (m)
    is_first : bool
        If True, initialize Fourier solution
    
    Returns
    -------
    u_prime : ndarray
        Perturbation velocity grid (same shape as h_grid)
    """
    calc = VelocityCalculator()
    
    if is_first:
        calc.compute_fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
    
    return calc.calculate_u_prime(z_bar, U, f_c, h_rho, is_first)
