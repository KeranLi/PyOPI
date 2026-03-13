"""
Fourier Solution for Linearized Euler Equations

This module calculates the Fourier solution for the linearized Euler equations
for flow of air over topography. The solution is based on a saturated base state
with a uniform buoyancy frequency, and uses the anelastic approximation.

Reference: Durran and Klemp, 1982
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .wind_grid import wind_grid


def fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho):
    """
    Calculates the Fourier solution for the linearized Euler equations
    for flow of air over topography.
    
    The solution is based on saturated base state with a uniform buoyancy 
    frequency, NM, and also uses the anelastic approximation.
    
    Reference: Durran and Klemp, 1982
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors indicating coordinates for geographic grids,
        with x and y oriented in the east and north directions, respectively
        (vectors with lengths nX and nY, m).
    h_grid : ndarray
        Geographic grid for topography, with x (east) in the row direction,
        and y (north) in the column direction (matrix, nY x nX, m).
    U : float
        Base-state wind speed (scalar, m/s).
    azimuth : float
        Base-state wind direction (down wind) in degrees from North.
    NM : float
        Saturation buoyancy frequency (scalar, rad/s).
    f_c : float
        Coriolis frequency (scalar, rad/s).
    h_rho : float
        Scale height for density (scalar, m).
    
    Returns
    -------
    dict : Dictionary containing:
        's', 't' : ndarray
            Grid vectors indicating coordinates for wind grids, with +s oriented
            in the down-wind direction, and +t oriented 90 degrees counterclockwise.
        'Sxy', 'Txy' : ndarray
            Grids containing the s and t coordinates for geographic grid nodes.
        'h_wind' : ndarray
            Wind grid for topography (matrix, nS x nT, m).
        'k_s', 'k_t' : ndarray
            Grid vectors indicating wavenumber coordinates (rad/m).
        'h_hat' : ndarray
            Grid with Fourier coefficients for topography (complex matrix).
        'k_z' : ndarray
            Grid of vertical wavenumbers (complex matrix).
    """
    # Transform topography to wind coordinates (s,t,z is right handed)
    Sxy, Txy, s, t, Xst, Yst = wind_grid(x, y, azimuth)
    
    # Interpolate h_grid onto wind grid
    # Note: RegularGridInterpolator uses (y, x) order for meshgrid format
    interp_func = RegularGridInterpolator(
        (y, x), 
        h_grid, 
        method='linear', 
        bounds_error=False, 
        fill_value=0
    )
    
    # Create points for interpolation (flatten and stack)
    points = np.column_stack([Yst.ravel(), Xst.ravel()])
    h_wind = interp_func(points).reshape(Xst.shape)
    
    # Parameters for wind grid
    d_s = s[1] - s[0]
    d_t = t[1] - t[0]
    n_s, n_t = h_wind.shape
    
    # Add zero padding around topography
    # Recommendation: n_s_pad = 2*n_s, n_t_pad = n_t
    n_s_pad = 2 * n_s
    n_t_pad = n_t
    
    # Calculate wavenumber grids for topography (rad/m)
    # Wavenumbers for s direction (wind direction)
    i_k_s_most_neg = int(np.ceil(n_s_pad / 2)) + 1
    k_s = np.arange(n_s_pad) / n_s_pad
    k_s[i_k_s_most_neg:n_s_pad] = k_s[i_k_s_most_neg:n_s_pad] - 1
    k_s = 2 * np.pi * k_s / d_s
    d_k_s = k_s[1] - k_s[0]
    
    # Wavenumbers for t direction
    i_k_t_most_neg = int(np.ceil(n_t_pad / 2)) + 1
    k_t = np.arange(n_t_pad) / n_t_pad
    k_t[i_k_t_most_neg:n_t_pad] = k_t[i_k_t_most_neg:n_t_pad] - 1
    k_t = 2 * np.pi * k_t / d_t
    
    # Calculate Fourier transform of topography
    h_hat = np.fft.fft2(h_wind, s=(n_s_pad, n_t_pad))
    
    # Calculate denominator for k_z equation
    # This step avoids singularities where abs(U*k_s) == abs(f_c)
    # Method from Queney, 1947, p. 46-48
    # h_hat((abs(denominator) < (U*d_k_s/2)^2) = 0
    # denominator(denominator == 0) = eps
    
    # denominator is n_s_pad-length column vector, expanded implicitly to match h_hat
    k_s_col = k_s.reshape(-1, 1)  # Column vector (n_s_pad, 1)
    denominator = (U * k_s_col)**2 - f_c**2
    
    # Find indices where denominator is too small
    # Expand to match h_hat shape for broadcasting
    i_zero = np.abs(denominator) < (U * d_k_s / 2)**2
    # Broadcast to h_hat shape
    i_zero_broadcast = np.broadcast_to(i_zero, h_hat.shape)
    h_hat[i_zero_broadcast] = 0
    
    # Avoid division by zero
    denominator[denominator == 0] = np.finfo(float).eps
    
    # Calculate k_z, vertical wave number grid
    # k_z^2 = (k_s^2 + k_t^2) * ((NM^2 - (U*k_s)^2) / denominator) - 1/(4*h_rho^2)
    k_s_sq = k_s_col**2
    k_t_sq = k_t**2
    
    k_z_sq = (k_s_sq + k_t_sq) * ((NM**2 - (U * k_s_col)**2) / denominator) - \
             1.0 / (4 * h_rho**2)
    
    # Ensure k_z_sq is complex to handle negative values (evanescent waves)
    k_z_sq = k_z_sq.astype(np.complex128)
    k_z = np.sqrt(k_z_sq)
    
    # Assign appropriate roots for sqrt(k_z_sq)
    # If k_z_sq > 0, then sqrt(k_z_sq) is real (propagating wave)
    # The sign of real k_z values is set to match the sign of its associated 
    # k_s value, ensuring that the wave propagates upward for both positive 
    # and negative wavenumbers for k_s.
    i_neg = (np.real(k_z_sq) > 0) & (k_s_col < 0)
    k_z[i_neg] = -k_z[i_neg]
    
    return {
        's': s,
        't': t,
        'Sxy': Sxy,
        'Txy': Txy,
        'h_wind': h_wind,
        'k_s': k_s,
        'k_t': k_t,
        'h_hat': h_hat,
        'k_z': k_z
    }


if __name__ == "__main__":
    # Simple test
    print("Testing fourier_solution module...")
    
    # Create a simple test grid
    x = np.linspace(-50000, 50000, 100)  # 100 km in x
    y = np.linspace(-50000, 50000, 100)  # 100 km in y
    X, Y = np.meshgrid(x, y)
    
    # Create a Gaussian mountain
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Parameters
    U = 10.0          # m/s
    azimuth = 90.0    # degrees (eastward)
    NM = 0.01         # rad/s
    f_c = 1e-4        # rad/s (typical Coriolis parameter)
    h_rho = 8000.0    # m (density scale height)
    
    # Run fourier_solution
    result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
    
    print(f"s vector length: {len(result['s'])}")
    print(f"t vector length: {len(result['t'])}")
    print(f"h_wind shape: {result['h_wind'].shape}")
    print(f"h_hat shape: {result['h_hat'].shape}")
    print(f"k_z shape: {result['k_z'].shape}")
    print("\nTest completed successfully!")
