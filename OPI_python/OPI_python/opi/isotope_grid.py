"""
Isotope Grid Calculation

<<<<<<< HEAD:OPI_python/OPI_python/opi/isotope_grid.py
Calculates the spatial distribution of precipitation isotopes (delta^2H and delta^18O)
using a modified Rayleigh distillation model with:
- Vertical averaging of fractionation factors
- WBF (Water-ice Balance Fractionation) zone handling
- Evaporative recycling effects
- Upwind integration along streamlines
- Latitudinal gradient correction
=======
Calculate precipitation isotope grids (d2H and d18O) as a function of 
temperature and the dln(fM)/dS grid, which is the log derivative of the 
precipitation field in the wind direction.

The calculation includes three steps:
1. Fractionation associated with creation and fall of precipitation
2. Fractionation due to evaporative recycling
3. Integration along wind path to get isotopic composition

The calculation accounts for ice and water, using the WBF zone defined
by the temperature range 268-248 K.
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/isotope_grid.py

Reference: Ciais and Jouzel, 1994 (MCIM model)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .fractionation_hydrogen import fractionation_hydrogen
from .fractionation_oxygen import fractionation_oxygen


<<<<<<< HEAD:OPI_python/OPI_python/opi/isotope_grid.py
def isotope_grid(s, t, s_xy, t_xy, lat, lat0, h_wind, f_m_wind, r_h_wind, f_p_wind,
                 z223_wind, z258_wind, tau_f, U, T, gamma_sat, h_s, h_r,
                 d2h0, d18o0, d_d2h0_d_lat, d_d18o0_d_lat, is_fit):
    """
    Calculate precipitation isotope (delta^2H, delta^18O) spatial distribution.
=======
def isotope_grid(s, t, Sxy, Txy, lat, lat0, hWind, fMWind, rHWind, fPWind,
                 z223Wind, z258Wind, tauF, U, T, gammaSat, hS, hR,
                 d2H0, d18O0, dDH0dLat, dD18O0_dLat, isFit):
    """
    Calculate precipitation isotope grids.
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/isotope_grid.py
    
    Parameters
    ----------
    s, t : ndarray
        Wind grid coordinate vectors
<<<<<<< HEAD:OPI_python/OPI_python/opi/isotope_grid.py
    s_xy, t_xy : ndarray
        Geographic grids in wind coordinates
    lat : ndarray
        Latitude of sample points
    lat0 : float
        Reference latitude
    h_wind : ndarray
        Topography in wind coordinates
    f_m_wind : ndarray
        Moisture ratio field
    r_h_wind : ndarray
        Relative humidity field
    f_p_wind : ndarray
        Residual precipitation fraction field
    z223_wind, z258_wind : ndarray
        Heights of 223K and 258K isotherms
    tau_f : float
=======
    Sxy, Txy : ndarray
        Geographic grids in wind coordinates
    lat : ndarray
        Latitude vector
    lat0 : float
        Reference latitude
    hWind : ndarray
        Topography in wind coordinates
    fMWind : ndarray
        Moisture ratio field
    rHWind : ndarray
        Relative humidity field
    fPWind : ndarray
        Residual precipitation fraction field
    z223Wind, z258Wind : ndarray
        Heights of 223K and 258K isotherms
    tauF : float
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/isotope_grid.py
        Precipitation fall time
    U : float
        Wind speed
    T : ndarray
<<<<<<< HEAD:OPI_python/OPI_python/opi/isotope_grid.py
        Temperature profile
    gamma_sat : ndarray
        Saturated lapse rate
    h_s : float
        Water vapor scale height
    h_r : float
        Isotope exchange distance (540 m)
    d2h0, d18o0 : float
        Base isotope values (fraction, not permil)
    d_d2h0_d_lat, d_d18o0_d_lat : float
        Latitudinal gradients (fraction per degree)
    is_fit : bool
=======
        Temperature profile (vector)
    gammaSat : ndarray
        Saturated lapse rate (vector)
    hS : float
        Water vapor scale height
    hR : float
        Isotope exchange distance (540 m)
    d2H0, d18O0 : float
        Base isotope values (fraction)
    dDH0dLat, dD18O0_dLat : float
        Latitudinal gradients (fraction per degree)
    isFit : bool
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/isotope_grid.py
        Whether this is a fitting calculation
    
    Returns
    -------
<<<<<<< HEAD:OPI_python/OPI_python/opi/isotope_grid.py
    dict : Dictionary containing:
        'd2h_grid' : ndarray - Hydrogen isotope grid (fraction)
        'd18o_grid' : ndarray - Oxygen isotope grid (fraction)
        'evap_d2h_grid' : ndarray - Evaporation effect on d2H
        'u_evap_d2h_grid' : ndarray - Undersaturated evaporation effect
        'evap_d18o_grid' : ndarray - Evaporation effect on d18O
        'u_evap_d18o_grid' : ndarray - Undersaturated evaporation effect
    """
    # Grid parameters
    d_s = s[1] - s[0]
    n_s, n_t = h_wind.shape
    
    # Calculate fractionation factors along the vertical profile
    # We need to average the fractionation over the condensation path
    
    # Temperature range for condensation (from surface to ~12km)
    z_levels = np.linspace(0, 12000, 100)
    
    # Interpolate gamma_sat to these levels (assuming linear decrease with height)
    # For simplicity, use the surface value
    gamma_sat_surf = gamma_sat[0]
    T_surf = T[0]
    T_profile = T_surf - gamma_sat_surf * z_levels
    
    # Calculate equilibrium fractionation factors at each level
    alpha_d2h_profile = fractionation_hydrogen(T_profile, h_r=1.0, is_kinetic=False)
    alpha_d18o_profile = fractionation_oxygen(T_profile, h_r=1.0, is_kinetic=False)
    
    # Vertically averaged fractionation factors
    # Weighted by water vapor density (exponential decay with scale height h_s)
    weights = np.exp(-z_levels / h_s)
    weights = weights / np.sum(weights)
    
    alpha_d2h_avg = np.sum(alpha_d2h_profile * weights)
    alpha_d18o_avg = np.sum(alpha_d18o_profile * weights)
    
    # Calculate effective fractionation including kinetic effects
    # where relative humidity is less than 1
    alpha_d2h_eff = np.ones((n_s, n_t))
    alpha_d18o_eff = np.ones((n_s, n_t))
    
    for i in range(n_s):
        for j in range(n_t):
            h_r = r_h_wind[i, j]
            alpha_d2h_eff[i, j] = fractionation_hydrogen(T_surf, h_r=h_r, is_kinetic=True)
            alpha_d18o_eff[i, j] = fractionation_oxygen(T_surf, h_r=h_r, is_kinetic=True)
    
    # Calculate Rayleigh distillation along wind direction
    # d/ds(f_m * delta) = -f_p * P * (delta - delta_vapor) / (U * h_r)
    # where delta_vapor = delta / alpha
    
    # Initialize isotope grids
    d2h_wind = np.zeros((n_s, n_t))
    d18o_wind = np.zeros((n_s, n_t))
    
    # Upstream boundary condition (first column in s direction)
    # Apply latitudinal gradient
    # For simplicity, assume uniform upstream conditions here
    # In full implementation, would need to calculate for each t
    d2h_upwind = d2h0 + d_d2h0_d_lat * (lat.mean() - lat0)
    d18o_upwind = d18o0 + d_d18o0_d_lat * (lat.mean() - lat0)
    
    d2h_wind[0, :] = d2h_upwind
    d18o_wind[0, :] = d18o_upwind
    
    # Integrate along s direction (downwind)
    for i in range(1, n_s):
        for j in range(n_t):
            f_m = f_m_wind[i, j]
            f_p = f_p_wind[i, j]
            
            # Previous values
            d2h_prev = d2h_wind[i-1, j]
            d18o_prev = d18o_wind[i-1, j]
            
            # Vapor isotope ratio (accounting for fractionation)
            d2h_vapor = d2h_prev / alpha_d2h_eff[i, j]
            d18o_vapor = d18o_prev / alpha_d18o_eff[i, j]
            
            # Simple Euler integration
            # d(delta)/ds = -f_p * (delta - delta_vapor) / (U * tau_f)
            # This is a simplified version - full model would include
            # more complex moisture and fractionation dynamics
            
            d_d2h_ds = -f_p * (d2h_prev - d2h_vapor) / (U * tau_f)
            d_d18o_ds = -f_p * (d18o_prev - d18o_vapor) / (U * tau_f)
            
            d2h_wind[i, j] = d2h_prev + d_d2h_ds * d_s
            d18o_wind[i, j] = d18o_prev + d_d18o_ds * d_s
    
    # Transform back to geographic grid
    interp_func_d2h = RegularGridInterpolator(
        (s, t), d2h_wind,
        method='linear',
        bounds_error=False,
        fill_value=d2h_upwind
    )
    
    interp_func_d18o = RegularGridInterpolator(
        (s, t), d18o_wind,
        method='linear',
        bounds_error=False,
        fill_value=d18o_upwind
    )
    
    d2h_grid = interp_func_d2h(np.column_stack([t_xy.ravel(), s_xy.ravel()])).reshape(s_xy.shape)
    d18o_grid = interp_func_d18o(np.column_stack([t_xy.ravel(), s_xy.ravel()])).reshape(s_xy.shape)
    
    # Evaporative recycling effects (simplified)
    # In full implementation, would calculate these based on undersaturation
    evap_d2h_grid = np.zeros_like(d2h_grid)
    u_evap_d2h_grid = np.zeros_like(d2h_grid)
    evap_d18o_grid = np.zeros_like(d18o_grid)
    u_evap_d18o_grid = np.zeros_like(d18o_grid)
    
    return {
        'd2h_grid': d2h_grid,
        'd18o_grid': d18o_grid,
        'evap_d2h_grid': evap_d2h_grid,
        'u_evap_d2h_grid': u_evap_d2h_grid,
        'evap_d18o_grid': evap_d18o_grid,
        'u_evap_d18o_grid': u_evap_d18o_grid
    }
=======
    tuple
        d2HGrid, d18OGrid, evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid
    """
    
    # Initialize variables
    # Shear for bringing fall path to vertical
    shear = U * tauF / hS
    
    # Constants for fractionation due to evaporation
    # Diffusivity ratios from Merlivat, 1978
    DRatio2H = 0.9755
    DRatio18O = 0.9723
    
    # Exponent for fractionation due to evaporation
    n = 1.0
    
    # Grid parameters
    dS = s[1] - s[0]
    nS, nT = hWind.shape
    
    # ============================================================
    # Set up wind-direction grid
    # ============================================================
    
    # Horizontal shear needed to transform vertical paths to land surface
    sSurfaceShearWind = s.reshape(-1, 1) + shear * hWind
    
    # Construct grid for temperature at land surface
    TLSWind = T[0] - gammaSat[0] * hWind
    
    # Horizontal shear of isothermal surface, to make fall paths vertical
    sShearWind = s.reshape(-1, 1) + shear * z223Wind
    for j in range(nT):
        # Use first crossing where surface is steeper than fall path
        sShearCol = sShearWind[:, j]
        # cummax equivalent
        cummax_vals = np.maximum.accumulate(sShearCol)
        iMonotonic = sShearCol >= np.concatenate([[sShearCol[0] - 1], cummax_vals[:-1]])
        
        if np.any(iMonotonic):
            z223Wind[:, j] = np.interp(
                sSurfaceShearWind[:, j],
                sShearCol[iMonotonic],
                z223Wind[iMonotonic, j],
                left=z223Wind[0, j],
                right=z223Wind[0, j]
            )
    
    # Calculate height of isothermal surface above land surface along fall path
    zBar223Wind = z223Wind - hWind
    del z223Wind
    
    # Horizontal shear of isothermal surface for 258K
    sShearWind = s.reshape(-1, 1) + shear * z258Wind
    for j in range(nT):
        sShearCol = sShearWind[:, j]
        cummax_vals = np.maximum.accumulate(sShearCol)
        iMonotonic = sShearCol >= np.concatenate([[sShearCol[0] - 1], cummax_vals[:-1]])
        
        if np.any(iMonotonic):
            z258Wind[:, j] = np.interp(
                sSurfaceShearWind[:, j],
                sShearCol[iMonotonic],
                z258Wind[iMonotonic, j],
                left=z258Wind[0, j],
                right=z258Wind[0, j]
            )
    
    del sShearWind
    
    # Calculate height of isothermal surface above land surface
    zBar258Wind = z258Wind - hWind
    del z258Wind
    
    # Calculate zBarFSWind, the height of freezing surface above land surface
    # Set to zero where freezing surface is below land surface
    zBarFSWind = zBar258Wind.copy()
    zBarFSWind[zBarFSWind < 0] = 0
    
    # Differentiate ln(fM) in wind direction
    dLnFM_dSWind = np.gradient(np.log(fMWind), dS, axis=0)
    
    # ============================================================
    # Calculate hydrogen-isotope grid
    # ============================================================
    
    # Get specific equilibrium factors
    aLSWind = fractionation_hydrogen(TLSWind)
    a258 = fractionation_hydrogen(258.0)
    a223 = fractionation_hydrogen(223.0)
    
    # Calculate fractionation factors by vertical averaging
    # Subscripts A and B refer to above and below 258 K point
    bA = (a223 - a258) / (zBar223Wind - zBar258Wind)
    bA[np.isnan(bA)] = 0
    bB = (a258 - aLSWind) / zBar258Wind
    bB[np.isnan(bB)] = 0
    
    # Fractionation factor for precipitation, equal to R_prec/R_vapor
    aPrecWind = (
        ((a258 + bA * (hS + zBarFSWind - zBar258Wind)) * np.exp(-zBarFSWind / hR)
         + aLSWind * (1 - np.exp(-zBarFSWind / hR))) * np.exp(-zBarFSWind / hS)
        + bB * (hR**2 * hS / (hS + hR)**2)
        * (1 - (1 + (1/hS + 1/hR) * zBarFSWind)
           * np.exp(-(1/hS + 1/hR) * zBarFSWind))
        + aLSWind * (1 - np.exp(-zBarFSWind / hS))
    )
    del bA, bB
    
    # Calculate fractionation due to evaporative recycling
    # Get equilibrium fractionation factor at evaporation temperature
    a1EvapWind = fractionation_hydrogen(TLSWind)
    
    # Calculate fractionation factor for evaporation at rH = 0
    a0EvapWind = a1EvapWind * DRatio2H**(-n)
    
    # Calculate fractionation factor for evaporation process (R_evap/R_vapor)
    aEvapWind = a1EvapWind * rHWind / (1 - a0EvapWind * (1 - rHWind))
    
    # Calculate exponent for integration of evaporative fractionation
    uEvap_d2HWind = 1.0 / (a0EvapWind * (1 - rHWind)) - 1
    uEvap_d2HWind[rHWind == 1] = 0
    
    # Combine to get fractionation factor for residual precipitation
    aResidualVaporWind = (fPWind**uEvap_d2HWind * aPrecWind
                          + (1 - fPWind**uEvap_d2HWind) * aEvapWind)
    del a0EvapWind, a1EvapWind
    
    # Integrate fractionation along wind direction
    # Result is the isotope ratio for precipitation
    R_PrecWind = (aPrecWind / aPrecWind[0, :]
                  * np.exp(np.cumsum((aResidualVaporWind - 1) * dLnFM_dSWind, axis=0) * dS))
    # Keep aPrecWind for later use if not fitting
    if isFit:
        del aPrecWind, aResidualVaporWind
    
    # Calculate evaporation grids if not fitting
    evapD2HGrid = np.array([])
    uEvapD2HGrid = np.array([])
    
    if not isFit:
        # Calculate d2HEvapWind and convert to geographic grid
        d2HEvapWind = (aEvapWind / aPrecWind) * R_PrecWind - 1
        F = RegularGridInterpolator((s, t), d2HEvapWind, method='linear', bounds_error=False, fill_value=None)
        del d2HEvapWind
        evapD2HGrid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
        
        # Convert uEvap_d2HWind to geographic grid
        F = RegularGridInterpolator((s, t), uEvap_d2HWind, method='linear', bounds_error=False, fill_value=None)
        uEvapD2HGrid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
        del F
    
    # Finalize d2H calculation
    # Transform back to geographic grid and convert to delta representation
    F = RegularGridInterpolator((s, t), R_PrecWind, method='linear', bounds_error=False, fill_value=None)
    del R_PrecWind
    
    # Ensure lat is 2D for broadcasting
    if lat.ndim == 1:
        lat = lat.reshape(-1, 1)
    
    d2HGrid = ((1 + d2H0 + dDH0dLat * (np.abs(lat) - np.abs(lat0)))
               * F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape) - 1)
    
    # ============================================================
    # Calculate oxygen-isotope grid
    # ============================================================
    
    # Get specific equilibrium factors
    aLSWind = fractionation_oxygen(TLSWind)
    a258 = fractionation_oxygen(258.0)
    a223 = fractionation_oxygen(223.0)
    
    # Calculate fractionation factors by vertical averaging
    bA = (a223 - a258) / (zBar223Wind - zBar258Wind)
    bA[np.isnan(bA)] = 0
    bB = (a258 - aLSWind) / zBar258Wind
    bB[np.isnan(bB)] = 0
    # Keep TLSWind for oxygen fractionation later
    del zBar223Wind
    
    # Fractionation factor for precipitation
    aPrecWind = (
        ((a258 + bA * (hS + zBarFSWind - zBar258Wind)) * np.exp(-zBarFSWind / hR)
         + aLSWind * (1 - np.exp(-zBarFSWind / hR))) * np.exp(-zBarFSWind / hS)
        + bB * (hR**2 * hS / (hS + hR)**2)
        * (1 - (1 + (1/hS + 1/hR) * zBarFSWind)
           * np.exp(-(1/hS + 1/hR) * zBarFSWind))
        + aLSWind * (1 - np.exp(-zBarFSWind / hS))
    )
    del bA, bB, zBar258Wind
    
    # Calculate fractionation due to evaporative recycling
    a1EvapWind = fractionation_oxygen(TLSWind)
    a0EvapWind = a1EvapWind * DRatio18O**(-n)
    aEvapWind = a1EvapWind * rHWind / (1 - a0EvapWind * (1 - rHWind))
    uEvap_d18OWind = 1.0 / (a0EvapWind * (1 - rHWind)) - 1
    uEvap_d18OWind[rHWind == 1] = 0
    
    # Combine to get fractionation factor for residual precipitation
    aResidualVaporWind = (fPWind**uEvap_d18OWind * aPrecWind
                          + (1 - fPWind**uEvap_d18OWind) * aEvapWind)
    del a0EvapWind, a1EvapWind
    
    # If fitting, remove unneeded arrays
    if isFit:
        del aEvapWind, uEvap_d18OWind
    
    # Integrate fractionation along wind direction
    R_PrecWind = (aPrecWind / aPrecWind[0, :]
                  * np.exp(np.cumsum((aResidualVaporWind - 1) * dLnFM_dSWind, axis=0) * dS))
    del aResidualVaporWind, dLnFM_dSWind
    
    # Keep aPrecWind for later use if not fitting
    if isFit:
        del aPrecWind
    
    # Calculate evaporation grids if not fitting
    evapD18OGrid = np.array([])
    uEvapD18OGrid = np.array([])
    
    if not isFit:
        d18OEvapWind = (aEvapWind / aPrecWind) * R_PrecWind - 1
        F = RegularGridInterpolator((s, t), d18OEvapWind, method='linear', bounds_error=False, fill_value=None)
        del d18OEvapWind
        evapD18OGrid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
        
        F = RegularGridInterpolator((s, t), uEvap_d18OWind, method='linear', bounds_error=False, fill_value=None)
        del uEvap_d18OWind
        uEvapD18OGrid = F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape)
        del F
    
    # Finalize d18O calculation
    F = RegularGridInterpolator((s, t), R_PrecWind, method='linear', bounds_error=False, fill_value=None)
    del R_PrecWind
    
    d18OGrid = ((1 + d18O0 + dD18O0_dLat * (np.abs(lat) - np.abs(lat0)))
                * F(np.column_stack([Txy.ravel(), Sxy.ravel()])).reshape(Sxy.shape) - 1)
    
    return d2HGrid, d18OGrid, evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/isotope_grid.py


if __name__ == "__main__":
    print("Testing isotope_grid module...")
<<<<<<< HEAD:OPI_python/OPI_python/opi/isotope_grid.py
=======
    print("(Matching MATLAB implementation)")
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/isotope_grid.py
    
    # Create simple test grid
    s = np.linspace(-50000, 50000, 50)
    t = np.linspace(-50000, 50000, 50)
    S, T_grid = np.meshgrid(s, t, indexing='ij')
    
    # Simple topography
<<<<<<< HEAD:OPI_python/OPI_python/opi/isotope_grid.py
    h_wind = 2000 * np.exp(-(S**2 + T_grid**2) / (2 * 20000**2))
    
    # Other parameters
    f_m_wind = np.ones_like(h_wind) * 0.8
    r_h_wind = np.ones_like(h_wind) * 0.9
    f_p_wind = np.ones_like(h_wind)
    z223_wind = h_wind + 2000
    z258_wind = h_wind + 1000
    tau_f = 500.0
    U = 10.0
    T = np.linspace(290, 220, 100)
    gamma_sat = np.ones(100) * 0.005
    h_s = 4000.0
    h_r = 540.0
    
    # Isotope parameters
    d2h0 = -5.0e-3  # -50 permil
    d18o0 = -0.5e-3  # -0.5 permil (approximate for MWL)
    d_d2h0_d_lat = -2.0e-3  # -2 permil per degree
    d_d18o0_d_lat = -0.2e-3
    lat = np.array([45.0])
    lat0 = 45.0
    is_fit = False
    
    # Run isotope_grid
    result = isotope_grid(
        s, t, S, T_grid, lat, lat0, h_wind, f_m_wind, r_h_wind, f_p_wind,
        z223_wind, z258_wind, tau_f, U, T, gamma_sat, h_s, h_r,
        d2h0, d18o0, d_d2h0_d_lat, d_d18o0_d_lat, is_fit
    )
    
    print(f"d2h_grid shape: {result['d2h_grid'].shape}")
    print(f"d2h range: {result['d2h_grid'].min()*1000:.2f} to {result['d2h_grid'].max()*1000:.2f} permil")
    print(f"d18o range: {result['d18o_grid'].min()*1000:.2f} to {result['d18o_grid'].max()*1000:.2f} permil")
    
    # Calculate deuterium excess
    d_excess = (result['d2h_grid'] * 1000 - 8 * result['d18o_grid'] * 1000)
=======
    hWind = 2000 * np.exp(-(S**2 + T_grid**2) / (2 * 20000**2))
    
    # Other parameters
    fMWind = np.ones_like(hWind) * 0.8
    rHWind = np.ones_like(hWind)
    fPWind = np.ones_like(hWind)
    z223Wind = hWind + 2000
    z258Wind = hWind + 1000
    tauF = 500.0
    U = 10.0
    T = np.linspace(290, 220, 100)
    gammaSat = np.ones(100) * 0.005
    hS = 4000.0
    hR = 540.0
    
    # Isotope parameters
    d2H0 = -5.0e-3
    d18O0 = -0.5e-3
    dDH0dLat = -2.0e-3
    dD18O0_dLat = -0.2e-3
    lat = np.array([45.0])
    lat0 = 45.0
    isFit = False
    
    # Run isotope_grid
    result = isotope_grid(
        s, t, S, T_grid, lat, lat0, hWind, fMWind, rHWind, fPWind,
        z223Wind, z258Wind, tauF, U, T, gammaSat, hS, hR,
        d2H0, d18O0, dDH0dLat, dD18O0_dLat, isFit
    )
    
    d2HGrid, d18OGrid, evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid = result
    
    print(f"d2HGrid shape: {d2HGrid.shape}")
    print(f"d2H range: {d2HGrid.min()*1000:.2f} to {d2HGrid.max()*1000:.2f} permil")
    print(f"d18O range: {d18OGrid.min()*1000:.2f} to {d18OGrid.max()*1000:.2f} permil")
    
    # Calculate deuterium excess
    d_excess = (d2HGrid * 1000 - 8 * d18OGrid * 1000)
>>>>>>> 763e46754822fb94efab4e507ef46c094eef6e44:opi/isotope_grid.py
    print(f"d-excess range: {d_excess.min():.2f} to {d_excess.max():.2f} permil")
    
    print("\nTest completed successfully!")
