"""
Isotope Grid Calculation

Calculate precipitation isotope grids (d2H and d18O) as a function of 
temperature and the dln(fM)/dS grid, which is the log derivative of the 
precipitation field in the wind direction.

The calculation includes three steps:
1. Fractionation associated with creation and fall of precipitation
2. Fractionation due to evaporative recycling
3. Integration along wind path to get isotopic composition

The calculation accounts for ice and water, using the WBF zone defined
by the temperature range 268-248 K.

Reference: Ciais and Jouzel, 1994 (MCIM model)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .fractionation_hydrogen import fractionation_hydrogen
from .fractionation_oxygen import fractionation_oxygen


def isotope_grid(s, t, Sxy, Txy, lat, lat0, hWind, fMWind, rHWind, fPWind,
                 z223Wind, z258Wind, tauF, U, T, gammaSat, hS, hR,
                 d2H0, d18O0, dDH0dLat, dD18O0_dLat, isFit):
    """
    Calculate precipitation isotope grids.
    
    Parameters
    ----------
    s, t : ndarray
        Wind grid coordinate vectors
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
        Precipitation fall time
    U : float
        Wind speed
    T : ndarray
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
        Whether this is a fitting calculation
    
    Returns
    -------
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


if __name__ == "__main__":
    print("Testing isotope_grid module...")
    print("(Matching MATLAB implementation)")
    
    # Create simple test grid
    s = np.linspace(-50000, 50000, 50)
    t = np.linspace(-50000, 50000, 50)
    S, T_grid = np.meshgrid(s, t, indexing='ij')
    
    # Simple topography
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
    print(f"d-excess range: {d_excess.min():.2f} to {d_excess.max():.2f} permil")
    
    print("\nTest completed successfully!")
