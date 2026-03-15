"""
Two-Wind Calculation Function for OPI

This module implements the two-wind model calculation functionality
for modeling precipitation from two distinct moisture sources.
"""

import numpy as np
from datetime import datetime
from ..physics.calc_one_wind import calc_one_wind
from ..io.data_loader import get_input
from ..constants import HR, SD_RES_RATIO


def opi_calc_two_winds(run_file_path=None, solution_vector=None, verbose=True):
    """
    Performs calculations for the two-wind OPI model.
    
    This function calculates the precipitation and isotope distribution for 
    a mixture of two unique steady wind fields.
    
    Parameters:
    -----------
    run_file_path : str, optional
        Path to the run file containing model parameters and file paths
    solution_vector : array-like, optional
        19-parameter solution vector for the two-wind model:
        [U1, az1, T0_1, M1, kappa1, tau_c1, d2h0_1, d_d2h0_d_lat_1, f_p0_1,
         U2, az2, T0_2, M2, kappa2, tau_c2, d2h0_2, d_d2h0_d_lat_2, f_p0_2, frac2]
    verbose : bool, optional
        Whether to print detailed progress information
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'solution_params': solution parameter values
        - 'derived_params': derived parameter values
        - 'precipitation': combined precipitation distribution
        - 'isotope': combined isotope distribution
        - 'precipitation1', 'precipitation2': individual wind field precipitation
        - 'isotope1', 'isotope2': individual wind field isotopes
    """
    if verbose:
        print("OPI Calculation for Two Wind Fields")
        print("=" * 35)
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load or create input data
    if run_file_path:
        import os
        from ..io.run_file import parse_run_file
        
        run_dir = os.path.dirname(run_file_path) if os.path.dirname(run_file_path) else os.getcwd()
        
        # Parse run file to get data path and topo file
        try:
            run_params = parse_run_file(run_file_path)
            data_path = run_params['data_path']
            topo_file = run_params['topo_file']
            r_tukey = run_params.get('r_tukey', 0.0)
            sample_file = run_params.get('sample_file', None)
            use_synthetic = False
        except Exception as e:
            # Fallback to synthetic if parsing fails
            if verbose:
                print(f"Warning: Could not parse run file ({e}), using synthetic data")
            input_data = create_synthetic_input()
            use_synthetic = True
        
        if not use_synthetic:
            if verbose:
                print(f"Loading data from: {data_path}")
                print(f"  Topography file: {topo_file}")
            
            try:
                input_data = get_input(
                    data_path=data_path,
                    topo_file=topo_file,
                    r_tukey=r_tukey,
                    sample_file=sample_file,
                    sd_res_ratio=SD_RES_RATIO
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not load topography data: {e}")
    else:
        input_data = create_synthetic_input()
        if verbose:
            print("No run file specified, using synthetic data for demonstration")
    
    # Default solution vector if not provided
    # Format matches run file: [9 wind1 params, fraction, 9 wind2 params]
    if solution_vector is None:
        solution_vector = np.array([
            # First wind field (9 parameters)
            8.0,       # U1: Wind speed 1 (m/s)
            90.0,      # az1: Azimuth 1 (degrees from North)
            288.0,     # T0_1: Temperature 1 (K)
            0.3,       # M1: Mountain-height number 1
            15.0,      # kappa1: Eddy diffusion 1 (m^2/s)
            800.0,     # tau_c1: Condensation time 1 (s)
            -5.0e-3,   # d2h0_1: d2H base 1
            -2.0e-3,   # d_d2h0_d_lat_1: d2H lat gradient 1
            0.7,       # f_p0_1: Residual precip 1
            
            # Mixture parameter (1 parameter)
            0.5,       # frac2: Fraction of second wind field
            
            # Second wind field (9 parameters)
            12.0,      # U2: Wind speed 2 (m/s)
            270.0,     # az2: Azimuth 2 (degrees from North)
            292.0,     # T0_2: Temperature 2 (K)
            0.25,      # M2: Mountain-height number 2
            10.0,      # kappa2: Eddy diffusion 2 (m^2/s)
            1200.0,    # tau_c2: Condensation time 2 (s)
            -8.0e-3,   # d2h0_2: d2H base 2
            -1.5e-3,   # d_d2h0_d_lat_2: d2H lat gradient 2
            0.75,      # f_p0_2: Residual precip 2
        ])
    else:
        solution_vector = np.array(solution_vector)
    
    if verbose:
        print(f"Solution vector has {len(solution_vector)} parameters as expected for two-wind model")
    
    # Split solution vector into components
    # Run file format: [U1, Az1, T0_1, M1, kappa1, tau_c1, d2h0_1, d_d2h0_d_lat_1, f_p0_1,  (9 params)
    #                   fraction,  (1 param)
    #                   U2, Az2, T0_2, M2, kappa2, tau_c2, d2h0_2, d_d2h0_d_lat_2, f_p0_2]  (9 params)
    sol1 = solution_vector[0:9]    # First wind field parameters
    frac2 = solution_vector[9]     # Fraction of second wind field (index 9)
    sol2 = solution_vector[10:19]  # Second wind field parameters (indices 10-18)
    
    if verbose:
        print(f"\nStarting two-wind calculations with parameters:")
        print(f"  Wind 1: U={sol1[0]:.1f} m/s, Az={sol1[1]:.1f} deg")
        print(f"  Wind 2: U={sol2[0]:.1f} m/s, Az={sol2[1]:.1f} deg, Frac2={frac2:.2f}")
    
    # Prepare sample data (empty for pure calculation mode)
    sample_d2h = np.array([np.nan])
    sample_d18o = np.array([np.nan])
    
    from ..catchment import catchment_nodes
    ij_catch = []
    ptr_catch = [0]
    
    cov = np.array([[1e-6, 0], [0, 1e-6]])
    
    # Calculate results for each wind field
    try:
        # First wind field calculation
        if verbose:
            print("\n  Calculating Wind Field 1...")
        
        result1 = calc_one_wind(
            beta=sol1,
            f_c=input_data['f_c'],
            h_r=input_data['h_r'],
            x=input_data['x'],
            y=input_data['y'],
            lat=np.array([input_data['lat0']]),
            lat0=input_data['lat0'],
            h_grid=input_data['h_grid'],
            b_mwl_sample=input_data['b_mwl_sample'],
            ij_catch=ij_catch,
            ptr_catch=ptr_catch,
            sample_d2h=sample_d2h,
            sample_d18o=sample_d18o,
            cov=cov,
            n_parameters_free=9,
            is_fit=False
        )
        
        # Second wind field calculation
        if verbose:
            print("  Calculating Wind Field 2...")
        
        result2 = calc_one_wind(
            beta=sol2,
            f_c=input_data['f_c'],
            h_r=input_data['h_r'],
            x=input_data['x'],
            y=input_data['y'],
            lat=np.array([input_data['lat0']]),
            lat0=input_data['lat0'],
            h_grid=input_data['h_grid'],
            b_mwl_sample=input_data['b_mwl_sample'],
            ij_catch=ij_catch,
            ptr_catch=ptr_catch,
            sample_d2h=sample_d2h,
            sample_d18o=sample_d18o,
            cov=cov,
            n_parameters_free=9,
            is_fit=False
        )
        
        # Extract all results from calc_one_wind (25 return values)
        # Returns: chi_r2, nu, std_residuals, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
        #          rho_s0, h_s, rho0, h_rho, d18o0, d_d18o0_d_lat, tau_f, p_grid, f_m_grid, r_h_grid,
        #          evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid,
        #          d2h_grid, d18o_grid, i_wet, d2h_pred, d18o_pred
        precip1 = result1[16]  # p_grid
        f_m_grid1 = result1[17]  # moisture ratio
        r_h_grid1 = result1[18]  # relative humidity
        d18o_grid1 = result1[23]  # d18O
        iso1 = result1[22]  # d2h_grid
        # Additional variables for visualization
        T1 = result1[4]  # Temperature profile
        gamma_sat1 = result1[6]  # Saturation lapse rate
        
        precip2 = result2[16]
        f_m_grid2 = result2[17]
        r_h_grid2 = result2[18]
        d18o_grid2 = result2[23]
        iso2 = result2[22]
        T2 = result2[4]
        gamma_sat2 = result2[6]
        
        # Combine results according to the mixing fraction
        # Weighted combination of precipitation
        total_precip = frac2 * precip2 + (1 - frac2) * precip1
        
        # Weighted combination of isotopes (weighted by precipitation)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            total_isotope = (frac2 * precip2 * iso2 + (1 - frac2) * precip1 * iso1) / total_precip
            total_isotope = np.where(total_precip > 0, total_isotope, 0)
        
        # Calculate precipitation source fraction (fraction from wind 1)
        # This is Fig 09 in MATLAB
        with np.errstate(divide='ignore', invalid='ignore'):
            fraction_p_grid = np.where(total_precip > 0, 
                                       (1 - frac2) * precip1 / total_precip, 
                                       0.5)
        
        if verbose:
            print(f"\nTwo-wind calculations completed successfully")
            print(f"  Precipitation range: {np.nanmin(total_precip):.4f} to {np.nanmax(total_precip):.4f}")
            print(f"  Isotope range: {np.nanmin(total_isotope)*1000:.1f} to {np.nanmax(total_isotope)*1000:.1f} permil")
        
        # Prepare results
        solution_params = {
            'wind1_U': sol1[0], 'wind1_az': sol1[1], 'wind1_T0': sol1[2],
            'wind1_M': sol1[3], 'wind1_kappa': sol1[4], 'wind1_tau_c': sol1[5],
            'wind1_d2h0': sol1[6], 'wind1_d_d2h0_d_lat': sol1[7], 'wind1_f_p0': sol1[8],
            'wind2_U': sol2[0], 'wind2_az': sol2[1], 'wind2_T0': sol2[2],
            'wind2_M': sol2[3], 'wind2_kappa': sol2[4], 'wind2_tau_c': sol2[5],
            'wind2_d2h0': sol2[6], 'wind2_d_d2h0_d_lat': sol2[7], 'wind2_f_p0': sol2[8],
            'mix_frac2': frac2
        }
        
        # Helper to safely convert to float
        def safe_float(val):
            if np.isscalar(val):
                return float(val)
            elif hasattr(val, 'item'):
                return float(val.item()) if val.size == 1 else float(np.mean(val))
            else:
                return float(val)
        
        derived_params = {
            'precip_fraction': float(np.mean(total_precip > 0)),
            'mean_isotope': float(np.mean(total_isotope)) * 1000,
            'tau_f_1': safe_float(result1[15]),
            'tau_f_2': safe_float(result2[15])
        }
        
        # Calculate surface temperature grids (for visualization)
        T_grid_1 = T1[0] - gamma_sat1[0] * input_data['h_grid'] if len(T1) > 0 else None
        T_grid_2 = T2[0] - gamma_sat2[0] * input_data['h_grid'] if len(T2) > 0 else None
        
        return {
            'solution_params': solution_params,
            'derived_params': derived_params,
            'precipitation': total_precip,
            'isotope': total_isotope,
            'precipitation1': precip1,
            'isotope1': iso1,
            'precipitation2': precip2,
            'isotope2': iso2,
            'd18o_grid': frac2 * d18o_grid2 + (1 - frac2) * d18o_grid1,
            'd18o_grid_1': d18o_grid1,
            'd18o_grid_2': d18o_grid2,
            'f_m_grid_1': f_m_grid1,
            'f_m_grid_2': f_m_grid2,
            'r_h_grid_1': r_h_grid1,
            'r_h_grid_2': r_h_grid2,
            'T_grid_1': T_grid_1,  # Surface temperature
            'T_grid_2': T_grid_2,
            'fraction_p_grid': fraction_p_grid,  # Fig 09: precipitation source
            'x': input_data['x'],
            'y': input_data['y'],
            'h_grid': input_data['h_grid'],
            'misc': {
                'frac2': frac2,
                'grid_shape': input_data['h_grid'].shape
            }
        }
        
    except Exception as e:
        if verbose:
            print(f"Error in two-wind calculation: {e}")
            import traceback
            traceback.print_exc()
        
        # Return fallback with all expected keys
        h_shape = input_data['h_grid'].shape
        nan_grid = np.full(h_shape, np.nan)
        return {
            'solution_params': {},
            'derived_params': {},
            'precipitation': nan_grid,
            'isotope': nan_grid,
            'precipitation1': nan_grid,
            'isotope1': nan_grid,
            'precipitation2': nan_grid,
            'isotope2': nan_grid,
            'x': input_data['x'],
            'y': input_data['y'],
            'h_grid': input_data['h_grid'],
            'misc': {'error': str(e)}
        }


def create_synthetic_input():
    """Create synthetic input data."""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    lat0 = 45.0
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    return {
        'lon': x / 111320,
        'lat': y / 111320,
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'lon0': 0.0,
        'lat0': lat0,
        'sample_x': np.array([]),
        'sample_y': np.array([]),
        'sample_d2h': np.array([]),
        'sample_d18o': np.array([]),
        'sample_lc': np.array([]),
        'b_mwl_sample': np.array([9.47e-3, 8.03]),
        'cov': np.array([[1e-6, 0], [0, 1e-6]]),
        'f_c': f_c,
        'h_r': HR
    }


if __name__ == "__main__":
    result = opi_calc_two_winds(verbose=True)
    print("\n" + "=" * 50)
    print("TWO-WIND CALCULATION COMPLETE")
    print("=" * 50)
    print(f"Combined precipitation shape: {result['precipitation'].shape}")
    print(f"Precipitation range: {result['precipitation'].min():.6f} to {result['precipitation'].max():.6f}")
    print(f"Isotope range: {result['isotope'].min()*1000:.1f} to {result['isotope'].max()*1000:.1f} permil")
