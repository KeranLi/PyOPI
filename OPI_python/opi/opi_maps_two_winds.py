"""
OPI Maps Two Winds - Visualization for Two-Wind Model Results

Equivalent to MATLAB's opiMaps_TwoWinds.m

Generates 20 figures showing:
- Topography and sample locations
- Precipitation rates (combined and individual)
- Streamlines and cross-sections
- Isotope distributions (d2H, d18O)
- Moisture ratios and temperature fields
"""

import os
import numpy as np
from scipy.io import loadmat

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from .viz.maps import plot_result_maps, plot_two_wind_comparison
from .viz.streamlines import plot_3d_streamlines, calculate_wind_arrow
from .viz.cross_section import plot_cross_section
from .io.coordinates import xy2lonlat


def load_matlab_results(results_file):
    """
    Load MATLAB results file, supporting both v7 and v7.3 formats.
    
    Parameters
    ----------
    results_file : str
        Path to .mat file
    
    Returns
    -------
    dict
        Dictionary with loaded data
    """
    try:
        # Try standard loadmat first (for v7 and earlier)
        return loadmat(results_file)
    except NotImplementedError:
        # MATLAB v7.3 format (HDF5)
        if not HAS_H5PY:
            raise ImportError(
                "MATLAB v7.3 format detected but h5py is not installed. "
                "Install with: pip install h5py"
            )
        
        results = {}
        with h5py.File(results_file, 'r') as f:
            def read_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Convert to numpy array and transpose if needed
                    # MATLAB stores arrays column-major, Python is row-major
                    data = np.array(obj)
                    if data.ndim >= 2:
                        data = data.T
                    # Convert string arrays
                    if data.dtype.kind == 'U' or data.dtype.kind == 'S':
                        try:
                            data = ''.join(chr(c[0]) for c in obj[:])
                        except:
                            pass
                    results[name] = data
            
            f.visititems(read_dataset)
        
        return results
    except Exception as e:
        raise ValueError(f"Error loading results file: {e}")


def opi_maps_two_winds(results_file, output_dir=None, save_plots=True, 
                       show_plots=False, verbose=True):
    """
    Generate maps for two-wind OPI results.
    
    Parameters
    ----------
    results_file : str
        Path to opiCalc_TwoWinds_Results.mat file
    output_dir : str, optional
        Directory to save figures (default: same as results file)
    save_plots : bool, optional
        Whether to save plots to files (default: True)
    show_plots : bool, optional
        Whether to display plots interactively (default: False)
    verbose : bool, optional
        Whether to print progress information (default: True)
    
    Returns
    -------
    list
        List of generated figure file paths
    """
    if verbose:
        print("OPI Maps for Two Wind Fields")
        print("=" * 30)
        print(f"Loading results from: {results_file}")
    
    # Load results (supports both v7 and v7.3 MATLAB formats)
    try:
        results = load_matlab_results(results_file)
    except Exception as e:
        raise ValueError(f"Error loading results file: {e}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file) or os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Output directory: {output_dir}")
    
    # Prepare data dictionary
    data = _prepare_data_dict(results, results_file)
    
    # Generate maps
    generated_files = []
    
    if save_plots:
        if verbose:
            print("\nGenerating maps...")
        
        # Use the viz module functions
        files = plot_two_wind_comparison(data, save_dir=output_dir)
        generated_files.extend(files)
        
        # Generate additional maps (moisture ratio, relative humidity, d18O)
        files = _plot_additional_maps(data, output_dir)
        generated_files.extend(files)
        
        if verbose:
            print(f"Generated {len(files)} maps total")
    
    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()
    
    if verbose:
        print(f"\nTotal figures generated: {len(generated_files)}")
    
    return generated_files


def _plot_additional_maps(data, save_dir):
    """Generate additional maps for moisture ratio, RH, d18O, and temperature."""
    import matplotlib.pyplot as plt
    from .viz.colormaps import haxby
    
    figures = []
    lon = data['lon']
    lat = data['lat']
    
    # Create 2D coordinate grids for consistent plotting
    LON, LAT = np.meshgrid(lon, lat)
    
    TC2K = 273.15  # Celsius to Kelvin conversion
    
    # Figure: d18O combined
    if 'd18o_grid' in data and data['d18o_grid'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        d18o_permil = data['d18o_grid'] * 1000  # Convert to permil
        im = ax.pcolormesh(LON, LAT, d18o_permil, shading='auto', cmap='coolwarm')
        ax.set_title('Fig. 8. Predicted d18O (Combined)', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='d18O (permil)')
        if save_dir:
            path = f'{save_dir}/fig08_d18o_combined.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Figure: d18O State 1
    if 'd18o_grid_1' in data and data['d18o_grid_1'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        d18o_permil = data['d18o_grid_1'] * 1000
        im = ax.pcolormesh(LON, LAT, d18o_permil, shading='auto', cmap='coolwarm')
        ax.set_title('Fig. 14. Predicted d18O - State #1', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='d18O (permil)')
        if save_dir:
            path = f'{save_dir}/fig14_d18o_state1.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Figure: d18O State 2
    if 'd18o_grid_2' in data and data['d18o_grid_2'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        d18o_permil = data['d18o_grid_2'] * 1000
        im = ax.pcolormesh(LON, LAT, d18o_permil, shading='auto', cmap='coolwarm')
        ax.set_title('Fig. 15. Predicted d18O - State #2', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='d18O (permil)')
        if save_dir:
            path = f'{save_dir}/fig15_d18o_state2.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Figure: Moisture ratio State 1
    if 'f_m_grid_1' in data and data['f_m_grid_1'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.pcolormesh(LON, LAT, data['f_m_grid_1'], shading='auto', cmap='viridis')
        ax.set_title('Fig. 12. Moisture Ratio - State #1', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='Moisture Ratio')
        if save_dir:
            path = f'{save_dir}/fig12_moisture_state1.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Figure: Moisture ratio State 2
    if 'f_m_grid_2' in data and data['f_m_grid_2'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.pcolormesh(LON, LAT, data['f_m_grid_2'], shading='auto', cmap='viridis')
        ax.set_title('Fig. 13. Moisture Ratio - State #2', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='Moisture Ratio')
        if save_dir:
            path = f'{save_dir}/fig13_moisture_state2.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Figure: Relative Humidity State 1
    if 'r_h_grid_1' in data and data['r_h_grid_1'] is not None:
        rh = data['r_h_grid_1']
        # Check if it's a grid or scalar
        if hasattr(rh, 'shape') and len(rh.shape) >= 2 and min(rh.shape) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.pcolormesh(LON, LAT, rh, shading='auto', 
                              cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.set_title('Fig. 18. Surface Velocity Ratio - State #1', fontsize=14)
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            plt.colorbar(im, ax=ax, label='Velocity Ratio')
            if save_dir:
                path = f'{save_dir}/fig18_velocity_state1.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                figures.append(path)
            plt.close()
    
    # Figure: Relative Humidity State 2
    if 'r_h_grid_2' in data and data['r_h_grid_2'] is not None:
        rh = data['r_h_grid_2']
        if hasattr(rh, 'shape') and len(rh.shape) >= 2 and min(rh.shape) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.pcolormesh(LON, LAT, rh, shading='auto',
                              cmap='RdYlBu_r', vmin=0, vmax=1)
            ax.set_title('Fig. 19. Surface Velocity Ratio - State #2', fontsize=14)
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            plt.colorbar(im, ax=ax, label='Velocity Ratio')
            if save_dir:
                path = f'{save_dir}/fig19_velocity_state2.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                figures.append(path)
            plt.close()
    
    # Figure: Surface Temperature State 1
    if 'T_grid_1' in data and data['T_grid_1'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        T_celsius = data['T_grid_1'] - TC2K
        im = ax.pcolormesh(LON, LAT, T_celsius, shading='auto', cmap='coolwarm')
        ax.set_title('Fig. 16. Surface-Air Temperature - State #1', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        if save_dir:
            path = f'{save_dir}/fig16_temperature_state1.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Figure: Surface Temperature State 2
    if 'T_grid_2' in data and data['T_grid_2'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        T_celsius = data['T_grid_2'] - TC2K
        im = ax.pcolormesh(LON, LAT, T_celsius, shading='auto', cmap='coolwarm')
        ax.set_title('Fig. 17. Surface-Air Temperature - State #2', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        if save_dir:
            path = f'{save_dir}/fig17_temperature_state2.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Fig 09: Precipitation source
    if 'fraction_p_grid' in data and data['fraction_p_grid'] is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        frac = data['fraction_p_grid']
        im = ax.pcolormesh(LON, LAT, frac, shading='auto', cmap='coolwarm', vmin=0, vmax=1)
        ax.set_title('Fig. 9. Precipitation Source', fontsize=14)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        plt.colorbar(im, ax=ax, label='Fraction, Source #1 (blue arrow)')
        if save_dir:
            path = f'{save_dir}/fig09_precipitation_source.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            figures.append(path)
        plt.close()
    
    # Fig 03, 04: 3D Streamlines (simplified version)
    if 'h_grid' in data and data['h_grid'] is not None:
        h_grid = data['h_grid']
        # Check dimensions match
        if h_grid.shape == (len(lat), len(lon)):
            # Get section origin and wind parameters
            section_lon0 = data.get('section_lon0', np.mean(lon))
            section_lat0 = data.get('section_lat0', np.mean(lat))
            section_h0 = data.get('section_h0', 0)
            
            # Wind 1 3D streamline
            if 'azimuth_1' in data and 'wind1_U' in data:
                arrow_data = calculate_wind_arrow(
                    {'lon_min': lon.min(), 'lon_max': lon.max(), 
                     'lat_min': lat.min(), 'lat_max': lat.max()},
                    data['azimuth_1']
                )
                
                fig, ax = plot_3d_streamlines(
                    lon, lat, h_grid,
                    section_lon0, section_lat0, section_h0,
                    data['azimuth_1'], data['wind1_U'],
                    arrow_data['lon_start'], arrow_data['lat_start'],
                    arrow_data['lon_offset'], arrow_data['lat_offset'],
                    title='Fig. 3. Streamlines for Precipitation State #1',
                    save_path=f'{save_dir}/fig03_streamlines_state1.png' if save_dir else None
                )
                figures.append(f'{save_dir}/fig03_streamlines_state1.png')
                plt.close(fig)
            
            # Wind 2 3D streamline
            if 'azimuth_2' in data and 'wind2_U' in data:
                arrow_data = calculate_wind_arrow(
                    {'lon_min': lon.min(), 'lon_max': lon.max(),
                     'lat_min': lat.min(), 'lat_max': lat.max()},
                    data['azimuth_2']
                )
                
                fig, ax = plot_3d_streamlines(
                    lon, lat, h_grid,
                    section_lon0, section_lat0, section_h0,
                    data['azimuth_2'], data['wind2_U'],
                    arrow_data['lon_start'], arrow_data['lat_start'],
                    arrow_data['lon_offset'], arrow_data['lat_offset'],
                    title='Fig. 4. Streamlines for Precipitation State #2',
                    save_path=f'{save_dir}/fig04_streamlines_state2.png' if save_dir else None
                )
                figures.append(f'{save_dir}/fig04_streamlines_state2.png')
                plt.close(fig)
        else:
            print(f"Warning: h_grid shape {h_grid.shape} doesn't match coordinate grid, skipping Fig 03/04")
    
    # Fig 05, 06: Cross sections using wind_path with cloud water calculation
    if 'h_grid' in data and data['h_grid'] is not None and 'x' in data and 'y' in data:
        from .wind_path import wind_path
        from .coordinates import lonlat2xy, xy2lonlat
        from .physics.cloud_water import calculate_cloud_water_section
        from scipy.interpolate import RegularGridInterpolator
        
        x = data['x']
        y = data['y']
        lon0 = data.get('lon0', np.mean(lon))
        lat0 = data.get('lat0', np.mean(lat))
        section_lon0 = data.get('section_lon0', np.mean(lon))
        section_lat0 = data.get('section_lat0', np.mean(lat))
        
        # Convert section origin to x, y coordinates
        x_section0, y_section0 = lonlat2xy(section_lon0, section_lat0, lon0, lat0)
        
        # Get map limits for cross-section bounds
        map_limits = data.get('map_limits', None)
        if map_limits is not None and len(map_limits) >= 4:
            # Convert map limits from lon/lat to x/y
            x_min, y_min = lonlat2xy(map_limits[0], map_limits[2], lon0, lat0)
            x_max, y_max = lonlat2xy(map_limits[1], map_limits[3], lon0, lat0)
            xy_limits = np.array([x_min, x_max, y_min, y_max])
        else:
            xy_limits = None
        
        h_grid = data['h_grid']
        f_c = data.get('f_c', 0)  # Coriolis frequency
        h_r = data.get('h_r', 1000)  # Characteristic distance for isotopic exchange
        
        # Check grid size and downsample if necessary to avoid memory issues
        # Cloud water calculation requires FFT on extended grid (2x size)
        # For large grids, downsample to keep memory usage reasonable
        ny, nx = h_grid.shape
        max_grid_size = 1500  # Maximum grid size for FFT (empirical limit)
        
        downsample_factor = 1
        if max(nx, ny) > max_grid_size:
            downsample_factor = int(np.ceil(max(nx, ny) / max_grid_size))
            print(f"  Note: Downsampling grid by factor {downsample_factor} for cloud water calculation")
            
            # Downsample h_grid
            h_grid_ds = h_grid[::downsample_factor, ::downsample_factor]
            
            # Downsample x and y coordinates
            x_ds = x[::downsample_factor]
            y_ds = y[::downsample_factor]
        else:
            h_grid_ds = h_grid
            x_ds = x
            y_ds = y
        
        # Wind 1 cross-section
        if 'azimuth_1' in data and 'p_grid_1' in data and 'z_bar_1' in data:
            try:
                azimuth_1 = data['azimuth_1']
                U_1 = data.get('wind1_U', 10)
                NM_1 = data.get('nm_1', 0.01)
                kappa_1 = data.get('kappa_1', 1000)
                tau_c_1 = data.get('tau_c_1', 1000)
                tau_f_1 = data.get('tau_f_1', 500)
                h_rho_1 = data.get('h_rho_1', 8000)
                h_s_1 = data.get('h_s_1', 1000)
                rho_s0_1 = data.get('rho_s0_1', 0.01)
                gamma_ratio_1 = data.get('gamma_ratio_1', 0.8)
                
                # Get vertical profiles
                z_bar_1 = data['z_bar_1'].flatten()
                T_1 = data['T_1'].flatten()
                gamma_env_1 = data.get('gamma_env_1', 0.006)
                gamma_sat_1 = data.get('gamma_sat_1', 0.005)
                if hasattr(gamma_env_1, '__len__'):
                    gamma_env_1 = float(gamma_env_1[0]) if len(gamma_env_1) > 0 else 0.006
                if hasattr(gamma_sat_1, '__len__'):
                    gamma_sat_1 = float(gamma_sat_1[0]) if len(gamma_sat_1) > 0 else 0.005
                
                # Calculate wind path for cross-section
                x_path_1, y_path_1, s_path_1, s_limits_1 = wind_path(
                    x_section0, y_section0, azimuth_1, x, y, xy_limits
                )
                
                # Interpolate topography along path
                h_interp = RegularGridInterpolator((y, x), h_grid, method='linear', bounds_error=False, fill_value=0)
                h_l_path_1 = h_interp(np.column_stack([y_path_1, x_path_1]))
                
                # Calculate cloud water density along section
                h_l_max_1 = np.max(h_l_path_1) + 2 * h_s_1
                
                # Calculate buoyancy frequency NM from mountain-height number M
                h_max = np.max(h_grid)
                m_1 = data.get('m_1', 1.0)
                nm_1 = m_1 * U_1 / h_max if h_max > 0 else 0.01
                
                print(f"  Calculating cloud water for state 1... (this may take a moment)")
                z_rho_c_1, rho_c_1, z_c_mean_1, z_248_path_1, z_268_path_1 = calculate_cloud_water_section(
                    x_path=x_path_1,
                    y_path=y_path_1,
                    z_max=h_l_max_1,
                    x=x_ds, y=y_ds, h_grid=h_grid_ds,
                    U=U_1, azimuth=azimuth_1, NM=nm_1, f_c=f_c,
                    kappa=kappa_1, tau_c=tau_c_1, tau_f=tau_f_1, h_rho=h_rho_1,
                    z_bar=z_bar_1, T=T_1,
                    gamma_env=gamma_env_1, gamma_sat=gamma_sat_1,
                    gamma_ratio=gamma_ratio_1, rho_s0=rho_s0_1, h_v=h_s_1
                )
                
                # Interpolate precipitation along path
                p_interp = RegularGridInterpolator((lat, lon), data['p_grid_1'], method='linear', bounds_error=False, fill_value=0)
                lon_path_1, lat_path_1 = xy2lonlat(x_path_1, y_path_1, lon0, lat0)
                p_section_1 = p_interp(np.column_stack([lat_path_1, lon_path_1]))
                
                # Interpolate d2H along path
                d2h_interp = RegularGridInterpolator((lat, lon), data['d2h_grid_1'], method='linear', bounds_error=False, fill_value=0)
                d2h_section_1 = d2h_interp(np.column_stack([lat_path_1, lon_path_1]))
                
                # Calculate d2H0 (base precipitation) with latitudinal gradient
                d2h0_1 = data.get('d2h0_1', -100e-3)
                dd2h0_dlat_1 = data.get('dd2h0_dlat_1', 0)
                d2h0_section_1 = d2h0_1 + dd2h0_dlat_1 * (lat_path_1 - lat0)
                
                # Calculate precipitation lines
                n_prec_lines = 20
                s_prec_offset_1 = h_l_max_1 * U_1 * tau_f_1 / h_s_1
                s_prec_lines_1 = np.linspace(s_path_1[0] + s_prec_offset_1/2, 
                                              s_path_1[-1] - s_prec_offset_1/2, n_prec_lines)
                s_prec_lines_grid = np.array([s_prec_lines_1 + s_prec_offset_1, s_prec_lines_1])
                h_prec_lines_grid = np.array([np.zeros(n_prec_lines), h_l_max_1 * np.ones(n_prec_lines)])
                
                fig, axes = plot_cross_section(
                    s_path=s_path_1,
                    z_rho_c=z_rho_c_1,
                    rho_c=rho_c_1,
                    h_l_path=h_l_path_1,
                    z_248_path=z_248_path_1,
                    z_268_path=z_268_path_1,
                    s_l_section=None,
                    z_l_section=None,
                    s_prec_lines=s_prec_lines_grid,
                    h_prec_lines=h_prec_lines_grid,
                    p_section=p_section_1,
                    d2h_section=d2h_section_1,
                    d2h0_section=d2h0_section_1,
                    s_limits=s_limits_1,
                    h_l_max=h_l_max_1,
                    azimuth=azimuth_1,
                    title='Fig. 5. Cross Section for Precipitation State #1',
                    save_path=f'{save_dir}/fig05_cross_section_state1.png' if save_dir else None
                )
                figures.append(f'{save_dir}/fig05_cross_section_state1.png')
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not generate Fig 05: {e}")
                import traceback
                traceback.print_exc()
        
        # Wind 2 cross-section
        if 'azimuth_2' in data and 'p_grid_2' in data and 'z_bar_2' in data:
            try:
                azimuth_2 = data['azimuth_2']
                U_2 = data.get('wind2_U', 10)
                NM_2 = data.get('nm_2', 0.01)
                kappa_2 = data.get('kappa_2', 1000)
                tau_c_2 = data.get('tau_c_2', 1000)
                tau_f_2 = data.get('tau_f_2', 500)
                h_rho_2 = data.get('h_rho_2', 8000)
                h_s_2 = data.get('h_s_2', 1000)
                rho_s0_2 = data.get('rho_s0_2', 0.01)
                gamma_ratio_2 = data.get('gamma_ratio_2', 0.8)
                
                # Get vertical profiles
                z_bar_2 = data['z_bar_2'].flatten()
                T_2 = data['T_2'].flatten()
                gamma_env_2 = data.get('gamma_env_2', 0.006)
                gamma_sat_2 = data.get('gamma_sat_2', 0.005)
                if hasattr(gamma_env_2, '__len__'):
                    gamma_env_2 = float(gamma_env_2[0]) if len(gamma_env_2) > 0 else 0.006
                if hasattr(gamma_sat_2, '__len__'):
                    gamma_sat_2 = float(gamma_sat_2[0]) if len(gamma_sat_2) > 0 else 0.005
                
                # Calculate wind path for cross-section
                x_path_2, y_path_2, s_path_2, s_limits_2 = wind_path(
                    x_section0, y_section0, azimuth_2, x, y, xy_limits
                )
                
                # Interpolate topography along path
                h_interp = RegularGridInterpolator((y, x), h_grid, method='linear', bounds_error=False, fill_value=0)
                h_l_path_2 = h_interp(np.column_stack([y_path_2, x_path_2]))
                
                # Calculate cloud water density along section
                h_l_max_2 = np.max(h_l_path_2) + 2 * h_s_2
                
                # Calculate buoyancy frequency NM from mountain-height number M
                h_max = np.max(h_grid)
                m_2 = data.get('m_2', 1.0)
                nm_2 = m_2 * U_2 / h_max if h_max > 0 else 0.01
                
                print(f"  Calculating cloud water for state 2... (this may take a moment)")
                z_rho_c_2, rho_c_2, z_c_mean_2, z_248_path_2, z_268_path_2 = calculate_cloud_water_section(
                    x_path=x_path_2,
                    y_path=y_path_2,
                    z_max=h_l_max_2,
                    x=x_ds, y=y_ds, h_grid=h_grid_ds,
                    U=U_2, azimuth=azimuth_2, NM=nm_2, f_c=f_c,
                    kappa=kappa_2, tau_c=tau_c_2, tau_f=tau_f_2, h_rho=h_rho_2,
                    z_bar=z_bar_2, T=T_2,
                    gamma_env=gamma_env_2, gamma_sat=gamma_sat_2,
                    gamma_ratio=gamma_ratio_2, rho_s0=rho_s0_2, h_v=h_s_2
                )
                
                # Interpolate precipitation along path
                p_interp = RegularGridInterpolator((lat, lon), data['p_grid_2'], method='linear', bounds_error=False, fill_value=0)
                lon_path_2, lat_path_2 = xy2lonlat(x_path_2, y_path_2, lon0, lat0)
                p_section_2 = p_interp(np.column_stack([lat_path_2, lon_path_2]))
                
                # Interpolate d2H along path
                d2h_interp = RegularGridInterpolator((lat, lon), data['d2h_grid_2'], method='linear', bounds_error=False, fill_value=0)
                d2h_section_2 = d2h_interp(np.column_stack([lat_path_2, lon_path_2]))
                
                # Calculate d2H0 (base precipitation) with latitudinal gradient
                d2h0_2 = data.get('d2h0_2', -100e-3)
                dd2h0_dlat_2 = data.get('dd2h0_dlat_2', 0)
                d2h0_section_2 = d2h0_2 + dd2h0_dlat_2 * (lat_path_2 - lat0)
                
                # Calculate precipitation lines
                n_prec_lines = 20
                s_prec_offset_2 = h_l_max_2 * U_2 * tau_f_2 / h_s_2
                s_prec_lines_2 = np.linspace(s_path_2[0] + s_prec_offset_2/2, 
                                              s_path_2[-1] - s_prec_offset_2/2, n_prec_lines)
                s_prec_lines_grid = np.array([s_prec_lines_2 + s_prec_offset_2, s_prec_lines_2])
                h_prec_lines_grid = np.array([np.zeros(n_prec_lines), h_l_max_2 * np.ones(n_prec_lines)])
                
                fig, axes = plot_cross_section(
                    s_path=s_path_2,
                    z_rho_c=z_rho_c_2,
                    rho_c=rho_c_2,
                    h_l_path=h_l_path_2,
                    z_248_path=z_248_path_2,
                    z_268_path=z_268_path_2,
                    s_l_section=None,
                    z_l_section=None,
                    s_prec_lines=s_prec_lines_grid,
                    h_prec_lines=h_prec_lines_grid,
                    p_section=p_section_2,
                    d2h_section=d2h_section_2,
                    d2h0_section=d2h0_section_2,
                    s_limits=s_limits_2,
                    h_l_max=h_l_max_2,
                    azimuth=azimuth_2,
                    title='Fig. 6. Cross Section for Precipitation State #2',
                    save_path=f'{save_dir}/fig06_cross_section_state2.png' if save_dir else None
                )
                figures.append(f'{save_dir}/fig06_cross_section_state2.png')
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not generate Fig 06: {e}")
                import traceback
                traceback.print_exc()
    
    # Fig 20: Outliers (requires sample residuals)
    if 'std_residuals' in data and data['std_residuals'] is not None:
        std_residuals = data['std_residuals']
        sample_lon = data.get('sample_lon', None)
        sample_lat = data.get('sample_lat', None)
        
        if sample_lon is not None and sample_lat is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot all samples
            ax.plot(sample_lon, sample_lat, '.b', markersize=8, label='Samples')
            
            # Annotate outliers (std_residuals > 3)
            outliers = std_residuals > 3
            for i in np.where(outliers)[0]:
                ax.text(sample_lon[i], sample_lat[i], f'{std_residuals[i]:.1f}',
                       fontsize=10, ha='center')
            
            ax.set_title('Fig. 20. Location of Outliers', fontsize=14)
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('Latitude (deg)')
            ax.legend()
            
            if save_dir:
                path = f'{save_dir}/fig20_outliers.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                figures.append(path)
            plt.close()
    
    return figures


def _prepare_data_dict(results, results_file):
    """Prepare data dictionary from loaded MATLAB results."""
    data = {}
    
    # Extract grids - use keys that plot_two_wind_comparison expects
    if 'pGrid' in results:
        data['p_grid'] = results['pGrid']
    if 'pGrid_1' in results:
        data['p_grid_1'] = results['pGrid_1']
    if 'pGrid_2' in results:
        data['p_grid_2'] = results['pGrid_2']
    
    if 'd2HGrid' in results:
        data['d2h_grid'] = results['d2HGrid']
    if 'd2HGrid_1' in results:
        data['d2h_grid_1'] = results['d2HGrid_1']
    if 'd2HGrid_2' in results:
        data['d2h_grid_2'] = results['d2HGrid_2']
    
    if 'd18OGrid' in results:
        data['d18o_grid'] = results['d18OGrid']
    if 'd18OGrid_1' in results:
        data['d18o_grid_1'] = results['d18OGrid_1']
    if 'd18OGrid_2' in results:
        data['d18o_grid_2'] = results['d18OGrid_2']
    
    # Load x, y coordinates (meters) for cross-section calculations
    if 'x' in results:
        data['x'] = results['x'].flatten()
    if 'y' in results:
        data['y'] = results['y'].flatten()
    
    # Load map origin and limits
    if 'lon0' in results:
        data['lon0'] = float(results['lon0'].flat[0])
    if 'lat0' in results:
        data['lat0'] = float(results['lat0'].flat[0])
    if 'mapLimits' in results:
        data['map_limits'] = results['mapLimits'].flatten()
    
    # Additional grids for moisture ratio and relative humidity
    if 'fMGrid_1' in results:
        data['f_m_grid_1'] = results['fMGrid_1']
    if 'fMGrid_2' in results:
        data['f_m_grid_2'] = results['fMGrid_2']
    if 'rHGrid_1' in results:
        data['r_h_grid_1'] = results['rHGrid_1']
    if 'rHGrid_2' in results:
        data['r_h_grid_2'] = results['rHGrid_2']
    
    # Temperature grids
    if 'TGrid_1' in results:
        data['T_grid_1'] = results['TGrid_1']
    if 'TGrid_2' in results:
        data['T_grid_2'] = results['TGrid_2']
    
    # Precipitation source fraction (Fig 09)
    if 'fractionPGrid' in results:
        data['fraction_p_grid'] = results['fractionPGrid']
    
    # Wind parameters for 3D streamlines (Fig 03, 04)
    if 'beta' in results:
        beta = results['beta'].flatten()
        if len(beta) >= 19:
            # Two-wind model parameters
            data['wind1_U'] = beta[0]
            data['azimuth_1'] = beta[1]
            data['t0_1'] = beta[2]
            data['m_1'] = beta[3]
            data['kappa_1'] = beta[4]
            data['tau_c_1'] = beta[5]
            data['d2h0_1'] = beta[6]
            data['dd2h0_dlat_1'] = beta[7]
            data['fp0_1'] = beta[8]
            data['fraction'] = beta[9]
            data['wind2_U'] = beta[10]
            data['azimuth_2'] = beta[11]
            data['t0_2'] = beta[12]
            data['m_2'] = beta[13]
            data['kappa_2'] = beta[14]
            data['tau_c_2'] = beta[15]
            data['d2h0_2'] = beta[16]
            data['dd2h0_dlat_2'] = beta[17]
            data['fp0_2'] = beta[18]
    
    # Additional parameters for cross-sections
    if 'hS_1' in results:
        data['h_s_1'] = float(results['hS_1'].flat[0])
    if 'hS_2' in results:
        data['h_s_2'] = float(results['hS_2'].flat[0])
    if 'hRho_1' in results:
        data['h_rho_1'] = float(results['hRho_1'].flat[0])
    if 'hRho_2' in results:
        data['h_rho_2'] = float(results['hRho_2'].flat[0])
    if 'rhoS0_1' in results:
        data['rho_s0_1'] = float(results['rhoS0_1'].flat[0])
    if 'rhoS0_2' in results:
        data['rho_s0_2'] = float(results['rhoS0_2'].flat[0])
    if 'tauF_1' in results:
        data['tau_f_1'] = float(results['tauF_1'].flat[0])
    if 'tauF_2' in results:
        data['tau_f_2'] = float(results['tauF_2'].flat[0])
    if 'gammaRatio_1' in results:
        data['gamma_ratio_1'] = float(results['gammaRatio_1'].flat[0])
    if 'gammaRatio_2' in results:
        data['gamma_ratio_2'] = float(results['gammaRatio_2'].flat[0])
    if 'zBar_1' in results:
        data['z_bar_1'] = results['zBar_1'].flatten()
    if 'zBar_2' in results:
        data['z_bar_2'] = results['zBar_2'].flatten()
    if 'T_1' in results:
        data['T_1'] = results['T_1'].flatten()
    if 'T_2' in results:
        data['T_2'] = results['T_2'].flatten()
    if 'gammaEnv_1' in results:
        data['gamma_env_1'] = results['gammaEnv_1'].flatten()
    if 'gammaEnv_2' in results:
        data['gamma_env_2'] = results['gammaEnv_2'].flatten()
    if 'gammaSat_1' in results:
        data['gamma_sat_1'] = results['gammaSat_1'].flatten()
    if 'gammaSat_2' in results:
        data['gamma_sat_2'] = results['gammaSat_2'].flatten()
    if 'fC' in results:
        data['f_c'] = float(results['fC'].flat[0])
    if 'hR' in results:
        data['h_r'] = float(results['hR'].flat[0])
    
    # Sample data for outliers (Fig 20)
    if 'sampleLon' in results:
        data['sample_lon'] = results['sampleLon'].flatten()
    if 'sampleLat' in results:
        data['sample_lat'] = results['sampleLat'].flatten()
    if 'stdResiduals' in results:
        data['std_residuals'] = results['stdResiduals'].flatten()
    
    # Section origin
    if 'sectionLon0' in results:
        val = results['sectionLon0']
        data['section_lon0'] = float(val.flat[0]) if hasattr(val, 'flat') else float(val)
    if 'sectionLat0' in results:
        val = results['sectionLat0']
        data['section_lat0'] = float(val.flat[0]) if hasattr(val, 'flat') else float(val)
    
    # Get coordinate grids - first check results file, then load from data
    if 'lon' in results and 'lat' in results:
        # MATLAB stores lon/lat as row vectors, need to match grid dimensions
        lon = results['lon'].flatten()
        lat = results['lat'].flatten()
        # Check dimensions match pGrid
        if 'pGrid' in results:
            p_shape = results['pGrid'].shape
            # lon should match columns (p_shape[1]), lat should match rows (p_shape[0])
            if len(lon) == p_shape[1] and len(lat) == p_shape[0]:
                # Correct assignment
                data['lon'] = lon
                data['lat'] = lat
            elif len(lon) == p_shape[0] and len(lat) == p_shape[1]:
                # Swapped, need to swap them
                data['lon'] = lat  # lat has p_shape[1] elements
                data['lat'] = lon  # lon has p_shape[0] elements
            else:
                # Create synthetic coordinates
                data['lon'] = np.arange(p_shape[1])
                data['lat'] = np.arange(p_shape[0])
        else:
            data['lon'] = lon
            data['lat'] = lat
    elif 'x' in results:
        data['x'] = results['x'].flatten()
        data['y'] = results['y'].flatten() if 'y' in results else None
        data['lon'] = data['x']
        data['lat'] = data['y']
    else:
        # Try to load from run file
        lon, lat, h_grid = _load_coordinates_from_run(results, results_file)
        data['lon'] = lon
        data['lat'] = lat
        if h_grid is not None:
            data['h_grid'] = h_grid
    
    # Ensure lon and lat are set
    if 'lon' not in data or data['lon'] is None:
        if 'x' in data:
            data['lon'] = data['x']
    if 'lat' not in data or data['lat'] is None:
        if 'y' in data:
            data['lat'] = data['y']
    
    # Try to load h_grid from results or from topography file
    if 'hGrid' in results:
        h_grid = results['hGrid']
        # Ensure h_grid matches pGrid orientation (lat, lon)
        if 'pGrid' in results:
            p_shape = results['pGrid'].shape
            if h_grid.shape != p_shape:
                if h_grid.shape == (p_shape[1], p_shape[0]):
                    h_grid = h_grid.T
                    print(f"  Note: Transposed h_grid to match pGrid orientation")
        data['h_grid'] = h_grid
    elif 'h_grid' not in data:
        # Try to load from run file's data directory
        try:
            from .io.run_file import parse_run_file
            from .io.data_loader import get_input
            
            def matlab_str_to_py(val):
                if hasattr(val, 'shape') and len(val.shape) >= 2:
                    return ''.join(chr(c) for c in val[0] if c > 0)
                return str(val).strip("[]'\"")
            
            run_path = results.get('runPath', None)
            if run_path is not None:
                run_path = matlab_str_to_py(run_path)
            else:
                run_path = os.path.dirname(results_file)
            
            run_file = results.get('runFile', None)
            if run_file is not None:
                run_file = matlab_str_to_py(run_file)
            else:
                run_file = 'run.run'
            
            run_file_path = os.path.join(run_path, run_file)
            
            if os.path.exists(run_file_path):
                run_params = parse_run_file(run_file_path)
                input_data = get_input(
                    data_path=run_params['data_path'],
                    topo_file=run_params['topo_file'],
                    r_tukey=0.0,
                    sample_file=None
                )
                h_grid = input_data['h_grid']
                # Ensure h_grid matches pGrid orientation
                if 'pGrid' in results:
                    p_shape = results['pGrid'].shape
                    if h_grid.shape != p_shape:
                        if h_grid.shape == (p_shape[1], p_shape[0]):
                            h_grid = h_grid.T
                            print(f"  Note: Transposed h_grid to match pGrid orientation")
                data['h_grid'] = h_grid
                # Don't overwrite lon/lat - use the ones from MATLAB results file
        except Exception as e:
            print(f"Warning: Could not load topography: {e}")
    
    # Get parameters if available
    if 'beta' in results:
        data['beta'] = results['beta'].flatten()
    
    return data


def _load_coordinates_from_run(results, results_file):
    """
    Try to load coordinates and topography from run file or data directory.
    """
    try:
        from .io.run_file import parse_run_file
        from .io.data_loader import get_input
        
        # Helper to convert MATLAB string (uint16 array) to Python string
        def matlab_str_to_py(val):
            if hasattr(val, 'shape') and len(val.shape) >= 2:
                # uint16 array representing UTF-16 string
                return ''.join(chr(c) for c in val[0] if c > 0)
            return str(val).strip("[]'\"")
        
        # Get run file path from results
        run_path = results.get('runPath', None)
        if run_path is not None:
            run_path = matlab_str_to_py(run_path)
        else:
            run_path = os.path.dirname(results_file)
        
        run_file = results.get('runFile', None)
        if run_file is not None:
            run_file = matlab_str_to_py(run_file)
        else:
            run_file = 'run.run'
        
        run_file_path = os.path.join(run_path, run_file)
        
        if os.path.exists(run_file_path):
            run_params = parse_run_file(run_file_path)
            input_data = get_input(
                data_path=run_params['data_path'],
                topo_file=run_params['topo_file'],
                r_tukey=0.0,
                sample_file=None
            )
            return input_data['lon'], input_data['lat'], input_data.get('h_grid')
        else:
            print(f"Run file not found: {run_file_path}")
    except Exception as e:
        print(f"Warning: Could not load from run file: {e}")
    
    # Fallback: create synthetic coordinates based on grid shape
    if 'pGrid' in results:
        shape = results['pGrid'].shape
        lon = np.linspace(0, 1, shape[1])
        lat = np.linspace(0, 1, shape[0])
        return lon, lat, None
    
    return None, None, None


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python opi_maps_two_winds.py <results_file> [output_dir]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    files = opi_maps_two_winds(results_file, output_dir, verbose=True)
    print(f"Generated files: {files}")
