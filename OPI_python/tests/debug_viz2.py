from opi.opi_maps_two_winds import load_matlab_results
import numpy as np

results_file = '../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat'
results = load_matlab_results(results_file)

print("Checking what triggers h_grid loading:")
print(f"  'hGrid' in results: {'hGrid' in results}")

# Simulate _prepare_data_dict logic
lon = results['lon'].flatten()
lat = results['lat'].flatten()
print(f"\nAfter flatten: lon={lon.shape}, lat={lat.shape}")

p_shape = results['pGrid'].shape
print(f"p_shape: {p_shape}")

# Check condition
print(f"\nCondition check:")
print(f"  len(lon)={len(lon)} == p_shape[1]={p_shape[1]}: {len(lon) == p_shape[1]}")
print(f"  len(lat)={len(lat)} == p_shape[0]={p_shape[0]}: {len(lat) == p_shape[0]}")

if len(lon) == p_shape[1] and len(lat) == p_shape[0]:
    print("Would use: data['lon'] = lon, data['lat'] = lat")
elif len(lon) == p_shape[0] and len(lat) == p_shape[1]:
    print("Would use: data['lon'] = lat, data['lat'] = lon")
