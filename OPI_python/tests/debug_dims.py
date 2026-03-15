from opi.opi_maps_two_winds import load_matlab_results
import numpy as np

results = load_matlab_results('../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat')

print('Grid dimensions:')
print(f'  pGrid shape: {results["pGrid"].shape}')
print(f'  lon shape: {results["lon"].shape}')
print(f'  lat shape: {results["lat"].shape}')

lon = results['lon'].flatten()
lat = results['lat'].flatten()

print(f'\nFlattened:')
print(f'  lon: {lon.shape}, range: [{lon.min():.2f}, {lon.max():.2f}]')
print(f'  lat: {lat.shape}, range: [{lat.min():.2f}, {lat.max():.2f}]')
