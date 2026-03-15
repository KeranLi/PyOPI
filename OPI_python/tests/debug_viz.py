from opi.opi_maps_two_winds import load_matlab_results, _prepare_data_dict
import numpy as np

results_file = '../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat'
results = load_matlab_results(results_file)
data = _prepare_data_dict(results, results_file)

print(f"lon shape: {data['lon'].shape}")
print(f"lat shape: {data['lat'].shape}")
print(f"p_grid shape: {data['p_grid'].shape}")

LON, LAT = np.meshgrid(data['lon'], data['lat'])
print(f"LON shape: {LON.shape}")
print(f"LAT shape: {LAT.shape}")
