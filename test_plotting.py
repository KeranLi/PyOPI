#!/usr/bin/env python3
"""
Test script to verify plotting functionality
"""

from opi import opi_calc_one_wind, opi_plots_one_wind
import numpy as np

def test_plotting():
    print("Testing plotting functionality...")
    
    # Get calculation result
    result = opi_calc_one_wind(verbose=False)
    print('Result keys:', list(result.keys()))
    
    # Check if it has the right structure
    if 'results' in result:
        print('Results sub-keys:', list(result['results'].keys()) if isinstance(result['results'], dict) else "Not a dict")
    
    # Try plotting
    try:
        plot_files = opi_plots_one_wind(result, verbose=True)
        print('Plotting successful!')
        print(f'Generated plot files: {plot_files}')
    except Exception as e:
        print(f'Plotting failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plotting()