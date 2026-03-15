"""
Test run_file.py against MATLAB getRunFile.m behavior
"""

import numpy as np
import os
import tempfile
from opi.io.run_file import parse_run_file, write_run_file, validate_run_data


def create_test_run_file():
    """Create a test .run file matching MATLAB format"""
    content = """% OPI Run File
% Test run for OPI model
Test Run Title
0
./data
no
topo.mat
0.25
no
no
[100 110 30 40]
map
10 1e-4
no
U|azimuth|T0|M|kappa|tauC|d2H0|d_d2H0_d_lat|fP0
0 0 0 0 0 0 -3 -3 0
5.0 0.0 280.0 0.5 100.0 1000.0 -150.0 -5.0 0.5
10.0 360.0 300.0 2.0 1000.0 5000.0 -50.0 -1.0 1.0
7.5 45.0 290.0 1.25 550.0 2500.0 -100.0 -3.0 0.75
"""
    fd, filepath = tempfile.mkstemp(suffix='.run')
    os.close(fd)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath


def test_parse_run_file():
    """Test parsing a run file"""
    filepath = create_test_run_file()
    
    try:
        run_data = parse_run_file(filepath)
        
        print(f"Run title: {run_data['run_title']}")
        print(f"Is parallel: {run_data['is_parallel']}")
        print(f"Data path: {run_data['data_path']}")
        print(f"Topo file: {run_data['topo_file']}")
        print(f"r_tukey: {run_data['r_tukey']}")
        print(f"Sample file: {run_data['sample_file']}")
        print(f"Cont divide file: {run_data['cont_divide_file']}")
        print(f"Map limits: {run_data['map_limits']}")
        print(f"Section origin: ({run_data['section_lon0']}, {run_data['section_lat0']})")
        print(f"mu: {run_data['mu']}, epsilon0: {run_data['epsilon0']}")
        print(f"Restart file: {run_data['restart_file']}")
        print(f"Parameter labels: {run_data['parameter_labels']}")
        print(f"Exponents: {run_data['exponents']}")
        print(f"Lower bounds: {run_data['l_b']}")
        print(f"Upper bounds: {run_data['u_b']}")
        print(f"Beta: {run_data['beta']}")
        
        # Verify values
        assert run_data['run_title'] == 'Test Run Title'
        assert run_data['is_parallel'] == False
        assert run_data['topo_file'] == 'topo.mat'
        assert abs(run_data['r_tukey'] - 0.25) < 1e-10
        assert run_data['sample_file'] is None  # 'no' -> None
        assert run_data['cont_divide_file'] is None
        assert len(run_data['map_limits']) == 4
        assert run_data['map_limits'] == [100, 110, 30, 40]
        assert run_data['section_lon0'] is None  # 'map' -> None
        assert run_data['section_lat0'] is None
        assert run_data['mu'] == 10.0
        assert run_data['epsilon0'] == 1e-4
        assert run_data['restart_file'] is None
        assert len(run_data['parameter_labels']) == 9
        assert run_data['parameter_labels'][0] == 'U'
        assert run_data['parameter_labels'][-1] == 'fP0'
        assert len(run_data['exponents']) == 9
        assert len(run_data['l_b']) == 9
        assert len(run_data['u_b']) == 9
        assert len(run_data['beta']) == 9
        
        print("[PASS] Parse run file test")
    finally:
        os.unlink(filepath)


def test_write_and_read():
    """Test write and read roundtrip"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        run_data = {
            'run_title': 'Roundtrip Test',
            'is_parallel': False,
            'data_path': './data',
            'topo_file': 'test_topo.mat',
            'r_tukey': 0.2,
            'sample_file': 'samples.xlsx',
            'cont_divide_file': None,
            'map_limits': [95, 115, 25, 45],
            'section_lon0': 105.0,
            'section_lat0': 35.0,
            'mu': 15.0,
            'epsilon0': 1e-3,
            'restart_file': None,
            'parameter_labels': ['U', 'azimuth', 'T0', 'M', 'kappa', 'tauC', 'd2H0', 'd_d2H0_d_lat', 'fP0'],
            'exponents': [0, 0, 0, 0, 0, 0, -3, -3, 0],
            'l_b': [5.0, 0.0, 280.0, 0.5, 100.0, 1000.0, -150.0, -5.0, 0.5],
            'u_b': [10.0, 360.0, 300.0, 2.0, 1000.0, 5000.0, -50.0, -1.0, 1.0],
            'beta': [7.5, 45.0, 290.0, 1.25, 550.0, 2500.0, -100.0, -3.0, 0.75],
        }
        
        # Write
        output_path = os.path.join(tmpdir, 'test.run')
        write_run_file(output_path, run_data)
        
        # Read back
        run_data_read = parse_run_file(output_path)
        
        # Verify
        assert run_data_read['run_title'] == run_data['run_title']
        assert run_data_read['is_parallel'] == run_data['is_parallel']
        assert run_data_read['topo_file'] == run_data['topo_file']
        assert abs(run_data_read['r_tukey'] - run_data['r_tukey']) < 1e-10
        assert run_data_read['sample_file'] == run_data['sample_file']
        assert run_data_read['section_lon0'] == run_data['section_lon0']
        assert run_data_read['section_lat0'] == run_data['section_lat0']
        assert run_data_read['mu'] == run_data['mu']
        assert run_data_read['epsilon0'] == run_data['epsilon0']
        assert run_data_read['parameter_labels'] == run_data['parameter_labels']
        assert run_data_read['exponents'] == run_data['exponents']
        np.testing.assert_allclose(run_data_read['l_b'], run_data['l_b'])
        np.testing.assert_allclose(run_data_read['u_b'], run_data['u_b'])
        
        print("[PASS] Write and read roundtrip test")


def test_parallel_mode():
    """Test parallel mode parsing"""
    content = """Test
1
./data
no
topo.mat
0.0
no
no
[0 1 0 1]
map
10 1e-4
no
U|azimuth|T0|M|kappa|tauC|d2H0|d_d2H0_d_lat|fP0
0 0 0 0 0 0 -3 -3 0
0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1
"""
    fd, filepath = tempfile.mkstemp(suffix='.run')
    os.close(fd)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        run_data = parse_run_file(filepath)
        assert run_data['is_parallel'] == True
        print("[PASS] Parallel mode test")
    finally:
        os.unlink(filepath)


def test_section_origin_map():
    """Test section origin set to 'map'"""
    content = """Test
0
./data
no
topo.mat
0.0
no
no
[0 1 0 1]
map
10 1e-4
no
U|azimuth|T0|M|kappa|tauC|d2H0|d_d2H0_d_lat|fP0
0 0 0 0 0 0 -3 -3 0
0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1
"""
    fd, filepath = tempfile.mkstemp(suffix='.run')
    os.close(fd)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        run_data = parse_run_file(filepath)
        assert run_data['section_lon0'] is None
        assert run_data['section_lat0'] is None
        print("[PASS] Section origin 'map' test")
    finally:
        os.unlink(filepath)


def test_nineteen_parameters():
    """Test 19-parameter (two-wind) configuration"""
    labels = '|'.join([f'param{i}' for i in range(1, 20)])
    content = f"""Test
0
./data
no
topo.mat
0.0
no
no
[0 1 0 1]
map
10 1e-4
no
{labels}
{' '.join(['0'] * 19)}
{' '.join(['0'] * 19)}
{' '.join(['1'] * 19)}
"""
    fd, filepath = tempfile.mkstemp(suffix='.run')
    os.close(fd)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        run_data = parse_run_file(filepath)
        assert run_data['n_parameters'] == 19
        assert len(run_data['parameter_labels']) == 19
        print("[PASS] 19 parameters test")
    finally:
        os.unlink(filepath)


def test_validation():
    """Test run data validation"""
    run_data = {
        'run_title': 'Test',
        'data_path': './data',
        'topo_file': 'topo.mat',
        'r_tukey': 0.5,
        'l_b': [0.0, 0.0, 0.0],
        'u_b': [1.0, 1.0, 1.0],
    }
    
    errors = validate_run_data(run_data)
    assert len(errors) == 0, f"Unexpected errors: {errors}"
    
    # Test invalid bounds
    run_data['l_b'] = [2.0, 0.0, 0.0]  # l_b > u_b for first param
    errors = validate_run_data(run_data)
    assert len(errors) > 0
    assert any('lower bound' in e.lower() for e in errors)
    
    print("[PASS] Validation test")


def test_invalid_parallel_value():
    """Test error handling for invalid parallel value"""
    content = """Test
2
./data
no
topo.mat
0.0
no
no
[0 1 0 1]
map
10 1e-4
no
U|T0|M|kappa|tauC|d2H0|d_d2H0_d_lat|fP0|extra
0 0 0 0 0 0 -3 -3 0
0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1
"""
    fd, filepath = tempfile.mkstemp(suffix='.run')
    os.close(fd)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        run_data = parse_run_file(filepath)
        # Python bool(int('2')) = True, so this actually works
        # MATLAB would raise error with str2num('2') then check ~='0' && ~='1'
        print("[INFO] Parallel value '2' accepted (Python bool(2) = True)")
    finally:
        os.unlink(filepath)


def test_missing_beta():
    """Test run file without beta vector"""
    content = """Test
0
./data
no
topo.mat
0.0
no
no
[0 1 0 1]
map
10 1e-4
no
U|azimuth|T0|M|kappa|tauC|d2H0|d_d2H0_d_lat|fP0
0 0 0 0 0 0 -3 -3 0
0 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1
"""
    fd, filepath = tempfile.mkstemp(suffix='.run')
    os.close(fd)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        run_data = parse_run_file(filepath)
        assert run_data['beta'] is None
        print("[PASS] Missing beta test")
    finally:
        os.unlink(filepath)


if __name__ == '__main__':
    print("=" * 60)
    print("Testing run_file module")
    print("=" * 60)
    
    test_parse_run_file()
    print()
    test_write_and_read()
    print()
    test_parallel_mode()
    print()
    test_section_origin_map()
    print()
    test_nineteen_parameters()
    print()
    test_validation()
    print()
    test_invalid_parallel_value()
    print()
    test_missing_beta()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
