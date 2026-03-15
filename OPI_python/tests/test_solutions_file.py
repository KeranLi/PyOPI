"""
Test solutions_file.py against MATLAB getSolutions.m behavior
"""

import numpy as np
import os
import tempfile
from opi.io.solutions_file import (
    SolutionsFileWriter, 
    parse_solutions_file, 
    get_best_solution
)


def create_test_solutions_file():
    """Create a test solutions file matching MATLAB format"""
    lines = [
        "% Solution file for optimization using opiFit",
        "% Start time: 13-Mar-2026 10:30:15",
        "% Original run path:",
        "% /test/path",
        "% Run title:",
        "Test Run Title",
        "% Number of observations:",
        "5",
        "% Parameter labels:",
        "U|azimuth|T0|M|kappa|tauC|d2H0|d_d2H0_d_lat|fP0",
        "% Exponent for power of 10 factoring for plotted variables labeled above:",
        "0\t0\t0\t0\t0\t0\t-3\t-3\t0",
        "% Lower and upper constraints for parameter search:",
        "5.0\t0.0\t280.0\t0.5\t100.0\t1000.0\t-150.0\t-5.0\t0.5",
        "10.0\t360.0\t300.0\t2.0\t1000.0\t5000.0\t-50.0\t-1.0\t1.0",
        "% Size of search set",
        "100",
        "% Specified epsilon value for stopping criterion, epsilon0",
        "0.001",
        "%    n\t        chiR2\t    nu\t         beta01\t         beta02\t         beta03",
        "     1\t      2.500000\t    10\t      7.500000\t     45.000000\t    290.000000\t     1.250000\t    550.000000\t   2500.000000\t   -100.000000\t     -3.000000\t     0.750000",
        "     2\t      1.800000\t    10\t      8.200000\t     50.000000\t    288.000000\t     1.100000\t    600.000000\t   3000.000000\t    -90.000000\t     -2.500000\t     0.850000",
        "     3\t      1.200000\t    10\t      8.500000\t     55.000000\t    285.000000\t     1.300000\t    650.000000\t   3500.000000\t    -80.000000\t     -2.000000\t     0.900000",
        "% Finish time: 13-Mar-2026 10:35:20",
        "END",
    ]
    content = "\n".join(lines)
    fd, filepath = tempfile.mkstemp(suffix='.txt')
    os.close(fd)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath


def test_parse_solutions_file():
    """Test parsing a solutions file"""
    filepath = create_test_solutions_file()
    
    try:
        results = parse_solutions_file(os.path.dirname(filepath), 
                                        os.path.basename(filepath))
        
        print(f"Run title: {results['run_title']}")
        print(f"Number of samples: {results['n_samples']}")
        print(f"Number of parameters: {results['n_parameters']}")
        print(f"Number of solutions: {results['n_solutions']}")
        print(f"Axis labels: {results['axis_labels']}")
        print(f"Exponents: {results['exponents']}")
        print(f"Lower bounds: {results['lb_all']}")
        print(f"Upper bounds: {results['ub_all']}")
        
        # Verify header info
        assert results['run_title'] == 'Test Run Title'
        assert results['n_samples'] == 5
        assert results['n_parameters'] == 9
        assert results['n_solutions'] == 3
        assert len(results['axis_labels']) == 9
        assert results['axis_labels'][0] == 'U'
        assert results['axis_labels'][-1] == 'fP0'
        
        # Verify bounds
        assert len(results['lb_all']) == 9
        assert len(results['ub_all']) == 9
        assert results['lb_all'][0] == 5.0  # U lower bound
        assert results['ub_all'][0] == 10.0  # U upper bound
        
        # Verify solutions array shape (should be n_solutions x (2 + n_parameters))
        # Columns: chi_r2, nu, beta1, beta2, ..., betaN
        assert results['solutions_all'].shape == (3, 11)  # chi_r2 + nu + 9 params
        
        print("[PASS] Parse solutions file test")
    finally:
        os.unlink(filepath)


def test_best_solution():
    """Test finding the best solution"""
    filepath = create_test_solutions_file()
    
    try:
        results = parse_solutions_file(os.path.dirname(filepath), 
                                        os.path.basename(filepath))
        
        print(f"\nBest solution:")
        print(f"  chi_r2: {results['chi_r2_best']}")
        print(f"  nu: {results['nu_best']}")
        print(f"  beta: {results['beta_best']}")
        
        # Best solution should have minimum chi_r2
        assert results['chi_r2_best'] == 1.2  # From solution 3
        assert results['nu_best'] == 10
        assert len(results['beta_best']) == 9
        
        # First beta should be ~8.5 (from solution 3)
        assert abs(results['beta_best'][0] - 8.5) < 0.1
        
        print("[PASS] Best solution test")
    finally:
        os.unlink(filepath)


def test_solutions_file_writer():
    """Test writing a solutions file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SolutionsFileWriter()
        writer.initialize(
            run_path=tmpdir,
            run_title='Test Write Run',
            n_samples=3,
            parameter_labels=['U', 'azimuth', 'T0'],
            exponents=[0, 0, 0],
            lb=[5.0, 0.0, 280.0],
            ub=[10.0, 360.0, 300.0]
        )
        
        # Write some solutions
        writer.write_solution(1, 2.5, 5, [7.5, 45.0, 290.0])
        writer.write_solution(2, 1.8, 5, [8.0, 50.0, 288.0])
        writer.write_solution(3, 1.2, 5, [8.5, 55.0, 285.0])
        
        writer.close()
        
        # Read back and verify
        results = parse_solutions_file(tmpdir, 'opiFit_Solutions.txt')
        
        assert results['run_title'] == 'Test Write Run'
        assert results['n_samples'] == 3
        assert results['n_parameters'] == 3
        assert results['n_solutions'] == 3
        assert results['chi_r2_best'] == 1.2
        
        print("[PASS] Solutions file writer test")


def test_write_and_read_roundtrip():
    """Test write and read roundtrip"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write
        writer = SolutionsFileWriter()
        writer.initialize(
            run_path=tmpdir,
            run_title='Roundtrip Test',
            n_samples=10,
            parameter_labels=['U', 'azimuth', 'T0', 'M', 'kappa'],
            exponents=[0, 0, 0, 0, 0],
            lb=[5.0, 0.0, 280.0, 0.5, 100.0],
            ub=[10.0, 360.0, 300.0, 2.0, 1000.0]
        )
        
        # Write solutions with varying chi_r2
        for i in range(10):
            chi_r2 = 3.0 - i * 0.2  # Decreasing chi_r2
            writer.write_solution(i+1, chi_r2, 8, 
                                  [7.0 + i*0.1, 45.0 + i, 285.0 + i, 1.0, 500.0])
        
        writer.close()
        
        # Read back
        results = parse_solutions_file(tmpdir, 'opiFit_Solutions.txt')
        
        assert results['run_title'] == 'Roundtrip Test'
        assert results['n_solutions'] == 10
        # Best should be the last one (lowest chi_r2)
        assert abs(results['chi_r2_best'] - 1.2) < 0.01
        assert abs(results['beta_best'][0] - 7.9) < 0.01  # 7.0 + 9*0.1
        
        print("[PASS] Write and read roundtrip test")


def test_get_best_solution():
    """Test get_best_solution convenience function"""
    filepath = create_test_solutions_file()
    
    try:
        best = get_best_solution(os.path.dirname(filepath), 
                                  os.path.basename(filepath))
        
        print(f"\nBest solution (convenience function):")
        print(f"  chi_r2: {best['chi_r2']}")
        print(f"  nu: {best['nu']}")
        print(f"  beta shape: {best['beta'].shape}")
        
        assert best['chi_r2'] == 1.2
        assert best['nu'] == 10
        assert len(best['beta']) == 9
        assert best['run_title'] == 'Test Run Title'
        
        print("[PASS] Get best solution test")
    finally:
        os.unlink(filepath)


def test_empty_file_error():
    """Test error handling for empty/invalid file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'empty.txt')
        with open(filepath, 'w') as f:
            f.write("% Just a comment\n")
        
        try:
            parse_solutions_file(tmpdir, 'empty.txt')
            print("[FAIL] Should have raised error")
        except (ValueError, IndexError):
            print("[PASS] Empty file error handling")


def test_nonexistent_file():
    """Test error handling for non-existent file"""
    try:
        parse_solutions_file('/nonexistent', 'file.txt')
        print("[FAIL] Should have raised error")
    except FileNotFoundError:
        print("[PASS] Non-existent file error handling")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing solutions_file module")
    print("=" * 60)
    
    test_parse_solutions_file()
    print()
    test_best_solution()
    print()
    test_solutions_file_writer()
    print()
    test_write_and_read_roundtrip()
    print()
    test_get_best_solution()
    print()
    test_empty_file_error()
    print()
    test_nonexistent_file()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
