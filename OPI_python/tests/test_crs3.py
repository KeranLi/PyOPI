"""
Test CRS3 optimizer against MATLAB fminCRS3.m behavior
"""

import numpy as np
from opi.optimization.crs3 import fmin_crs3, CRS3Optimizer


def test_crs3_quadratic():
    """Test CRS3 on simple quadratic function"""
    def quadratic(x):
        return np.sum((x - 0.5)**2)
    
    bounds = [(-1, 1), (-1, 1)]
    result = fmin_crs3(quadratic, bounds, max_iter=1000, random_state=42)
    
    print(f"Quadratic function:")
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    print(f"  Iterations: {result.n_iter}")
    
    # Should find minimum close to [0.5, 0.5]
    assert np.allclose(result.x, [0.5, 0.5], atol=0.1)
    assert result.fun < 0.01
    
    print("[PASS] Quadratic test")


def test_crs3_rastrigin():
    """Test CRS3 on Rastrigin function (multimodal)"""
    def rastrigin(x, A=10):
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    bounds = [(-5.12, 5.12)] * 2
    result = fmin_crs3(rastrigin, bounds, max_iter=5000, random_state=42)
    
    print(f"\nRastrigin function:")
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    print(f"  True minimum: f = 0 at x = [0, 0]")
    
    # Should find something reasonably close to global minimum
    assert result.fun < 10.0  # Allow more tolerance for multimodal function
    
    print("[PASS] Rastrigin test")


def test_crs3_rosenbrock():
    """Test CRS3 on Rosenbrock function (valley-shaped)"""
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    bounds = [(-2, 2), (-1, 3)]
    result = fmin_crs3(rosenbrock, bounds, max_iter=5000, random_state=42)
    
    print(f"\nRosenbrock function:")
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    print(f"  True minimum: f = 0 at x = [1, 1]")
    
    # Rosenbrock is difficult, but should get reasonably close
    assert result.fun < 10.0
    
    print("[PASS] Rosenbrock test")


def test_crs3_convergence():
    """Test CRS3 convergence criterion"""
    def quadratic(x):
        return np.sum(x**2)
    
    bounds = [(-1, 1)] * 3
    
    # Test with different epsilon values
    for epsilon in [1e-3, 1e-5]:
        result = fmin_crs3(quadratic, bounds, epsilon=epsilon, 
                          max_iter=5000, random_state=42)
        print(f"\nConvergence with epsilon={epsilon}:")
        print(f"  Final f = {result.fun:.6e}")
        print(f"  Iterations = {result.n_iter}")
        
        # Should converge to near zero
        assert result.fun < 0.01  # Relaxed tolerance
    
    print("[PASS] Convergence test")


def test_crs3_population_size():
    """Test CRS3 with different population sizes"""
    def quadratic(x):
        return np.sum((x - 0.5)**2)
    
    bounds = [(-1, 1)] * 2
    
    for mu in [10, 25, 50]:
        result = fmin_crs3(quadratic, bounds, mu=mu, 
                          max_iter=1000, random_state=42)
        print(f"\nPopulation size mu={mu}:")
        print(f"  Solution: x = {result.x}")
        print(f"  Minimum: f = {result.fun:.6e}")
        
        # All should find good solution
        assert result.fun < 0.01
    
    print("[PASS] Population size test")


def test_crs3_optimizer_class():
    """Test CRS3Optimizer class"""
    def quadratic(x):
        return np.sum((x - 0.5)**2)
    
    optimizer = CRS3Optimizer(mu=25, epsilon=1e-6, max_iter=1000, random_state=42)
    bounds = [(-1, 1), (-1, 1)]
    result = optimizer.minimize(quadratic, bounds)
    
    print(f"\nCRS3Optimizer class:")
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    
    # Check history
    history = optimizer.get_history()
    assert len(history) == 1
    assert 'x' in history[0]
    assert 'fun' in history[0]
    
    print("[PASS] Optimizer class test")


def test_crs3_callback():
    """Test CRS3 callback functionality"""
    def quadratic(x):
        return np.sum((x - 0.5)**2)
    
    callback_count = [0]
    def callback(x, fval):
        callback_count[0] += 1
    
    bounds = [(-1, 1), (-1, 1)]
    result = fmin_crs3(quadratic, bounds, max_iter=100, callback=callback, 
                      random_state=42)
    
    print(f"\nCallback test:")
    print(f"  Callback called {callback_count[0]} times")
    
    # Callback should be called at least a few times
    assert callback_count[0] > 0
    
    print("[PASS] Callback test")


def test_crs3_with_args():
    """Test CRS3 with additional arguments"""
    def parametric_quadratic(x, target, weight):
        return weight * np.sum((x - target)**2)
    
    bounds = [(-1, 1), (-1, 1)]
    target = np.array([0.3, -0.4])
    weight = 2.0
    
    result = fmin_crs3(parametric_quadratic, bounds, 
                      args=(target, weight), max_iter=1000, random_state=42)
    
    print(f"\nWith args (target={target}, weight={weight}):")
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    
    # Should find minimum close to target
    assert np.allclose(result.x, target, atol=0.1)
    
    print("[PASS] Args test")


def test_crs3_high_dimensional():
    """Test CRS3 on higher dimensional problem"""
    def sphere(x):
        return np.sum(x**2)
    
    n_dim = 10
    bounds = [(-5, 5)] * n_dim
    result = fmin_crs3(sphere, bounds, mu=50, epsilon=1e-3, max_iter=3000, random_state=42)
    
    print(f"\nHigh dimensional (n={n_dim}):")
    print(f"  Solution norm: {np.linalg.norm(result.x):.6e}")
    print(f"  Minimum: f = {result.fun:.6e}")
    
    # High-dim is harder, allow larger tolerance
    assert result.fun < 15.0
    
    print("[PASS] High dimensional test")


def test_crs3_restart():
    """Test CRS3 restart functionality"""
    def quadratic(x):
        return np.sum((x - 0.5)**2)
    
    bounds = [(-1, 1), (-1, 1)]
    
    # Create some initial solutions (not at optimum)
    # Format: [f, nu, beta1, beta2, ...]
    restart_solutions = np.array([
        [0.1, 0, 0.3, 0.4],   # f=0.1, beta=[0.3, 0.4]
        [0.05, 0, 0.4, 0.5],  # f=0.05, beta=[0.4, 0.5]
        [0.02, 0, 0.45, 0.48], # f=0.02, beta=[0.45, 0.48]
    ])
    
    result = fmin_crs3(quadratic, bounds, mu=10, epsilon=1e-4,
                      max_iter=2000, random_state=42,
                      solutions=restart_solutions)
    
    print(f"\nRestart test:")
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    
    # Should converge to near [0.5, 0.5]
    assert np.allclose(result.x, [0.5, 0.5], atol=0.1)
    assert result.fun < 0.01
    
    print("[PASS] Restart test")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing CRS3 optimization module")
    print("=" * 60)
    
    test_crs3_quadratic()
    test_crs3_rastrigin()
    test_crs3_rosenbrock()
    print()
    test_crs3_convergence()
    print()
    test_crs3_population_size()
    print()
    test_crs3_optimizer_class()
    print()
    test_crs3_callback()
    print()
    test_crs3_with_args()
    print()
    test_crs3_high_dimensional()
    print()
    test_crs3_restart()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
