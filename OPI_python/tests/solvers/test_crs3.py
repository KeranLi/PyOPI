"""
Tests for CRS3 optimizer.

This module contains tests for the Controlled Random Search with 3 Parents (CRS3)
optimization algorithm implementation.
"""

import numpy as np
import pytest
from opi.solvers import CRS3Optimizer, fmin_crs3


def sphere_function(x: np.ndarray) -> float:
    """Simple sphere function: f(x) = sum(x_i^2), minimum at origin."""
    return np.sum(x**2)


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x) = (1-x_0)^2 + 100*(x_1-x_0^2)^2
    Global minimum of 0 at (1, 1).
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def quadratic_function(x: np.ndarray) -> float:
    """Simple quadratic: f(x) = sum((x_i - i)^2), minimum at (0, 1, 2, ...)."""
    offset = np.arange(len(x))
    return np.sum((x - offset)**2)


class TestCRS3Initialization:
    """Tests for CRS3Optimizer initialization."""
    
    def test_crs3_initialization(self):
        """Test that CRS3Optimizer can be created with valid bounds."""
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        optimizer = CRS3Optimizer(bounds)
        
        assert optimizer.n_dim == 2
        assert np.array_equal(optimizer.lower, np.array([0.0, 0.0]))
        assert np.array_equal(optimizer.upper, np.array([10.0, 10.0]))
        assert optimizer.population_size == 10 * (2 + 1)  # Default: 10*(n+1)
        assert optimizer.max_iter == 10000
        assert optimizer.max_eval == 50000
        assert optimizer.tol == 1e-6
    
    def test_crs3_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        bounds = (np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))
        optimizer = CRS3Optimizer(
            bounds=bounds,
            population_size=50,
            max_iter=5000,
            max_eval=20000,
            tol=1e-8,
            random_state=42
        )
        
        assert optimizer.n_dim == 3
        assert optimizer.population_size == 50
        assert optimizer.max_iter == 5000
        assert optimizer.max_eval == 20000
        assert optimizer.tol == 1e-8
    
    def test_crs3_initialization_1d(self):
        """Test initialization with 1D bounds."""
        bounds = (np.array([0.0]), np.array([1.0]))
        optimizer = CRS3Optimizer(bounds)
        
        assert optimizer.n_dim == 1
        assert optimizer.population_size == 10 * (1 + 1)


class TestCRS3Optimization:
    """Tests for CRS3Optimizer minimization functionality."""
    
    def test_crs3_simple_function(self):
        """Test minimization of simple quadratic (sphere function)."""
        bounds = (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
        optimizer = CRS3Optimizer(bounds, random_state=42, tol=1e-5)
        
        result = optimizer.minimize(sphere_function, x0=np.array([3.0, 3.0]))
        
        assert result.success
        assert result.fun < 1e-3
        assert np.allclose(result.x, np.array([0.0, 0.0]), atol=0.1)
        assert result.n_iter > 0
        assert result.n_eval >= result.n_iter
    
    def test_crs3_rosenbrock(self):
        """Test optimization on Rosenbrock function."""
        bounds = (np.array([-2.0, -1.0]), np.array([2.0, 3.0]))
        optimizer = CRS3Optimizer(
            bounds, 
            random_state=42, 
            tol=1e-4,
            population_size=30
        )
        
        result = optimizer.minimize(rosenbrock_function, x0=np.array([0.0, 0.0]))
        
        # Rosenbrock is challenging, so we check reasonable convergence
        assert result.fun < 1.0  # Should find a good solution
        # Global minimum is at (1, 1)
        assert np.abs(result.x[0] - 1.0) < 0.5
        assert np.abs(result.x[1] - 1.0) < 0.5
        assert result.n_eval > 0
    
    def test_crs3_higher_dimensional(self):
        """Test optimization in higher dimensions."""
        n_dim = 5
        bounds = (np.zeros(n_dim), np.ones(n_dim) * 5)
        optimizer = CRS3Optimizer(bounds, random_state=42, tol=1e-4)
        
        result = optimizer.minimize(quadratic_function, x0=np.ones(n_dim))
        
        assert result.success
        assert result.fun < 0.1
        # Minimum is at (0, 1, 2, 3, 4)
        expected = np.arange(n_dim)
        assert np.allclose(result.x, expected, atol=0.2)


class TestCRS3Bounds:
    """Tests for CRS3Optimizer bounds handling."""
    
    def test_crs3_bounds_respected(self):
        """Test that solution respects bounds constraints."""
        bounds = (np.array([2.0, 2.0]), np.array([5.0, 5.0]))
        optimizer = CRS3Optimizer(bounds, random_state=42)
        
        # Sphere function has minimum at origin, but we constrain it to [2,5]
        result = optimizer.minimize(sphere_function, x0=np.array([3.0, 3.0]))
        
        # Solution should be within bounds
        assert np.all(result.x >= bounds[0])
        assert np.all(result.x <= bounds[1])
        # With bounds [2,5], minimum of sphere is at [2,2]
        assert np.allclose(result.x, np.array([2.0, 2.0]), atol=0.1)
    
    def test_crs3_bounds_with_initial_guess_outside(self):
        """Test behavior when initial guess is outside bounds."""
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        optimizer = CRS3Optimizer(bounds, random_state=42)
        
        # Initial guess outside bounds should be clipped
        result = optimizer.minimize(sphere_function, x0=np.array([-5.0, 15.0]))
        
        # Solution should still respect bounds
        assert np.all(result.x >= bounds[0])
        assert np.all(result.x <= bounds[1])


class TestCRS3ResultStructure:
    """Tests for CRS3Result structure and attributes."""
    
    def test_crs3_result_structure(self):
        """Verify result object has expected attributes."""
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        optimizer = CRS3Optimizer(bounds, random_state=42)
        
        result = optimizer.minimize(sphere_function, x0=np.array([5.0, 5.0]))
        
        # Check all expected attributes exist
        assert hasattr(result, 'x')
        assert hasattr(result, 'fun')
        assert hasattr(result, 'n_iter')
        assert hasattr(result, 'n_eval')
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')
        
        # Check types
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, (float, np.floating))
        assert isinstance(result.n_iter, (int, np.integer))
        assert isinstance(result.n_eval, (int, np.integer))
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        
        # Check consistency
        assert len(result.x) == 2
        assert result.n_iter >= 0
        assert result.n_eval >= 0
        assert result.fun == sphere_function(result.x)
    
    def test_crs3_result_with_max_eval_limit(self):
        """Test result structure when hitting max_eval limit."""
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        optimizer = CRS3Optimizer(bounds, max_eval=50, random_state=42)
        
        result = optimizer.minimize(sphere_function, x0=np.array([5.0, 5.0]))
        
        assert not result.success
        assert "evaluations" in result.message.lower() or "maximum" in result.message.lower()
        assert result.n_eval <= 60  # Allow some tolerance for initialization
    
    def test_crs3_result_convergence_message(self):
        """Test that convergence produces appropriate message."""
        bounds = (np.array([0.0, 0.0]), np.array([0.1, 0.1]))
        optimizer = CRS3Optimizer(bounds, random_state=42, tol=1e-2)
        
        result = optimizer.minimize(sphere_function, x0=np.array([0.05, 0.05]))
        
        # Should converge quickly in tight bounds
        assert result.success or result.n_eval >= result.max_eval


class TestFminCRS3:
    """Tests for the fmin_crs3 convenience function."""
    
    def test_fmin_crs3_basic(self):
        """Test fmin_crs3 convenience function."""
        bounds = (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
        
        result = fmin_crs3(
            sphere_function,
            x0=np.array([3.0, 3.0]),
            bounds=bounds,
            random_state=42,
            tol=1e-5
        )
        
        assert result.success
        assert result.fun < 1e-3
        assert np.allclose(result.x, np.array([0.0, 0.0]), atol=0.1)
    
    def test_fmin_crs3_rosenbrock(self):
        """Test fmin_crs3 on Rosenbrock function."""
        bounds = (np.array([-2.0, -1.0]), np.array([2.0, 3.0]))
        
        result = fmin_crs3(
            rosenbrock_function,
            x0=np.array([0.0, 0.0]),
            bounds=bounds,
            random_state=42,
            population_size=30,
            tol=1e-4
        )
        
        assert result.fun < 1.0
        assert result.n_eval > 0


class TestCRS3EdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_crs3_no_initial_guess(self):
        """Test optimization without providing initial guess."""
        bounds = (np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        optimizer = CRS3Optimizer(bounds, random_state=42)
        
        # Should work without x0
        result = optimizer.minimize(sphere_function)
        
        assert result.n_eval > 0
        assert np.all(result.x >= bounds[0])
        assert np.all(result.x <= bounds[1])
    
    def test_crs3_constant_function(self):
        """Test with constant function."""
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        optimizer = CRS3Optimizer(bounds, random_state=42, tol=1e-10)
        
        result = optimizer.minimize(lambda x: 5.0, x0=np.array([5.0, 5.0]))
        
        # Should converge since all values are equal
        assert result.fun == 5.0
        assert result.success
        assert "converged" in result.message.lower() or "below tolerance" in result.message.lower()
    
    def test_crs3_callback(self):
        """Test callback functionality."""
        bounds = (np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        optimizer = CRS3Optimizer(bounds, random_state=42)
        
        callback_calls = []
        
        def callback(x):
            callback_calls.append(x.copy())
        
        result = optimizer.minimize(sphere_function, x0=np.array([3.0, 3.0]), callback=callback)
        
        # Callback should have been called at least once
        assert len(callback_calls) > 0
        # Last callback should be close to result
        assert np.allclose(callback_calls[-1], result.x, atol=0.01)
