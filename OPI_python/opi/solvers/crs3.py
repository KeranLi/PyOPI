"""
Controlled Random Search with 3 Parents (CRS3) Optimizer

Implementation of the CRS3 global optimization algorithm.
This is the default optimizer used in the original OPI MATLAB implementation.

References
----------
Price (1977) - A controlled random search procedure for global optimisation
    Computer Journal, 20, 367-370.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CRS3Result:
    """Result from CRS3 optimization."""
    x: np.ndarray          # Best parameter vector
    fun: float             # Best function value
    n_iter: int            # Number of iterations
    n_eval: int            # Number of function evaluations
    success: bool          # Whether optimization converged
    message: str           # Status message


class CRS3Optimizer:
    """
    Controlled Random Search with 3 Parents (CRS3) optimizer.
    
    CRS3 is a global optimization algorithm that maintains a population
    of candidate solutions and generates new candidates by reflecting
    points through the centroid of a subset of the population.
    
    Parameters
    ----------
    bounds : tuple of ndarray
        (lower_bounds, upper_bounds) arrays defining parameter bounds
    population_size : int, optional
        Size of the candidate population. Default is 10*(n+1) where n
        is the number of parameters.
    max_iter : int, optional
        Maximum number of iterations. Default 10000.
    max_eval : int, optional
        Maximum number of function evaluations. Default 50000.
    tol : float, optional
        Convergence tolerance. Default 1e-6.
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    n_dim_ : int
        Number of parameters
    population_ : ndarray
        Current population of shape (population_size, n_dim)
    values_ : ndarray
        Function values for current population
        
    Example
    -------
    >>> bounds = (np.array([0, 0]), np.array([10, 10]))
    >>> optimizer = CRS3Optimizer(bounds)
    >>> result = optimizer.minimize(lambda x: np.sum(x**2), x0=np.array([5, 5]))
    >>> print(f"Minimum at: {result.x}")
    """
    
    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray],
                 population_size: Optional[int] = None,
                 max_iter: int = 10000,
                 max_eval: int = 50000,
                 tol: float = 1e-6,
                 random_state: Optional[int] = None):
        self.bounds = bounds
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.n_dim = len(self.lower)
        
        self.population_size = population_size or 10 * (self.n_dim + 1)
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tol = tol
        
        self.rng = np.random.RandomState(random_state)
        
        # These will be set during optimization
        self.population_ = None
        self.values_ = None
        self.n_eval_ = 0
        
    def minimize(self, func: Callable[[np.ndarray], float],
                 x0: Optional[np.ndarray] = None,
                 callback: Optional[Callable] = None) -> CRS3Result:
        """
        Minimize a function using CRS3.
        
        Parameters
        ----------
        func : callable
            Objective function to minimize
        x0 : ndarray, optional
            Initial guess. If None, random initialization is used.
        callback : callable, optional
            Called after each iteration with current best parameters
            
        Returns
        -------
        CRS3Result
            Optimization result
        """
        # Initialize population
        self._initialize_population(func, x0)
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            # Store current best
            best_idx = np.argmin(self.values_)
            best_x = self.population_[best_idx].copy()
            best_val = self.values_[best_idx]
            
            # Generate new candidate using CRS3 reflection
            new_x = self._generate_candidate()
            
            # Evaluate new candidate
            new_val = func(new_x)
            self.n_eval_ += 1
            
            # Replace worst if better
            worst_idx = np.argmax(self.values_)
            if new_val < self.values_[worst_idx]:
                self.population_[worst_idx] = new_x
                self.values_[worst_idx] = new_val
            
            # Check convergence
            val_range = np.max(self.values_) - np.min(self.values_)
            if val_range < self.tol:
                return CRS3Result(
                    x=self.population_[np.argmin(self.values_)],
                    fun=np.min(self.values_),
                    n_iter=iteration + 1,
                    n_eval=self.n_eval_,
                    success=True,
                    message="Converged: function value range below tolerance"
                )
            
            # Check max evaluations
            if self.n_eval_ >= self.max_eval:
                return CRS3Result(
                    x=self.population_[np.argmin(self.values_)],
                    fun=np.min(self.values_),
                    n_iter=iteration + 1,
                    n_eval=self.n_eval_,
                    success=False,
                    message="Maximum number of evaluations reached"
                )
            
            # Callback
            if callback:
                callback(self.population_[np.argmin(self.values_)])
        
        # Max iterations reached
        return CRS3Result(
            x=self.population_[np.argmin(self.values_)],
            fun=np.min(self.values_),
            n_iter=self.max_iter,
            n_eval=self.n_eval_,
            success=False,
            message="Maximum number of iterations reached"
        )
    
    def _initialize_population(self, func: Callable, x0: Optional[np.ndarray]):
        """Initialize the candidate population."""
        self.population_ = np.zeros((self.population_size, self.n_dim))
        self.values_ = np.zeros(self.population_size)
        
        # First member: initial guess or random
        if x0 is not None:
            self.population_[0] = np.clip(x0, self.lower, self.upper)
        else:
            self.population_[0] = self.lower + self.rng.random(self.n_dim) * (
                self.upper - self.lower
            )
        
        self.values_[0] = func(self.population_[0])
        self.n_eval_ = 1
        
        # Rest: random within bounds
        for i in range(1, self.population_size):
            self.population_[i] = self.lower + self.rng.random(self.n_dim) * (
                self.upper - self.lower
            )
            self.values_[i] = func(self.population_[i])
            self.n_eval_ += 1
    
    def _generate_candidate(self) -> np.ndarray:
        """Generate a new candidate using CRS3 reflection."""
        # Select 3 random distinct parents
        parents = self.rng.choice(self.population_size, size=3, replace=False)
        
        # Calculate centroid of two parents
        centroid = 0.5 * (self.population_[parents[0]] + self.population_[parents[1]])
        
        # Reflect third parent through centroid
        new_x = 2 * centroid - self.population_[parents[2]]
        
        # Ensure within bounds
        new_x = np.clip(new_x, self.lower, self.upper)
        
        return new_x


def fmin_crs3(func: Callable[[np.ndarray], float],
              x0: np.ndarray,
              bounds: Tuple[np.ndarray, np.ndarray],
              max_iter: int = 10000,
              max_eval: int = 50000,
              tol: float = 1e-6,
              population_size: Optional[int] = None,
              random_state: Optional[int] = None) -> CRS3Result:
    """
    Minimize a function using the CRS3 algorithm.
    
    Convenience function that creates a CRS3Optimizer and runs it.
    
    Parameters
    ----------
    func : callable
        Objective function to minimize
    x0 : ndarray
        Initial guess
    bounds : tuple of ndarray
        (lower_bounds, upper_bounds)
    max_iter : int, optional
        Maximum iterations
    max_eval : int, optional
        Maximum function evaluations
    tol : float, optional
        Convergence tolerance
    population_size : int, optional
        Population size
    random_state : int, optional
        Random seed
        
    Returns
    -------
    CRS3Result
        Optimization result
        
    Example
    -------
    >>> def rosenbrock(x):
    ...     return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    >>> bounds = (np.array([-5, -5]), np.array([5, 5]))
    >>> result = fmin_crs3(rosenbrock, x0=np.array([0, 0]), bounds=bounds)
    >>> print(f"Minimum: {result.fun:.6f} at {result.x}")
    """
    optimizer = CRS3Optimizer(
        bounds=bounds,
        population_size=population_size,
        max_iter=max_iter,
        max_eval=max_eval,
        tol=tol,
        random_state=random_state
    )
    return optimizer.minimize(func, x0)
