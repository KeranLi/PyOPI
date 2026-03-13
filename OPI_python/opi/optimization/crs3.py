"""
CRS3 Optimization Algorithm - MATLAB Compatible Implementation

Controlled Random Search 3 (CRS3) is a global optimization algorithm.
This implementation matches MATLAB's fminCRS3.m exactly.

References:
- Price, W.L., 1987. A controlled random search procedure for global optimisation.
- Brachetti et al., 1997. A convergence analysis of CRS3.
- Jiao et al., 2006. A modification to the new version of the Price's Algorithm.
"""

import numpy as np
from scipy.optimize import OptimizeResult
import warnings


def fmin_crs3(func, bounds, args=(), mu=25, epsilon=1e-4, max_iter=10000,
              random_state=None, verbose=False, callback=None, 
              solutions=None, parallel=False):
    """
    CRS3 (Controlled Random Search 3) global optimization.
    
    Matches MATLAB's fminCRS3 function.
    
    Parameters
    ----------
    func : callable
        Objective function to minimize: func(x, *args) -> (f, nu)
        where f is the objective value and nu is degrees of freedom.
    bounds : sequence of tuples
        Bounds for each dimension: [(low, high), ...]
    args : tuple, optional
        Additional arguments to pass to func
    mu : int, optional
        Population size factor. Default is 25.
    epsilon : float, optional
        Convergence tolerance (std of search set < epsilon). Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations. Default is 10000.
    random_state : int or RandomState, optional
        Random seed or generator
    verbose : bool, optional
        Print progress information
    callback : callable, optional
        Called after each iteration: callback(x, fval)
    solutions : ndarray, optional
        Restart solutions array (n_solutions x (2 + n_parameters))
        Each row: [f, nu, beta1, beta2, ...]
    parallel : bool, optional
        Use parallel computation (not fully implemented yet)
    
    Returns
    -------
    OptimizeResult
        Result object with x, fun, success, n_iter, etc.
    """
    # Initialize random number generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # CRS3 variable omega (scales the weighting during search)
    omega = 1000.0
    
    # Parse bounds
    bounds = np.array(bounds)
    n_parameters = len(bounds)
    beta_lb = bounds[:, 0]
    beta_ub = bounds[:, 1]
    
    # Check bounds
    if np.any(beta_lb > beta_ub):
        raise ValueError("Upper bounds must be greater than lower bounds")
    
    # Determine number of free parameters
    n_free = np.sum((beta_ub - beta_lb) != 0)
    
    # Define size of solution set (Price 1987, p. 136)
    m_set = int(np.ceil(mu * (n_free + 1)))
    
    if verbose:
        print(f"Modified controlled random search, fminCRS3")
        print(f"Number of parameters = {n_parameters}")
        print(f"Number of free parameters = {n_free}")
        print(f"Factor for size of search set, mu = {mu}")
        print(f"Size of search set = {m_set}")
        print(f"Specified epsilon for stopping criterion = {epsilon}")
        print(f"Workers for parallel calculation = {1 if not parallel else 'auto'}\n")
    
    # Initialize arrays
    beta_set = np.zeros((m_set, n_parameters))
    f_set = np.zeros(m_set)
    
    # Process restart solutions if present
    m = 0
    if solutions is not None and len(solutions) > 0:
        # Remove solutions where f is nan
        solutions = solutions[~np.isnan(solutions[:, 0])]
        n_restart = len(solutions)
        
        # Load initial solutions into search set
        m = min(n_restart, m_set)
        f_set[:m] = solutions[:m, 0]
        beta_set[:m, :] = solutions[:m, 2:2+n_parameters]
        
        if verbose:
            print(f"Loaded {m} restart solutions")
    
    # Complete rest of initial solution set
    if m < m_set:
        for i in range(m, m_set):
            # Generate random point within bounds
            beta = beta_lb + (beta_ub - beta_lb) * rng.uniform(0, 1, n_parameters)
            
            # Evaluate objective function
            result = func(beta, *args)
            if isinstance(result, tuple):
                f, nu = result[0], result[1]
            else:
                f = result
            
            beta_set[i, :] = beta
            f_set[i] = f
            
            if callback is not None:
                callback(beta, f)
    
    # Set up main search set
    # Save best and worst solutions from the first 1:m_set solutions
    f0_min = np.min(f_set)
    f0_max = np.max(f_set)
    
    # Sort to get best m_set solutions
    i_sort = np.argsort(f_set)
    beta_set = beta_set[i_sort[:m_set], :]
    f_set = f_set[i_sort[:m_set]]
    
    # Track best three and worst
    i_min = np.array([0, 1, 2])  # Indices of best three
    f_min = f_set[i_min]
    i_max = m_set - 1  # Index of worst
    f_max = f_set[i_max]
    
    # Main search loop
    is_new_min = False
    n_iter = 0
    
    while n_iter < max_iter:
        n_iter += 1
        
        # Generate new point in parameter space
        if is_new_min:
            # Try to estimate a better parameter point using quadratic approximation
            # Three-point quadratic solution (Jiao et al., 2006)
            beta1, beta2, beta3 = beta_set[i_min[0]], beta_set[i_min[1]], beta_set[i_min[2]]
            f1, f2, f3 = f_min[0], f_min[1], f_min[2]
            
            # Quadratic approximation formula
            numerator = (beta2**2 - beta3**2) * f1 + (beta3**2 - beta1**2) * f2 + (beta1**2 - beta2**2) * f3
            denominator = (beta2 - beta3) * f1 + (beta3 - beta1) * f2 + (beta1 - beta2) * f3
            
            # Avoid division by zero
            beta = np.where(np.abs(denominator) > 1e-10, 0.5 * numerator / denominator, beta1)
            
            is_new_min = False
            # Check if quadratic estimate is within bounds
            if np.all(beta >= beta_lb) and np.all(beta <= beta_ub):
                is_new_min = True
        
        if not is_new_min:
            # Use reflection method to estimate new parameter point
            while True:
                # Select N+1 trial solutions from the search set randomly
                i_subset = rng.choice(m_set, size=n_parameters + 1, replace=False)
                beta_subset = beta_set[i_subset, :]
                f_subset = f_set[i_subset]
                
                # Calculate weighted centroid (excluding worst of subset)
                # Find worst in subset
                worst_in_subset = np.argmax(f_subset)
                
                # Calculate phi and weights
                phi = omega * (f_max - f_min[0])**2 / (f0_max - f0_min + 1e-10)
                
                # Compute weights (excluding worst point)
                weights = 1.0 / (f_subset - f_min[0] + phi + 1e-10)
                weights[worst_in_subset] = 0
                weights = weights / np.sum(weights)
                
                # Weighted centroid
                beta_center = np.sum(weights[:, np.newaxis] * beta_subset, axis=0)
                f_center = np.sum(weights * f_subset)
                
                # Calculate weighted reflection point
                beta_worst = beta_subset[worst_in_subset, :]
                f_worst = f_subset[worst_in_subset]
                
                if f_center <= f_worst:
                    alpha = 1 - (f_worst - f_center) / (f_max - f_min[0] + phi + 1e-10)
                    beta = beta_center - alpha * (beta_worst - beta_center)
                else:
                    alpha = 1 + (f_worst - f_center) / (f_max - f_min[0] + phi + 1e-10)
                    beta = beta_worst - alpha * (beta_center - beta_worst)
                
                # Check if new point is within specified bounds
                if np.all(beta >= beta_lb) and np.all(beta <= beta_ub):
                    break
        
        # Evaluate objective function
        result = func(beta, *args)
        if isinstance(result, tuple):
            f, nu = result[0], result[1]
        else:
            f = result
        
        if callback is not None:
            callback(beta, f)
        
        # Check to see if there is new minimum solution
        is_new_min_candidate = f < f_min[2]
        
        # Update solution set
        if f < f_max:
            # Use the trial solution as a replacement for worst solution
            beta_set[i_max, :] = beta
            f_set[i_max] = f
            
            # Sort revised search set
            i_sort = np.argsort(f_set)
            i_max = i_sort[-1]
            f_max = f_set[i_max]
            i_min = i_sort[:3]
            f_min = f_set[i_min]
            
            # Check stopping criterion
            # Stop when std(f_set) < epsilon
            if np.std(f_set) < epsilon:
                if verbose:
                    print(f"Converged at iteration {n_iter}")
                    print(f"std(f_set) = {np.std(f_set):.6e} < epsilon = {epsilon}")
                break
    
    # Return best solution
    beta_min = beta_set[i_min[0], :]
    f_min_final = f_set[i_min[0]]
    
    if verbose and n_iter >= max_iter:
        print(f"Maximum iterations ({max_iter}) reached")
        print(f"Final std(f_set) = {np.std(f_set):.6e}")
    
    result = OptimizeResult(
        x=beta_min,
        fun=f_min_final,
        n_iter=n_iter,
        success=True,
        message="Optimization terminated successfully.",
        population=beta_set,
        f_population=f_set
    )
    
    return result


class CRS3Optimizer:
    """
    CRS3 Optimizer class for sequential optimization with restart capability.
    """
    
    def __init__(self, mu=25, epsilon=1e-4, max_iter=10000, random_state=None):
        """
        Initialize CRS3 optimizer.
        
        Parameters
        ----------
        mu : int
            Population size factor
        epsilon : float
            Convergence tolerance (std of search set)
        max_iter : int
            Maximum iterations
        random_state : int or RandomState
            Random seed
        """
        self.mu = mu
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self.history = []
    
    def minimize(self, func, bounds, args=(), x0=None, callback=None, 
                 solutions=None, parallel=False):
        """
        Minimize objective function.
        
        Parameters
        ----------
        func : callable
            Objective function
        bounds : sequence of tuples
            Bounds for variables
        args : tuple
            Additional arguments
        x0 : ndarray, optional
            Initial guess (not used in CRS3, but kept for API consistency)
        callback : callable
            Callback function
        solutions : ndarray, optional
            Restart solutions array
        parallel : bool, optional
            Use parallel computation
        
        Returns
        -------
        OptimizeResult
        """
        result = fmin_crs3(
            func, bounds, args=args,
            mu=self.mu, epsilon=self.epsilon, max_iter=self.max_iter,
            random_state=self.random_state, verbose=False, callback=callback,
            solutions=solutions, parallel=parallel
        )
        
        self.history.append({
            'x': result.x,
            'fun': result.fun,
            'n_iter': result.n_iter
        })
        
        return result
    
    def get_history(self):
        """Get optimization history."""
        return self.history


if __name__ == "__main__":
    print("Testing CRS3 optimization (MATLAB compatible)...")
    
    # Test 1: Simple quadratic
    print("\nTest 1: Simple quadratic")
    def quadratic(x):
        return np.sum((x - 0.5)**2)
    
    bounds = [(-1, 1), (-1, 1)]
    result = fmin_crs3(quadratic, bounds, mu=10, epsilon=1e-4, 
                      max_iter=5000, verbose=True, random_state=42)
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    print(f"  Iterations: {result.n_iter}")
    
    # Test 2: With restart solutions
    print("\nTest 2: With restart solutions")
    # Create some initial solutions
    restart_solutions = np.array([
        [0.1, 0, 0.4, 0.6],  # f=0.1, nu=0, beta=[0.4, 0.6]
        [0.05, 0, 0.45, 0.55],  # f=0.05, nu=0, beta=[0.45, 0.55]
    ])
    result = fmin_crs3(quadratic, bounds, mu=10, epsilon=1e-4,
                      max_iter=5000, verbose=False, random_state=42,
                      solutions=restart_solutions)
    print(f"  Solution: x = {result.x}")
    print(f"  Minimum: f = {result.fun:.6e}")
    
    print("\nTests completed!")
