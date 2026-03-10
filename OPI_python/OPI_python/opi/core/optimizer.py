"""
Base optimizer class for OPI.
"""

from abc import abstractmethod
from typing import Callable, Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

from .base import OPIBase
from .exceptions import OptimizationError
from ..models.config import OptimizerConfig
from ..models.results import OptimizationResult


class BaseOptimizer(OPIBase):
    """
    Abstract base class for parameter optimizers.
    
    Provides common functionality for optimization algorithms.
    Subclasses must implement the _optimize method.
    
    Example:
        class CRS3Optimizer(BaseOptimizer):
            def _optimize(self, objective, bounds, x0):
                # CRS3 implementation
                pass
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialize optimizer.
        
        Args:
            config: Optimizer configuration
        """
        from ..models.config import OptimizerConfig
        super().__init__(None)  # Config handled separately
        self._config = config or OptimizerConfig()
        self._history: List[Tuple[NDArray[np.float64], float]] = []
    
    @property
    def config(self) -> OptimizerConfig:
        """Get optimizer configuration."""
        return self._config
    
    def optimize(self, 
                 objective: Callable[[NDArray[np.float64]], float],
                 bounds: List[Tuple[float, float]],
                 x0: Optional[NDArray[np.float64]] = None) -> OptimizationResult:
        """
        Optimize objective function.
        
        This is the main entry point. It handles:
        - Validation
        - History tracking
        - Error handling
        
        Args:
            objective: Function to minimize
            bounds: List of (min, max) bounds for each parameter
            x0: Initial guess (optional)
            
        Returns:
            Optimization results
            
        Raises:
            OptimizationError: If optimization fails
        """
        # Validate inputs
        self._validate_inputs(objective, bounds, x0)
        
        # Wrap objective to track history
        self._history = []
        
        def wrapped_objective(x):
            value = objective(x)
            self._history.append((x.copy(), value))
            return value
        
        try:
            self.log_info(f"Starting optimization with {len(bounds)} parameters")
            
            # Call implementation
            result = self._optimize(wrapped_objective, bounds, x0)
            
            # Add history to result
            if self._history:
                params_history = np.array([h[0] for h in self._history])
                values_history = np.array([h[1] for h in self._history])
                result.parameter_history = params_history
                result.misfit_history = values_history
            
            self.log_info(f"Optimization completed: {result.message}")
            
            return result
            
        except Exception as e:
            raise OptimizationError(
                f"Optimization failed: {str(e)}",
                iteration=len(self._history)
            ) from e
    
    @abstractmethod
    def _optimize(self,
                  objective: Callable[[NDArray[np.float64]], float],
                  bounds: List[Tuple[float, float]],
                  x0: Optional[NDArray[np.float64]]) -> OptimizationResult:
        """
        Implementation of optimization algorithm.
        
        Subclasses must implement this method.
        
        Args:
            objective: Function to minimize
            bounds: Parameter bounds
            x0: Initial guess
            
        Returns:
            Optimization results
        """
        pass
    
    def _validate_inputs(self,
                        objective: Callable,
                        bounds: List[Tuple[float, float]],
                        x0: Optional[NDArray[np.float64]]):
        """
        Validate optimization inputs.
        
        Args:
            objective: Objective function
            bounds: Parameter bounds
            x0: Initial guess
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not callable(objective):
            raise ValueError("objective must be callable")
        
        if not bounds:
            raise ValueError("bounds must not be empty")
        
        for i, (lb, ub) in enumerate(bounds):
            if lb >= ub:
                raise ValueError(
                    f"Bound {i}: lower bound {lb} must be < upper bound {ub}"
                )
        
        if x0 is not None:
            if len(x0) != len(bounds):
                raise ValueError(
                    f"x0 length {len(x0)} doesn't match bounds length {len(bounds)}"
                )
            for i, (xi, (lb, ub)) in enumerate(zip(x0, bounds)):
                if not (lb <= xi <= ub):
                    raise ValueError(
                        f"x0[{i}] = {xi} is outside bounds [{lb}, {ub}]"
                    )


class MultiStartOptimizer(BaseOptimizer):
    """
    Wrapper that runs optimizer multiple times from different starting points.
    
    Useful for non-convex problems to find global minimum.
    """
    
    def __init__(self, 
                 base_optimizer: BaseOptimizer,
                 n_starts: int = 5,
                 config: Optional[OptimizerConfig] = None):
        """
        Initialize multi-start optimizer.
        
        Args:
            base_optimizer: Base optimizer to use
            n_starts: Number of random starts
            config: Configuration
        """
        super().__init__(config)
        self._base_optimizer = base_optimizer
        self._n_starts = n_starts
    
    def _optimize(self,
                  objective: Callable[[NDArray[np.float64]], float],
                  bounds: List[Tuple[float, float]],
                  x0: Optional[NDArray[np.float64]]) -> OptimizationResult:
        """Run multiple optimizations from different starts."""
        import numpy as np
        
        best_result = None
        best_value = float('inf')
        
        for i in range(self._n_starts):
            self.log_info(f"Multi-start iteration {i+1}/{self._n_starts}")
            
            # Generate random starting point if not provided
            if x0 is None or i > 0:
                x0_start = np.array([
                    np.random.uniform(lb, ub) for lb, ub in bounds
                ])
            else:
                x0_start = x0
            
            # Run optimization
            result = self._base_optimizer.optimize(objective, bounds, x0_start)
            
            # Track best
            if result.final_misfit < best_value:
                best_value = result.final_misfit
                best_result = result
        
        if best_result is None:
            raise OptimizationError("All optimization runs failed")
        
        return best_result
