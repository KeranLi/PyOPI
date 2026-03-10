"""
Parameter Sweep and Ensemble Simulation for OPI

This module provides parallel parameter sweep and ensemble simulation
capabilities for OPI models using joblib.

Classes
-------
ParameterSweep : Run simulations across parameter grids
EnsembleRunner : Run ensemble simulations with perturbations

Examples
--------
>>> from opi.parallel import ParameterSweep
>>> from opi.models import OneWindModel
>>> model = OneWindModel(topography)
>>> sweep = ParameterSweep(model, n_jobs=4)
>>> results = sweep.run(param_grid)
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from dataclasses import dataclass, asdict
from copy import deepcopy
import warnings

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


@dataclass
class SweepResult:
    """Result from a single parameter sweep run.
    
    Attributes
    ----------
    params : Dict[str, Any]
        Parameters used for this run
    result : Any
        Model result object
    success : bool
        Whether the run completed successfully
    error : Optional[str]
        Error message if failed
    run_id : int
        Unique identifier for this run
    """
    params: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str]
    run_id: int


@dataclass
class EnsembleResult:
    """Result from ensemble simulation.
    
    Attributes
    ----------
    base_params : Dict[str, Any]
        Base parameters used
    perturbations : Dict[str, np.ndarray]
        Applied perturbations for each parameter
    results : List[Any]
        List of model results
    failed_indices : List[int]
        Indices of failed runs
    statistics : Dict[str, Any]
        Summary statistics
    """
    base_params: Dict[str, Any]
    perturbations: Dict[str, np.ndarray]
    results: List[Any]
    failed_indices: List[int]
    statistics: Dict[str, Any]


class ParameterSweep:
    """Parameter sweep for sensitivity analysis.
    
    Run simulations across a grid of parameter values to analyze
    model sensitivity and behavior.
    
    Parameters
    ----------
    model : object
        OPI model instance (e.g., OneWindModel, TwoWindModel)
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    backend : str, default='loky'
        Joblib parallel backend ('loky', 'threading', 'multiprocessing')
    verbose : int, default=0
        Verbosity level for joblib
    
    Attributes
    ----------
    model : object
        The model instance
    n_jobs : int
        Number of parallel jobs
    backend : str
        Parallel backend
    
    Examples
    --------
    >>> from opi.models import OneWindModel
    >>> from opi.parallel import ParameterSweep
    >>> 
    >>> model = OneWindModel(topography)
    >>> sweep = ParameterSweep(model, n_jobs=4)
    >>> 
    >>> # Define parameter grid
    >>> param_grid = {
    ...     'U': [5, 10, 15, 20],
    ...     'T0': [280, 285, 290, 295]
    ... }
    >>> results = sweep.run(param_grid, base_params={'M': 0.8, ...})
    
    Notes
    -----
    Requires joblib for parallelization. Falls back to sequential
    execution if joblib is not available.
    """
    
    def __init__(
        self,
        model: Any,
        n_jobs: int = -1,
        backend: str = 'loky',
        verbose: int = 0
    ):
        if not HAS_JOBLIB:
            warnings.warn(
                "joblib not available. Install with: pip install joblib. "
                "Falling back to sequential execution.",
                RuntimeWarning
            )
        
        self.model = model
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
    
    def run(
        self,
        param_grid: Dict[str, Sequence],
        base_params: Optional[Dict[str, Any]] = None,
        samples: Optional[Dict] = None,
        return_errors: bool = False
    ) -> List[SweepResult]:
        """Run simulations for each parameter combination.
        
        Parameters
        ----------
        param_grid : Dict[str, Sequence]
            Dictionary mapping parameter names to lists of values
        base_params : Dict[str, Any], optional
            Base parameters to use for parameters not in param_grid
        samples : Dict, optional
            Sample data to pass to model.run()
        return_errors : bool, default=False
            If True, include failed runs in results
            
        Returns
        -------
        List[SweepResult]
            List of results for each parameter combination
            
        Raises
        ------
        ValueError
            If param_grid is empty or contains invalid parameters
            
        Examples
        --------
        >>> param_grid = {
        ...     'U': [5.0, 10.0, 15.0],
        ...     'T0': [280.0, 290.0]
        ... }
        >>> base = {'M': 0.8, 'kappa': 1000.0, 'tau_c': 1000.0}
        >>> results = sweep.run(param_grid, base_params=base)
        >>> 
        >>> # Access results
        >>> for r in results:
        ...     if r.success:
        ...         print(f"U={r.params['U']}: chi_r2={r.result.chi_r2}")
        """
        if not param_grid:
            raise ValueError("param_grid cannot be empty")
        
        base_params = base_params or {}
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        from itertools import product
        combinations = list(product(*param_values))
        
        # Create list of all parameter sets
        param_sets = []
        for i, combo in enumerate(combinations):
            params = deepcopy(base_params)
            params.update(dict(zip(param_names, combo)))
            param_sets.append((i, params))
        
        # Run simulations
        if HAS_JOBLIB and self.n_jobs != 1:
            results = self._run_parallel(param_sets, samples, return_errors)
        else:
            results = self._run_sequential(param_sets, samples, return_errors)
        
        return results
    
    def run_one_factor(
        self,
        param_name: str,
        param_values: Sequence,
        base_params: Dict[str, Any],
        samples: Optional[Dict] = None,
        return_errors: bool = False
    ) -> List[SweepResult]:
        """Run one-factor-at-a-time sensitivity analysis.
        
        Vary one parameter while keeping all others constant.
        
        Parameters
        ----------
        param_name : str
            Name of parameter to vary
        param_values : Sequence
            Values to test for the parameter
        base_params : Dict[str, Any]
            Base parameters (constant for all runs)
        samples : Dict, optional
            Sample data to pass to model.run()
        return_errors : bool, default=False
            If True, include failed runs in results
            
        Returns
        -------
        List[SweepResult]
            Results for each parameter value
            
        Raises
        ------
        ValueError
            If param_name is already in base_params and would be overwritten
            
        Examples
        --------
        >>> base = {'U': 10.0, 'T0': 288.0, 'M': 0.8, ...}
        >>> results = sweep.run_one_factor(
        ...     'U', [5.0, 10.0, 15.0, 20.0], base
        ... )
        >>> 
        >>> # Plot sensitivity
        >>> U_vals = [r.params['U'] for r in results if r.success]
        >>> chi_vals = [r.result.chi_r2 for r in results if r.success]
        """
        param_grid = {param_name: param_values}
        return self.run(param_grid, base_params, samples, return_errors)
    
    def _run_parallel(
        self,
        param_sets: List[Tuple[int, Dict]],
        samples: Optional[Dict],
        return_errors: bool
    ) -> List[SweepResult]:
        """Run parameter sets in parallel."""
        
        def run_single(args: Tuple[int, Dict]) -> SweepResult:
            run_id, params = args
            return self._execute_run(run_id, params, samples)
        
        # Create progress bar
        pbar = tqdm(total=len(param_sets), desc="Running sweep") if HAS_TQDM else None
        
        # Run in parallel
        parallel = Parallel(
            n_jobs=self.n_jobs,
            backend=self.backend,
            verbose=self.verbose
        )
        
        results = parallel(delayed(run_single)(ps) for ps in param_sets)
        
        if pbar:
            pbar.close()
        
        if not return_errors:
            results = [r for r in results if r.success]
        
        return results
    
    def _run_sequential(
        self,
        param_sets: List[Tuple[int, Dict]],
        samples: Optional[Dict],
        return_errors: bool
    ) -> List[SweepResult]:
        """Run parameter sets sequentially."""
        results = []
        
        iterator = tqdm(param_sets, desc="Running sweep") if HAS_TQDM else param_sets
        
        for run_id, params in iterator:
            result = self._execute_run(run_id, params, samples)
            if result.success or return_errors:
                results.append(result)
        
        return results
    
    def _execute_run(
        self,
        run_id: int,
        params: Dict[str, Any],
        samples: Optional[Dict]
    ) -> SweepResult:
        """Execute a single model run."""
        try:
            result = self.model.run(params, samples)
            return SweepResult(
                params=params,
                result=result,
                success=True,
                error=None,
                run_id=run_id
            )
        except Exception as e:
            return SweepResult(
                params=params,
                result=None,
                success=False,
                error=str(e),
                run_id=run_id
            )


class EnsembleRunner:
    """Ensemble simulation with parameter perturbations.
    
    Run multiple simulations with random perturbations to base parameters
    to assess uncertainty and robustness of results.
    
    Parameters
    ----------
    model : object
        OPI model instance
    n_ensemble : int
        Number of ensemble members
    n_jobs : int, default=-1
        Number of parallel jobs
    backend : str, default='loky'
        Joblib parallel backend
    random_state : int, optional
        Random seed for reproducibility
    verbose : int, default=0
        Verbosity level
    
    Attributes
    ----------
    model : object
        The model instance
    n_ensemble : int
        Number of ensemble members
    rng : np.random.Generator
        Random number generator
    
    Examples
    --------
    >>> from opi.models import OneWindModel
    >>> from opi.parallel import EnsembleRunner
    >>> 
    >>> model = OneWindModel(topography)
    >>> runner = EnsembleRunner(model, n_ensemble=100, n_jobs=4)
    >>> 
    >>> base_params = {'U': 10.0, 'T0': 288.0, 'M': 0.8, ...}
    >>> ensemble = runner.run(base_params, noise_std=0.1)
    
    Notes
    -----
    Parameter perturbations are applied as relative Gaussian noise:
    perturbed_value = base_value * (1 + noise)
    where noise ~ N(0, noise_std)
    """
    
    def __init__(
        self,
        model: Any,
        n_ensemble: int,
        n_jobs: int = -1,
        backend: str = 'loky',
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        if not HAS_JOBLIB:
            warnings.warn(
                "joblib not available. Install with: pip install joblib. "
                "Falling back to sequential execution.",
                RuntimeWarning
            )
        
        self.model = model
        self.n_ensemble = n_ensemble
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
    
    def run(
        self,
        base_params: Dict[str, Any],
        noise_std: float = 0.1,
        noise_params: Optional[List[str]] = None,
        samples: Optional[Dict] = None,
        relative_noise: bool = True
    ) -> EnsembleResult:
        """Run ensemble with parameter perturbations.
        
        Parameters
        ----------
        base_params : Dict[str, Any]
            Base parameter values
        noise_std : float, default=0.1
            Standard deviation of Gaussian noise
        noise_params : List[str], optional
            Parameters to perturb. If None, all numeric parameters
            are perturbed.
        samples : Dict, optional
            Sample data to pass to model.run()
        relative_noise : bool, default=True
            If True, noise is relative (multiplicative).
            If False, noise is absolute (additive).
            
        Returns
        -------
        EnsembleResult
            Ensemble results with statistics
            
        Raises
        ------
        ValueError
            If base_params is empty or n_ensemble <= 0
            
        Examples
        --------
        >>> base = {'U': 10.0, 'T0': 288.0, 'M': 0.8}
        >>> 
        >>> # Perturb all parameters with 10% relative noise
        >>> result = runner.run(base, noise_std=0.1)
        >>> 
        >>> # Perturb only specific parameters
        >>> result = runner.run(
        ...     base, 
        ...     noise_std=0.05, 
        ...     noise_params=['U', 'M']
        ... )
        >>> 
        >>> # Access ensemble statistics
        >>> print(result.statistics['mean_chi_r2'])
        >>> print(result.statistics['std_chi_r2'])
        """
        if not base_params:
            raise ValueError("base_params cannot be empty")
        if self.n_ensemble <= 0:
            raise ValueError("n_ensemble must be positive")
        
        # Determine which parameters to perturb
        if noise_params is None:
            noise_params = [
                k for k, v in base_params.items() 
                if isinstance(v, (int, float, np.number))
            ]
        
        # Generate perturbations
        perturbations = self._generate_perturbations(
            base_params, noise_params, noise_std, relative_noise
        )
        
        # Create parameter sets
        param_sets = []
        for i in range(self.n_ensemble):
            params = deepcopy(base_params)
            for p in noise_params:
                params[p] = perturbations[p][i]
            param_sets.append((i, params))
        
        # Run ensemble
        if HAS_JOBLIB and self.n_jobs != 1:
            results_list = self._run_parallel(param_sets, samples)
        else:
            results_list = self._run_sequential(param_sets, samples)
        
        # Process results
        successful_results = []
        failed_indices = []
        
        for i, result in enumerate(results_list):
            if result.success:
                successful_results.append(result.result)
            else:
                failed_indices.append(i)
        
        # Calculate statistics
        statistics = self._calculate_statistics(successful_results)
        
        return EnsembleResult(
            base_params=base_params,
            perturbations=perturbations,
            results=successful_results,
            failed_indices=failed_indices,
            statistics=statistics
        )
    
    def _generate_perturbations(
        self,
        base_params: Dict[str, Any],
        noise_params: List[str],
        noise_std: float,
        relative_noise: bool
    ) -> Dict[str, np.ndarray]:
        """Generate random perturbations for parameters."""
        perturbations = {}
        
        for param in noise_params:
            if param not in base_params:
                continue
            
            base_value = base_params[param]
            
            if relative_noise:
                # Relative (multiplicative) noise
                noise = self.rng.normal(0, noise_std, self.n_ensemble)
                perturbed = base_value * (1 + noise)
            else:
                # Absolute (additive) noise
                noise = self.rng.normal(0, noise_std, self.n_ensemble)
                perturbed = base_value + noise
            
            perturbations[param] = perturbed
        
        return perturbations
    
    def _run_parallel(
        self,
        param_sets: List[Tuple[int, Dict]],
        samples: Optional[Dict]
    ) -> List[SweepResult]:
        """Run ensemble members in parallel."""
        
        def run_single(args: Tuple[int, Dict]) -> SweepResult:
            run_id, params = args
            try:
                result = self.model.run(params, samples)
                return SweepResult(
                    params=params, result=result, success=True,
                    error=None, run_id=run_id
                )
            except Exception as e:
                return SweepResult(
                    params=params, result=None, success=False,
                    error=str(e), run_id=run_id
                )
        
        pbar = tqdm(total=len(param_sets), desc="Running ensemble") if HAS_TQDM else None
        
        parallel = Parallel(
            n_jobs=self.n_jobs,
            backend=self.backend,
            verbose=self.verbose
        )
        
        results = parallel(delayed(run_single)(ps) for ps in param_sets)
        
        if pbar:
            pbar.close()
        
        return results
    
    def _run_sequential(
        self,
        param_sets: List[Tuple[int, Dict]],
        samples: Optional[Dict]
    ) -> List[SweepResult]:
        """Run ensemble members sequentially."""
        results = []
        
        iterator = tqdm(param_sets, desc="Running ensemble") if HAS_TQDM else param_sets
        
        for run_id, params in iterator:
            try:
                result = self.model.run(params, samples)
                results.append(SweepResult(
                    params=params, result=result, success=True,
                    error=None, run_id=run_id
                ))
            except Exception as e:
                results.append(SweepResult(
                    params=params, result=None, success=False,
                    error=str(e), run_id=run_id
                ))
        
        return results
    
    def _calculate_statistics(
        self,
        results: List[Any]
    ) -> Dict[str, Any]:
        """Calculate ensemble statistics."""
        if not results:
            return {'n_successful': 0}
        
        stats = {'n_successful': len(results)}
        
        # Extract chi_r2 if available
        chi_r2_values = []
        for r in results:
            if hasattr(r, 'chi_r2') and np.isfinite(r.chi_r2):
                chi_r2_values.append(r.chi_r2)
        
        if chi_r2_values:
            chi_r2_array = np.array(chi_r2_values)
            stats['chi_r2'] = {
                'mean': float(np.mean(chi_r2_array)),
                'std': float(np.std(chi_r2_array)),
                'min': float(np.min(chi_r2_array)),
                'max': float(np.max(chi_r2_array)),
                'median': float(np.median(chi_r2_array)),
                'q25': float(np.percentile(chi_r2_array, 25)),
                'q75': float(np.percentile(chi_r2_array, 75))
            }
        
        # Extract precipitation statistics if available
        p_grids = []
        for r in results:
            if hasattr(r, 'p_grid'):
                p_grids.append(r.p_grid)
        
        if p_grids:
            p_stack = np.stack(p_grids)
            stats['precipitation'] = {
                'mean_grid': np.mean(p_stack, axis=0),
                'std_grid': np.std(p_stack, axis=0),
                'min_grid': np.min(p_stack, axis=0),
                'max_grid': np.max(p_stack, axis=0)
            }
        
        # Extract isotope statistics if available
        d2H_grids = []
        for r in results:
            if hasattr(r, 'd2H_grid'):
                d2H_grids.append(r.d2H_grid)
        
        if d2H_grids:
            d2H_stack = np.stack(d2H_grids)
            stats['d2H'] = {
                'mean_grid': np.nanmean(d2H_stack, axis=0),
                'std_grid': np.nanstd(d2H_stack, axis=0)
            }
        
        return stats
