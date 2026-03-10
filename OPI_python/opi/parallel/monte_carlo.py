"""
Monte Carlo Simulation for OPI

This module provides Monte Carlo simulation capabilities for uncertainty
quantification and probabilistic analysis of OPI models.

Classes
-------
MonteCarloSimulator : Run Monte Carlo simulations with parameter distributions

Examples
--------
>>> from opi.parallel import MonteCarloSimulator
>>> from opi.models import OneWindModel
>>> model = OneWindModel(topography)
>>> mc = MonteCarloSimulator(model, n_samples=1000, n_jobs=4)
>>> 
>>> # Define parameter distributions
>>> distributions = {
...     'U': ('normal', {'loc': 10.0, 'scale': 2.0}),
...     'T0': ('uniform', {'low': 280.0, 'high': 290.0})
... }
>>> results = mc.run(distributions)
>>> stats = mc.analyze_results(results)
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from dataclasses import dataclass
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


# Distribution type aliases
DistributionSpec = Tuple[str, Dict[str, Any]]
ParamDistributions = Dict[str, DistributionSpec]


@dataclass
class MonteCarloResult:
    """Result from a single Monte Carlo sample.
    
    Attributes
    ----------
    sample_id : int
        Unique identifier for this sample
    params : Dict[str, Any]
        Sampled parameters
    result : Any
        Model result object
    success : bool
        Whether the simulation succeeded
    error : Optional[str]
        Error message if failed
    """
    sample_id: int
    params: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str]


@dataclass
class MonteCarloStatistics:
    """Statistics from Monte Carlo analysis.
    
    Attributes
    ----------
    n_samples : int
        Total number of samples
    n_successful : int
        Number of successful simulations
    success_rate : float
        Fraction of successful simulations
    parameter_stats : Dict[str, Dict[str, float]]
        Statistics for each parameter
    output_stats : Dict[str, Any]
        Statistics for model outputs
    confidence_intervals : Dict[str, Dict[str, np.ndarray]]
        Confidence intervals for outputs
    """
    n_samples: int
    n_successful: int
    success_rate: float
    parameter_stats: Dict[str, Dict[str, float]]
    output_stats: Dict[str, Any]
    confidence_intervals: Dict[str, Dict[str, np.ndarray]]


class MonteCarloSimulator:
    """Monte Carlo simulator for uncertainty quantification.
    
    Run simulations with parameters sampled from specified probability
    distributions to quantify uncertainty in model outputs.
    
    Parameters
    ----------
    model : object
        OPI model instance (e.g., OneWindModel, TwoWindModel)
    n_samples : int
        Number of Monte Carlo samples
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    backend : str, default='loky'
        Joblib parallel backend ('loky', 'threading', 'multiprocessing')
    random_state : int, optional
        Random seed for reproducibility
    verbose : int, default=0
        Verbosity level for joblib
    
    Attributes
    ----------
    model : object
        The model instance
    n_samples : int
        Number of samples
    n_jobs : int
        Number of parallel jobs
    rng : np.random.Generator
        Random number generator
    
    Examples
    --------
    >>> from opi.models import OneWindModel
    >>> from opi.parallel import MonteCarloSimulator
    >>> 
    >>> model = OneWindModel(topography)
    >>> mc = MonteCarloSimulator(model, n_samples=1000, n_jobs=4)
    >>> 
    >>> # Define parameter distributions
    >>> distributions = {
    ...     'U': ('normal', {'loc': 10.0, 'scale': 2.0}),
    ...     'T0': ('uniform', {'low': 280.0, 'high': 290.0}),
    ...     'M': ('truncated_normal', {'loc': 0.8, 'scale': 0.1, 'low': 0.5, 'high': 1.2})
    ... }
    >>> 
    >>> # Run simulation
    >>> results = mc.run(distributions)
    >>> 
    >>> # Analyze results
    >>> stats = mc.analyze_results(results)
    >>> print(f"Mean chi_r2: {stats.output_stats['chi_r2']['mean']}")
    >>> print(f"95% CI: [{stats.output_stats['chi_r2']['ci_95'][0]}, "
    ...       f"{stats.output_stats['chi_r2']['ci_95'][1]}]")
    
    Notes
    -----
    Available distributions:
    - 'normal': Normal distribution (loc, scale)
    - 'uniform': Uniform distribution (low, high)
    - 'lognormal': Log-normal distribution (mean, sigma)
    - 'truncated_normal': Truncated normal (loc, scale, low, high)
    - 'triangular': Triangular distribution (low, mode, high)
    - 'beta': Beta distribution (a, b) on [0, 1]
    """
    
    # Supported distributions
    SUPPORTED_DISTRIBUTIONS = [
        'normal', 'uniform', 'lognormal', 'truncated_normal',
        'triangular', 'beta', 'gamma', 'exponential'
    ]
    
    def __init__(
        self,
        model: Any,
        n_samples: int,
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
        
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        
        self.model = model
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)
    
    def run(
        self,
        param_distributions: ParamDistributions,
        base_params: Optional[Dict[str, Any]] = None,
        samples: Optional[Dict] = None,
        return_errors: bool = False
    ) -> List[MonteCarloResult]:
        """Run Monte Carlo simulation with parameter distributions.
        
        Parameters
        ----------
        param_distributions : Dict[str, Tuple[str, Dict]]
            Dictionary mapping parameter names to (distribution, params) tuples.
            Example: {'U': ('normal', {'loc': 10.0, 'scale': 2.0})}
        base_params : Dict[str, Any], optional
            Base parameters for parameters not being sampled
        samples : Dict, optional
            Sample data to pass to model.run()
        return_errors : bool, default=False
            If True, include failed simulations in results
            
        Returns
        -------
        List[MonteCarloResult]
            Results for each Monte Carlo sample
            
        Raises
        ------
        ValueError
            If param_distributions is empty or contains unsupported distributions
            
        Examples
        --------
        >>> # Normal distribution
        >>> dists = {'U': ('normal', {'loc': 10.0, 'scale': 2.0})}
        >>> 
        >>> # Uniform distribution
        >>> dists = {'T0': ('uniform', {'low': 280.0, 'high': 290.0})}
        >>> 
        >>> # Truncated normal (bounded)
        >>> dists = {
        ...     'M': ('truncated_normal', {
        ...         'loc': 0.8, 'scale': 0.1, 'low': 0.5, 'high': 1.2
        ...     })
        ... }
        >>> 
        >>> results = mc.run(dists, base_params={'kappa': 1000.0})
        """
        if not param_distributions:
            raise ValueError("param_distributions cannot be empty")
        
        # Validate distributions
        self._validate_distributions(param_distributions)
        
        base_params = base_params or {}
        
        # Sample parameters
        sampled_params = self._sample_parameters(param_distributions)
        
        # Create parameter sets
        param_sets = []
        for i in range(self.n_samples):
            params = deepcopy(base_params)
            params.update({k: v[i] for k, v in sampled_params.items()})
            param_sets.append((i, params))
        
        # Run simulations
        if HAS_JOBLIB and self.n_jobs != 1:
            results = self._run_parallel(param_sets, samples)
        else:
            results = self._run_sequential(param_sets, samples)
        
        if not return_errors:
            results = [r for r in results if r.success]
        
        return results
    
    def analyze_results(
        self,
        results: List[MonteCarloResult],
        confidence_levels: Sequence[float] = (0.68, 0.95, 0.99)
    ) -> MonteCarloStatistics:
        """Calculate statistics from Monte Carlo results.
        
        Parameters
        ----------
        results : List[MonteCarloResult]
            Results from run()
        confidence_levels : Sequence[float], default=(0.68, 0.95, 0.99)
            Confidence levels for intervals (as fractions, e.g., 0.95 for 95%)
            
        Returns
        -------
        MonteCarloStatistics
            Comprehensive statistics from the simulation
            
        Examples
        --------
        >>> results = mc.run(distributions)
        >>> stats = mc.analyze_results(results)
        >>> 
        >>> # Overall success rate
        >>> print(f"Success rate: {stats.success_rate:.2%}")
        >>> 
        >>> # Chi-squared statistics
        >>> chi_stats = stats.output_stats['chi_r2']
        >>> print(f"Mean: {chi_stats['mean']:.3f}")
        >>> print(f"Std: {chi_stats['std']:.3f}")
        >>> print(f"95% CI: [{chi_stats['ci_95'][0]:.3f}, {chi_stats['ci_95'][1]:.3f}]")
        >>> 
        >>> # Grid-based statistics (if available)
        >>> if 'precipitation' in stats.output_stats:
        ...     p_mean = stats.output_stats['precipitation']['mean']
        ...     p_std = stats.output_stats['precipitation']['std']
        """
        successful = [r for r in results if r.success]
        n_successful = len(successful)
        n_total = len(results)
        
        # Parameter statistics
        param_stats = self._calculate_parameter_stats(successful)
        
        # Output statistics
        output_stats = self._calculate_output_stats(successful)
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            successful, confidence_levels
        )
        
        return MonteCarloStatistics(
            n_samples=n_total,
            n_successful=n_successful,
            success_rate=n_successful / n_total if n_total > 0 else 0.0,
            parameter_stats=param_stats,
            output_stats=output_stats,
            confidence_intervals=confidence_intervals
        )
    
    def _validate_distributions(
        self,
        distributions: ParamDistributions
    ) -> None:
        """Validate distribution specifications."""
        for param, (dist_name, dist_params) in distributions.items():
            if dist_name not in self.SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Unsupported distribution '{dist_name}' for parameter '{param}'. "
                    f"Supported: {self.SUPPORTED_DISTRIBUTIONS}"
                )
    
    def _sample_parameters(
        self,
        distributions: ParamDistributions
    ) -> Dict[str, np.ndarray]:
        """Sample parameters from distributions."""
        sampled = {}
        
        for param, (dist_name, dist_params) in distributions.items():
            samples = self._sample_distribution(dist_name, dist_params)
            sampled[param] = samples
        
        return sampled
    
    def _sample_distribution(
        self,
        dist_name: str,
        dist_params: Dict[str, Any]
    ) -> np.ndarray:
        """Sample from a single distribution."""
        if dist_name == 'normal':
            return self.rng.normal(
                loc=dist_params.get('loc', 0.0),
                scale=dist_params.get('scale', 1.0),
                size=self.n_samples
            )
        
        elif dist_name == 'uniform':
            return self.rng.uniform(
                low=dist_params.get('low', 0.0),
                high=dist_params.get('high', 1.0),
                size=self.n_samples
            )
        
        elif dist_name == 'lognormal':
            return self.rng.lognormal(
                mean=dist_params.get('mean', 0.0),
                sigma=dist_params.get('sigma', 1.0),
                size=self.n_samples
            )
        
        elif dist_name == 'truncated_normal':
            from scipy import stats as scipy_stats
            low = dist_params.get('low', -np.inf)
            high = dist_params.get('high', np.inf)
            loc = dist_params.get('loc', 0.0)
            scale = dist_params.get('scale', 1.0)
            
            a = (low - loc) / scale if np.isfinite(low) else -np.inf
            b = (high - loc) / scale if np.isfinite(high) else np.inf
            
            return scipy_stats.truncnorm.rvs(
                a, b, loc=loc, scale=scale,
                size=self.n_samples,
                random_state=self.rng
            )
        
        elif dist_name == 'triangular':
            return self.rng.triangular(
                left=dist_params.get('low', 0.0),
                mode=dist_params.get('mode', 0.5),
                right=dist_params.get('high', 1.0),
                size=self.n_samples
            )
        
        elif dist_name == 'beta':
            return self.rng.beta(
                a=dist_params.get('a', 1.0),
                b=dist_params.get('b', 1.0),
                size=self.n_samples
            )
        
        elif dist_name == 'gamma':
            return self.rng.gamma(
                shape=dist_params.get('shape', 1.0),
                scale=dist_params.get('scale', 1.0),
                size=self.n_samples
            )
        
        elif dist_name == 'exponential':
            return self.rng.exponential(
                scale=dist_params.get('scale', 1.0),
                size=self.n_samples
            )
        
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")
    
    def _run_parallel(
        self,
        param_sets: List[Tuple[int, Dict]],
        samples: Optional[Dict]
    ) -> List[MonteCarloResult]:
        """Run simulations in parallel."""
        
        def run_single(args: Tuple[int, Dict]) -> MonteCarloResult:
            sample_id, params = args
            return self._execute_run(sample_id, params, samples)
        
        pbar = tqdm(total=len(param_sets), desc="Monte Carlo") if HAS_TQDM else None
        
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
    ) -> List[MonteCarloResult]:
        """Run simulations sequentially."""
        results = []
        
        iterator = tqdm(param_sets, desc="Monte Carlo") if HAS_TQDM else param_sets
        
        for sample_id, params in iterator:
            result = self._execute_run(sample_id, params, samples)
            results.append(result)
        
        return results
    
    def _execute_run(
        self,
        sample_id: int,
        params: Dict[str, Any],
        samples: Optional[Dict]
    ) -> MonteCarloResult:
        """Execute a single simulation."""
        try:
            result = self.model.run(params, samples)
            return MonteCarloResult(
                sample_id=sample_id,
                params=params,
                result=result,
                success=True,
                error=None
            )
        except Exception as e:
            return MonteCarloResult(
                sample_id=sample_id,
                params=params,
                result=None,
                success=False,
                error=str(e)
            )
    
    def _calculate_parameter_stats(
        self,
        results: List[MonteCarloResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for sampled parameters."""
        if not results:
            return {}
        
        stats = {}
        param_names = list(results[0].params.keys())
        
        for param in param_names:
            values = np.array([r.params.get(param, np.nan) for r in results])
            values = values[np.isfinite(values)]
            
            if len(values) > 0:
                stats[param] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return stats
    
    def _calculate_output_stats(
        self,
        results: List[MonteCarloResult]
    ) -> Dict[str, Any]:
        """Calculate statistics for model outputs."""
        if not results:
            return {}
        
        stats = {}
        
        # Chi-squared statistics
        chi_r2_values = []
        for r in results:
            if hasattr(r.result, 'chi_r2') and np.isfinite(r.result.chi_r2):
                chi_r2_values.append(r.result.chi_r2)
        
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
        
        # Grid-based statistics
        p_grids = [r.result.p_grid for r in results if hasattr(r.result, 'p_grid')]
        if p_grids:
            p_stack = np.stack(p_grids)
            stats['precipitation'] = {
                'mean': float(np.mean(p_stack)),
                'std': float(np.std(p_stack)),
                'mean_grid': np.mean(p_stack, axis=0),
                'std_grid': np.std(p_stack, axis=0)
            }
        
        d2H_grids = [r.result.d2H_grid for r in results if hasattr(r.result, 'd2H_grid')]
        if d2H_grids:
            d2H_stack = np.stack(d2H_grids)
            stats['d2H'] = {
                'mean': float(np.nanmean(d2H_stack)),
                'std': float(np.nanstd(d2H_stack)),
                'mean_grid': np.nanmean(d2H_stack, axis=0),
                'std_grid': np.nanstd(d2H_stack, axis=0)
            }
        
        d18O_grids = [r.result.d18O_grid for r in results if hasattr(r.result, 'd18O_grid')]
        if d18O_grids:
            d18O_stack = np.stack(d18O_grids)
            stats['d18O'] = {
                'mean': float(np.nanmean(d18O_stack)),
                'std': float(np.nanstd(d18O_stack)),
                'mean_grid': np.nanmean(d18O_stack, axis=0),
                'std_grid': np.nanstd(d18O_stack, axis=0)
            }
        
        return stats
    
    def _calculate_confidence_intervals(
        self,
        results: List[MonteCarloResult],
        confidence_levels: Sequence[float]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate confidence intervals for outputs."""
        intervals = {}
        
        # Chi-squared intervals
        chi_r2_values = np.array([
            r.result.chi_r2 for r in results 
            if hasattr(r.result, 'chi_r2') and np.isfinite(r.result.chi_r2)
        ])
        
        if len(chi_r2_values) > 0:
            intervals['chi_r2'] = {}
            for level in confidence_levels:
                alpha = 1 - level
                lower = np.percentile(chi_r2_values, 100 * alpha / 2)
                upper = np.percentile(chi_r2_values, 100 * (1 - alpha / 2))
                intervals['chi_r2'][f'ci_{int(level*100)}'] = np.array([lower, upper])
        
        # Grid-based intervals
        p_grids = [r.result.p_grid for r in results if hasattr(r.result, 'p_grid')]
        if p_grids:
            p_stack = np.stack(p_grids)
            intervals['precipitation'] = {}
            for level in confidence_levels:
                alpha = 1 - level
                lower = np.percentile(p_stack, 100 * alpha / 2, axis=0)
                upper = np.percentile(p_stack, 100 * (1 - alpha / 2), axis=0)
                intervals['precipitation'][f'ci_{int(level*100)}'] = np.stack([lower, upper])
        
        return intervals


def run_monte_carlo_sensitivity(
    model: Any,
    base_params: Dict[str, Any],
    param_distributions: ParamDistributions,
    n_samples: int = 1000,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> Tuple[List[MonteCarloResult], MonteCarloStatistics]:
    """Convenience function for Monte Carlo sensitivity analysis.
    
    This is a high-level function that runs a complete Monte Carlo
    simulation and returns both raw results and statistics.
    
    Parameters
    ----------
    model : object
        OPI model instance
    base_params : Dict[str, Any]
        Base parameter values
    param_distributions : ParamDistributions
        Parameter distributions to sample
    n_samples : int, default=1000
        Number of Monte Carlo samples
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, optional
        Random seed
        
    Returns
    -------
    Tuple[List[MonteCarloResult], MonteCarloStatistics]
        Raw results and computed statistics
        
    Examples
    --------
    >>> from opi.parallel.monte_carlo import run_monte_carlo_sensitivity
    >>> from opi.models import OneWindModel
    >>> 
    >>> model = OneWindModel(topography)
    >>> base = {'kappa': 1000.0, 'tau_c': 1000.0}
    >>> dists = {
    ...     'U': ('normal', {'loc': 10.0, 'scale': 2.0}),
    ...     'M': ('uniform', {'low': 0.5, 'high': 1.0})
    ... }
    >>> 
    >>> results, stats = run_monte_carlo_sensitivity(
    ...     model, base, dists, n_samples=500, n_jobs=4
    ... )
    """
    simulator = MonteCarloSimulator(
        model=model,
        n_samples=n_samples,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    results = simulator.run(param_distributions, base_params)
    statistics = simulator.analyze_results(results)
    
    return results, statistics
