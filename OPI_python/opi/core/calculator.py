"""
Base calculator class for OPI.
"""

from abc import abstractmethod
from typing import TypeVar, Generic, Optional, List
import time

from .base import OPIBase, ContextManagerMixin
from .exceptions import CalculationError, ValidationError
from ..models.config import OPIConfig
from ..models.domain import Topography, Grid
from ..models.parameters import SampleCollection
from ..models.results import CalculationMetadata


InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class OPICalculator(OPIBase, ContextManagerMixin, Generic[InputType, OutputType]):
    """
    Abstract base class for OPI calculators.
    
    Provides a template method pattern for calculations with:
    - Input validation
    - Pre-processing
    - Core calculation
    - Post-processing
    - Error handling
    
    Type parameters:
        InputType: Type of input parameters
        OutputType: Type of output results
    
    Example:
        class OneWindCalculator(OPICalculator[OneWindParameters, OneWindResults]):
            def _validate_input(self, params):
                # Validate parameters
                pass
            
            def _calculate(self, params):
                # Perform calculation
                pass
    """
    
    def __init__(self, config: Optional[OPIConfig] = None):
        super().__init__(config)
        self._topography: Optional[Topography] = None
        self._samples: Optional[SampleCollection] = None
        self._metadata: Optional[CalculationMetadata] = None
    
    @property
    def topography(self) -> Optional[Topography]:
        """Get loaded topography."""
        return self._topography
    
    @topography.setter
    def topography(self, topo: Topography):
        """Set topography."""
        self._topography = topo
    
    @property
    def samples(self) -> Optional[SampleCollection]:
        """Get loaded samples."""
        return self._samples
    
    @samples.setter
    def samples(self, samples: SampleCollection):
        """Set samples."""
        self._samples = samples
    
    def calculate(self, params: InputType) -> OutputType:
        """
        Execute calculation with full error handling.
        
        This is the template method that orchestrates the calculation.
        Subclasses should override the protected methods, not this one.
        
        Args:
            params: Input parameters
            
        Returns:
            Calculation results
            
        Raises:
            ValidationError: If input is invalid
            CalculationError: If calculation fails
        """
        start_time = time.time()
        self._metadata = CalculationMetadata()
        
        try:
            self.log_info(f"Starting {self.__class__.__name__} calculation")
            
            # 1. Validate input
            self.log_debug("Validating input...")
            self._validate_input(params)
            
            # 2. Pre-processing
            self.log_debug("Pre-processing...")
            self._preprocess(params)
            
            # 3. Core calculation
            self.log_info("Performing core calculation...")
            results = self._calculate(params)
            
            # 4. Post-processing
            self.log_debug("Post-processing...")
            results = self._postprocess(results)
            
            # 5. Finalize
            if hasattr(results, 'finalize'):
                results.finalize()
            
            elapsed = time.time() - start_time
            self.log_info(f"Calculation completed in {elapsed:.2f} seconds")
            
            return results
            
        except ValidationError:
            raise
        except Exception as e:
            raise CalculationError(
                f"Calculation failed: {str(e)}",
                details={'error_type': type(e).__name__}
            ) from e
    
    @abstractmethod
    def _validate_input(self, params: InputType):
        """
        Validate input parameters.
        
        Args:
            params: Input parameters to validate
            
        Raises:
            ValidationError: If parameters are invalid
        """
        pass
    
    def _preprocess(self, params: InputType):
        """
        Pre-processing step. Override if needed.
        
        Args:
            params: Input parameters
        """
        pass
    
    @abstractmethod
    def _calculate(self, params: InputType) -> OutputType:
        """
        Core calculation. Must be implemented by subclasses.
        
        Args:
            params: Input parameters
            
        Returns:
            Calculation results
        """
        pass
    
    def _postprocess(self, results: OutputType) -> OutputType:
        """
        Post-processing step. Override if needed.
        
        Args:
            results: Raw calculation results
            
        Returns:
            Processed results
        """
        return results
    
    def get_summary(self) -> dict:
        """Get summary of last calculation."""
        if self._metadata is None:
            return {}
        
        return {
            'start_time': self._metadata.start_time,
            'end_time': self._metadata.end_time,
            'duration': self._metadata.duration_seconds
        }


class CalculatorChain(OPICalculator):
    """
    Chain multiple calculators together.
    
    Output of calculator N is input to calculator N+1.
    
    Example:
        chain = CalculatorChain([
            Preprocessor(),
            MainCalculator(),
            Postprocessor()
        ])
        results = chain.calculate(initial_input)
    """
    
    def __init__(self, calculators: List[OPICalculator]):
        """
        Initialize chain.
        
        Args:
            calculators: List of calculators to chain
        """
        super().__init__()
        self._calculators = calculators
    
    def _validate_input(self, params):
        """Validate with first calculator."""
        if self._calculators:
            self._calculators[0]._validate_input(params)
    
    def _calculate(self, params):
        """Execute chain."""
        result = params
        for calc in self._calculators:
            result = calc.calculate(result)
        return result
