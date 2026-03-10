"""
Exception hierarchy for OPI.

Provides structured error handling with specific exception types
for different error conditions.
"""


class OPIError(Exception):
    """Base exception for all OPI errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class CalculationError(OPIError):
    """Raised when a calculation fails."""
    pass


class ValidationError(OPIError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(OPIError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, issues: list = None):
        super().__init__(message)
        self.issues = issues or []
    
    def __str__(self):
        if self.issues:
            issues_str = "\n".join(f"  - {issue}" for issue in self.issues)
            return f"{self.message}\nIssues:\n{issues_str}"
        return self.message


class OptimizationError(OPIError):
    """Raised when optimization fails to converge or encounters an error."""
    
    def __init__(self, message: str, iteration: int = None, 
                 current_value: float = None):
        super().__init__(message)
        self.iteration = iteration
        self.current_value = current_value


class DataError(OPIError):
    """Raised when data loading or processing fails."""
    pass


class FileFormatError(DataError):
    """Raised when file format is not supported or invalid."""
    pass


class ConvergenceError(OptimizationError):
    """Raised when iterative solver fails to converge."""
    pass


class SingularityError(CalculationError):
    """Raised when numerical singularity is encountered."""
    pass


class GridError(OPIError):
    """Raised when grid operations fail."""
    pass
