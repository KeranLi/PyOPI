"""
Base classes for OPI components.
"""

import logging
from abc import ABC
from typing import Optional, TypeVar, Generic
from ..models.config import OPIConfig, DEFAULT_CONFIG


T = TypeVar('T')


class OPIBase(ABC):
    """
    Abstract base class for all OPI components.
    
    Provides common functionality like logging and configuration access.
    
    Attributes:
        config: OPI configuration
        logger: Logger instance
    """
    
    def __init__(self, config: Optional[OPIConfig] = None):
        """
        Initialize base component.
        
        Args:
            config: OPI configuration. If None, uses default config.
        """
        self._config = config or DEFAULT_CONFIG
        self._logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
    
    @property
    def config(self) -> OPIConfig:
        """Get configuration."""
        return self._config
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger."""
        return self._logger
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self._config.logging
        
        # Set level
        level = getattr(logging, log_config.level.upper(), logging.INFO)
        self._logger.setLevel(level)
        
        # Remove existing handlers
        self._logger.handlers = []
        
        # Add handlers based on config
        if log_config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter(log_config.format)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        
        if log_config.log_to_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config.log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(level)
            formatter = logging.Formatter(log_config.format)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
    
    def log_info(self, message: str):
        """Log info message."""
        self._logger.info(message)
    
    def log_debug(self, message: str):
        """Log debug message."""
        self._logger.debug(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self._logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        self._logger.error(message)


class ContextManagerMixin:
    """Mixin for context manager support."""
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, cleanup resources."""
        self.cleanup()
        return False
    
    def cleanup(self):
        """Cleanup resources. Override in subclasses."""
        pass


class ObservableMixin:
    """Mixin for observer pattern support."""
    
    def __init__(self):
        self._observers = []
    
    def add_observer(self, observer):
        """Add an observer."""
        self._observers.append(observer)
    
    def remove_observer(self, observer):
        """Remove an observer."""
        self._observers.remove(observer)
    
    def notify_observers(self, event: str, data: dict):
        """Notify all observers of an event."""
        for observer in self._observers:
            observer.on_event(event, data)
