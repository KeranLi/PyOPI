"""
Persistent Path Management

Provides persistent storage of file paths across function calls.
Matches MATLAB's fnPersistentPath.m
"""

import os
from typing import Optional


class PersistentPathManager:
    """
    Manager for persistent file paths.
    
    Maintains a persistent path that survives across function calls
    and can be used for default file dialogs.
    """
    
    _instance = None
    _persistent_path: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._persistent_path = os.getcwd()
        return cls._instance
    
    def get_path(self) -> str:
        """
        Get the current persistent path.
        
        Returns
        -------
        path : str
            Current persistent path (no trailing slash)
        """
        if self._persistent_path is None:
            self._persistent_path = os.getcwd()
        return self._persistent_path.rstrip('/\\')
    
    def set_path(self, new_path: str) -> None:
        """
        Set a new persistent path.
        
        Parameters
        ----------
        new_path : str
            New path to store (trailing slashes removed)
        """
        # Remove trailing slashes
        self._persistent_path = new_path.rstrip('/\\')
    
    def join(self, *paths: str) -> str:
        """
        Join paths with persistent path.
        
        Parameters
        ----------
        *paths : str
            Path components to join
        
        Returns
        -------
        full_path : str
            Joined path
        """
        return os.path.join(self.get_path(), *paths)


# Global instance
_path_manager = PersistentPathManager()


def fn_persistent_path(new_path: Optional[str] = None) -> str:
    """
    Get or set persistent path.
    
    Matches MATLAB's fnPersistentPath function.
    
    Parameters
    ----------
    new_path : str, optional
        If provided, set this as the new persistent path.
        If None, return the current persistent path.
    
    Returns
    -------
    path : str
        Current persistent path (no trailing slash)
    
    Examples
    --------
    >>> # Get current persistent path
    >>> path = fn_persistent_path()
    
    >>> # Set new persistent path
    >>> fn_persistent_path('/path/to/data')
    
    >>> # Use in file dialog
    >>> import tkinter.filedialog as fd
    >>> filepath = fd.askopenfilename(initialdir=fn_persistent_path())
    >>> fn_persistent_path(os.path.dirname(filepath))
    """
    manager = _path_manager
    
    if new_path is None:
        return manager.get_path()
    else:
        manager.set_path(new_path)
        return manager.get_path()


def get_data_path() -> str:
    """Get default data path relative to persistent path."""
    return os.path.join(fn_persistent_path(), 'data')


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, creating if necessary.
    
    Parameters
    ----------
    path : str
        Directory path
    
    Returns
    -------
    path : str
        Same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path
