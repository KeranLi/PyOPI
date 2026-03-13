"""
Coordinate transformation functions for I/O module.

This module re-exports coordinate functions from the main opi module
to avoid code duplication.
"""

from ..coordinates import lonlat2xy, xy2lonlat

__all__ = ['lonlat2xy', 'xy2lonlat']
