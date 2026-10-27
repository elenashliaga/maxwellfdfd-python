"""
MaxwellFDFD source module.

This module provides electromagnetic sources for exciting fields in the
simulation.

Public API
----------
Source Classes:
    Source : Abstract base class for all sources
    PointSrc : Point source (electric point dipole)
    LineSrc : Line source (constant dipole distribution along a line)
    RectSrc : Rectangular source (constant dipole distribution on a rectangle)
    PlaneSrc : Plane wave source (constant dipole distribution on a plane)
    ModalSrc : Modal source (waveguide mode)
    TFSFPlaneSrc : Total-field/scattered-field plane wave source

Mixin:
    WithBloch : Mixin for sources that support Bloch boundary conditions

Utility Functions:
    assign_source : Assign sources to the grid
    ind_for_loc : Find index of a grid location
"""

# Import base class
from .source import Source

# Import concrete source classes
from .point_src import PointSrc
from .line_src import LineSrc
from .rect_src import RectSrc
from .plane_src import PlaneSrc
from .modal_src import ModalSrc
from .tfsf_plane_src import TFSFPlaneSrc

# Import mixin
from .with_bloch import WithBloch

# Import utility functions
from .utils import assign_source, ind_for_loc

# Export public API
__all__ = [
    # Base class
    'Source',
    # Concrete source classes
    'PointSrc',
    'LineSrc',
    'RectSrc',
    'PlaneSrc',
    'ModalSrc',
    'TFSFPlaneSrc',
    # Mixin
    'WithBloch',
    # Utility functions
    'assign_source',
    'ind_for_loc',
]

__version__ = '0.1.0'