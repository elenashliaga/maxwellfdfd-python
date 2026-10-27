"""
MaxwellFDFD base module.

This module provides fundamental enumerations, classes, and utility functions
used throughout the MaxwellFDFD package.

Public API
----------
Enumerations:
    Axis : Cartesian axes (x, y, z)
    BC : Boundary conditions (e=PEC, m=PMC, p=periodic)
    Dir : Directions (h=horizontal, v=vertical)
    GT : Grid types (prim=primary, dual=dual)
    PhysQ : Physical quantities
    PML : PML kinds (sc=stretched-coordinate, u=uniaxial)
    Sign : Signs (n=negative, p=positive)
    FT : Field types (e=E, h=H)

Classes:
    Oscillation : Oscillation parameters (L0, wvlen, omega, eV)
    EquationType : Equation type definition
    PhysC : Physical constants
    PhysUnit : Physical unit handling

Utility Functions:
    chkarg : Argument validation
    create_spdiag : Create sparse diagonal matrix
    expand2mat : Expand to matrix
    expand2row : Expand to row vector
    is2dlike : Check if 2D-like
    isexpandable2mat : Check if expandable to matrix
    isexpandable2row : Check if expandable to row vector
    isphysQcell : Check if physical quantity cell
    istypeof : Check type
    istypesizeof : Check type and size
"""

# Import enumerations
from .axis import Axis
from .bc import BC
from .dir import Dir
from .gt import GT
from .physq import PhysQ
from .pml import PML
from .sign import Sign
from .ft import FT

# Import classes
from .oscillation import Oscillation
from .equation_type import EquationType
from .physc import PhysC
from .physunit import PhysUnit

# Import utility functions (public API)
from .utils import (
    chkarg,
    create_spdiag,
    expand2mat,
    expand2row,
    is2dlike,
    isexpandable2mat,
    isexpandable2row,
    isphysQcell,
    istypeof,
    istypesizeof,
)

# Export public API
__all__ = [
    # Enumerations
    'Axis',
    'BC',
    'Dir',
    'GT',
    'PhysQ',
    'PML',
    'Sign',
    'FT',
    # Classes
    'Oscillation',
    'EquationType',
    'PhysC',
    'PhysUnit',
    # Utility functions
    'chkarg',
    'create_spdiag',
    'expand2mat',
    'expand2row',
    'is2dlike',
    'isexpandable2mat',
    'isexpandable2row',
    'isphysQcell',
    'istypeof',
    'istypesizeof',
]

__version__ = '0.1.0'