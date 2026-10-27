"""
MaxwellFDFD grid module.

This module provides staggered grid structures for finite-difference discretization
of Maxwell's equations. The grid module implements the Yee grid scheme with support
for 1D, 2D, and 3D grids, including PML regions and various boundary conditions.

Public API
----------
Grid Classes:
    Grid1d : 1D staggered grid
    Grid2d : 2D staggered grid (inherits from Grid1d)
    Grid3d : 3D staggered grid (inherits from Grid2d)

Grid Generation Utilities:
    complete_lprim1d : Complete 1D primary grid by filling gaps
    generate_lprim1d_part : Generate 1D primary grid part
    generate_lprim3d : Generate 3D grid

Key Concepts
-----------
The grid module implements a staggered Yee grid where:
- Primary grid points store field components (E_x, E_y, E_z)
- Dual grid points store field components (H_x, H_y, H_z)
- Grid locations are stored as dictionaries with GT enum keys
- PML regions can be specified at both ends of each axis
- Boundary conditions are specified per axis (PEC, PMC, or periodic)

The grid generation algorithm creates smoothly varying grids that:
1. Respect material boundaries
2. Include source locations
3. Maintain smooth transitions between regions
4. Support PML regions with specified thickness

Examples
--------
>>> from maxwellfdfd.base import Axis, BC, Sign, PhysUnit
>>> from maxwellfdfd.grid import Grid1d
>>> unit = PhysUnit()
>>> lprim = [0.0, 0.5, 1.0, 1.5, 2.0]
>>> Npml = [5, 5]  # 5 cells at each end
>>> bc = BC.P
>>> grid1d = Grid1d(Axis.X, unit, lprim, Npml, bc)
>>> grid1d.N
4
>>> grid1d.L
2.0
"""

# Import grid classes
from .grid1d import Grid1d
from .grid2d import Grid2d
from .grid3d import Grid3d

# Import grid generation utilities
from .complete_lprim1d import complete_lprim1d
from .generate_lprim1d_part import generate_lprim1d_part
from .generate_lprim3d import generate_lprim3d

# Export public API
__all__ = [
    # Grid classes
    'Grid1d',
    'Grid2d',
    'Grid3d',
    # Grid generation utilities
    'complete_lprim1d',
    'generate_lprim1d_part',
    'generate_lprim3d',
]

__version__ = '0.1.0'