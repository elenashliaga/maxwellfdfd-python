"""
MaxwellFDFD diff module.

This module provides finite-difference operators and equation solvers for
the finite-difference frequency-domain (FDFD) method. It implements curl
and divergence operators, matrix equations, and both direct and iterative
solvers for Maxwell's equations.

Public API
----------
Classes:
    MatrixEquation : Matrix equation class for FDFD system

Functions:
    create_curls : Create curl operators for Maxwell's equations
    create_divs : Create divergence operators
    create_Ds : Create Ds operators (stretched coordinate PML)
    create_Dw : Create Dw operators (frequency domain)
    create_masks : Create mask operators for boundary conditions
    eh_from_eh : Convert E and H fields from e and h fields
    solve_eq_direct : Direct solver for FDFD equations
    solve_eq_iterative : Iterative solver for FDFD equations

Examples
--------
Create curl operators:

>>> import numpy as np
>>> from maxwellfdfd.base import Axis, BC, GT
>>> from maxwellfdfd.grid import Grid3d
>>> from maxwellfdfd.base import PhysUnit
>>> from maxwellfdfd.diff import create_curls
>>> unit = PhysUnit()
>>> lprim = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
>>> Npml = [[1, 1], [1, 1], [1, 1]]
>>> bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
>>> grid3d = Grid3d(unit, lprim, Npml, bc)
>>> Ce, Cm = create_curls(GT.Prim, None, grid3d)

Solve FDFD equations:

>>> from maxwellfdfd.base import FT, PML, EquationType
>>> from maxwellfdfd.diff import solve_eq_direct
>>> eqtype = EquationType(FT.E, GT.Prim)
>>> eps_cell = {Axis.X: np.ones((2, 2, 2)), Axis.Y: np.ones((2, 2, 2)), Axis.Z: np.ones((2, 2, 2))}
>>> mu_cell = {Axis.X: np.ones((2, 2, 2)), Axis.Y: np.ones((2, 2, 2)), Axis.Z: np.ones((2, 2, 2))}
>>> s_factor_cell = {}
>>> J_cell = {Axis.X: np.zeros((2, 2, 2)), Axis.Y: np.zeros((2, 2, 2)), Axis.Z: np.zeros((2, 2, 2))}
>>> M_cell = {Axis.X: np.zeros((2, 2, 2)), Axis.Y: np.zeros((2, 2, 2)), Axis.Z: np.zeros((2, 2, 2))}
>>> omega = 2 * np.pi * 3e8 / 1.55e-6
>>> E, H = solve_eq_direct(eqtype, PML.SC, omega, eps_cell, mu_cell, s_factor_cell, J_cell, M_cell, grid3d)

Notes
-----
The diff module implements the core finite-difference operators used in
FDFD simulations. These operators discretize the curl and divergence
operations in Maxwell's equations on a staggered Yee grid.

Key features:
- Sparse matrix operators for efficient computation
- Support for PML (Perfectly Matched Layers) using stretched coordinates
- Boundary condition handling (PEC, PMC, periodic)
- Both direct and iterative solvers for flexibility
- Matrix-free operators for large-scale problems

For more details, see the individual function documentation.
"""

# Import public classes and functions
from .matrix_equation import MatrixEquation
from .create_curls import create_curls
from .create_divs import create_divs
from .create_Ds import create_Ds
from .create_Dw import create_Dw
from .create_masks import create_masks
from .eh_from_eh import eh_from_eh
from .solve_eq_direct import solve_eq_direct
from .solve_eq_iterative import solve_eq_iterative

# Define public API
__all__ = [
    # Classes
    'MatrixEquation',
    # FDFD operators
    'create_curls',
    'create_divs',
    'create_Ds',
    'create_Dw',
    'create_masks',
    # Field conversion
    'eh_from_eh',
    # Solvers
    'solve_eq_direct',
    'solve_eq_iterative',
]

# Module version
__version__ = '0.1.0'