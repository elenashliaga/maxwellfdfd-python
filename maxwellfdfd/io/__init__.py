"""
MaxwellFDFD io module.

This module is the top-level orchestration layer for MaxwellFDFD. It
provides the main user-facing entry point :func:`maxwell_run`, along with
data structures, system building, and utility functions for constructing
and solving finite-difference frequency-domain (FDFD) simulations.

The io module depends on ALL other modules:
``base``, ``shape``, ``source``, ``grid``, ``material``, and ``diff``.

Public API
----------
Main Interface:
    maxwell_run : Main solver entry point
    build_system : Build FDFD system from parameter groups

Data Structures:
    EMObject : Electromagnetic object (Shape + Material)
    Scalar2d : 2D scalar field data structure
    Scalar3d : 3D scalar field data structure
    ProgMark : Progress/timing marker

Vector Utilities:
    flip_array : Flip array along all dimensions
    flip_vec : Flip each component of a cell vector
    div_vec : Element-wise division of cell vectors
    mult_vec : Element-wise multiplication of cell vectors
    neg_vec : Negate each component of a cell vector

General Utilities:
    array2scalar : Convert raw array to Scalar2d/Scalar3d
    cell2array : Stack cell/dict values into ND+1 array
    collapse_complex : Collapse interleaved real/imag to complex
    expand_complex : Expand complex to interleaved real/imag
    expand_node_array : Extend eps/mu to ghost points
    real_cell : Take real part of each cell element
    reordering_indices : Generate bandwidth-reducing permutation
    slice_scalar3d : Slice Scalar3d to Scalar2d
    transform_gt : Transform grid type of scalar field
    vec2amp : Compute vector field amplitude

Grid Utilities:
    generate_s_factor : Generate PML stretched-coordinate s-factors
    read_grid : Read grid from HDF5 file

File I/O:
    read_output : Read E/H solution from HDF5 files
    write_input : Write simulation input to HDF5 file
    fds : Server-based iterative solver interface

Examples
--------
Run a simple FDFD simulation:

>>> from maxwellfdfd.base import Axis, BC
>>> from maxwellfdfd.shape import Box
>>> from maxwellfdfd.source import PointSrc
>>> from maxwellfdfd.io import maxwell_run
>>> E, H, objs, srcs, extra = maxwell_run(
...     'OSC', 1e-9, 1550,
...     'DOM', ['Palik/SiO2', 'none'],
...         [[-700, 700], [-600, 600], [-200, 1700]], 20, BC.P, 200,
...     'OBJ',
...         ['Palik/SiO2', 'none'],
...         Box([[-50, 50], [-50, 50], [-200, 1700]], [2, 2, 20]),
...     'SRCJ', PointSrc(Axis.X, [0, 0, 200]),
... )

Inspect the domain without solving:

>>> E, H, objs, srcs, extra = maxwell_run(
...     'OSC', 1e-9, 1550,
...     'DOM', ...,
...     'OBJ', ...,
...     inspect_only=True,
... )
"""

# Main interface
from .maxwell_run import maxwell_run
from .build_system import build_system

# Data structures
from .em_object import EMObject
from .scalar2d import Scalar2d
from .scalar3d import Scalar3d
from .progmark import ProgMark

# Vector utilities
from .vec_utils import (
    flip_array,
    flip_vec,
    div_vec,
    mult_vec,
    neg_vec,
)

# General utilities
from .utils import (
    array2scalar,
    cell2array,
    collapse_complex,
    expand_complex,
    expand_node_array,
    real_cell,
    reordering_indices,
    slice_scalar3d,
    transform_gt,
    vec2amp,
)

# Grid utilities
from .grid_utils import (
    generate_s_factor,
    read_grid,
)

# File I/O
from .io_utils import (
    read_output,
    write_input,
    fds,
)

# Define public API
__all__ = [
    # Main interface
    'maxwell_run',
    'build_system',
    # Data structures
    'EMObject',
    'Scalar2d',
    'Scalar3d',
    'ProgMark',
    # Vector utilities
    'flip_array',
    'flip_vec',
    'div_vec',
    'mult_vec',
    'neg_vec',
    # General utilities
    'array2scalar',
    'cell2array',
    'collapse_complex',
    'expand_complex',
    'expand_node_array',
    'real_cell',
    'reordering_indices',
    'slice_scalar3d',
    'transform_gt',
    'vec2amp',
    # Grid utilities
    'generate_s_factor',
    'read_grid',
    # File I/O
    'read_output',
    'write_input',
    'fds',
]

__version__ = '0.1.0'
