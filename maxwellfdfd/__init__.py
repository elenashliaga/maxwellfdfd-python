"""
MaxwellFDFD: A Python library for finite-difference frequency-domain (FDFD)
electromagnetic simulations.

This is the Python port of the MATLAB MaxwellFDFD library. It provides:

- ``base``    : Fundamental enumerations (Axis, BC, GT, Sign, ...) and utilities.
- ``shape``   : Geometric shapes (Box, Sphere, Cylinder, ...) defined via level sets.
- ``source``  : Electromagnetic sources (PointSrc, PlaneSrc, ...).
- ``grid``    : Staggered Yee grids (Grid1d, Grid2d, Grid3d).
- ``material``: Materials and material assignment to grid nodes.
- ``diff``    : Finite-difference operators (curl, div, PML) and linear solvers.
- ``io``      : Top-level orchestration and the :func:`maxwell_run` entry point.

The simplest way to run a simulation is through :func:`maxwellfdfd.io.maxwell_run`.

Example
-------
>>> from maxwellfdfd.base import Axis, BC
>>> from maxwellfdfd.shape import Box
>>> from maxwellfdfd.source import PointSrc
>>> from maxwellfdfd.io import maxwell_run
"""

# Submodules are imported in dependency order to avoid circular imports.
from . import base
from . import shape
from . import source
from . import grid
from . import material
from . import diff
from . import io

# Re-export the most commonly used public symbols for convenience.
from .base import (
    Axis,
    BC,
    Dir,
    GT,
    Sign,
    FT,
    PML,
    PhysQ,
    Oscillation,
    EquationType,
    PhysC,
    PhysUnit,
)
from .shape import (
    Shape,
    Box,
    Rectangle,
    Sphere,
    Ellipsoid,
    Domain,
)
from .source import (
    Source,
    PointSrc,
    LineSrc,
    RectSrc,
    PlaneSrc,
)
from .grid import (
    Grid1d,
    Grid2d,
    Grid3d,
)
from .material import (
    Material,
    assign_material_node,
)
from .diff import (
    create_curls,
    create_divs,
    create_Ds,
    create_Dw,
    MatrixEquation,
    solve_eq_direct,
    solve_eq_iterative,
)
from .io import maxwell_run

__version__ = "0.1.0"

__all__ = [
    # submodules
    "base",
    "shape",
    "source",
    "grid",
    "material",
    "diff",
    "io",
    # base
    "Axis",
    "BC",
    "Dir",
    "GT",
    "Sign",
    "FT",
    "PML",
    "PhysQ",
    "Oscillation",
    "EquationType",
    "PhysC",
    "PhysUnit",
    # shape
    "Shape",
    "Box",
    "Rectangle",
    "Sphere",
    "Ellipsoid",
    "Domain",
    # source
    "Source",
    "PointSrc",
    "LineSrc",
    "RectSrc",
    "PlaneSrc",
    # grid
    "Grid1d",
    "Grid2d",
    "Grid3d",
    # material
    "Material",
    "assign_material_node",
    # diff
    "create_curls",
    "create_divs",
    "create_Ds",
    "create_Dw",
    "MatrixEquation",
    "solve_eq_direct",
    "solve_eq_iterative",
    # io
    "maxwell_run",
]
