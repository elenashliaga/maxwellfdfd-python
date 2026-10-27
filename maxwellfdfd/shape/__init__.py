"""
MaxwellFDFD shape module.

This module provides geometric shapes for defining simulation domains and objects.
Shapes are defined using level set functions, where a level set function takes a
position (x, y, z) and returns a real number. For points in the interior, boundary,
and exterior of the shape, LSF > 0, LSF == 0, and LSF < 0 respectively.

Public API
----------
Base Classes:
    Shape : Abstract base class for all shapes
    Interval : 1D interval class
    ZeroVolShape : Base class for shapes without volume
    GenericCylinder : Abstract base class for cylinders

Basic Shapes:
    Box : 3D box shape
    Rectangle : 2D rectangle shape
    Point : Point shape
    Line : Line shape
    Plane : Plane shape
    Sphere : Sphere shape
    Ellipsoid : Ellipsoid shape
    Domain : Computation domain shape

Cylinder Shapes:
    CircularCylinder : Cylinder with circular cross section
    EllipticCylinder : Cylinder with elliptical cross section
    SectoralCylinder : Cylinder with sectoral cross section
    SegmentalCylinder : Cylinder with segmental cross section
    CircularShellCylinder : Hollow cylindrical pipe
    PolygonalCylinder : Cylinder with polygonal cross section

Utilities:
    periodize_shape : Create periodic array of shapes
"""

# Import base classes
from .shape import Shape
from .interval import Interval
from .zero_vol_shape import ZeroVolShape
from .generic_cylinder import GenericCylinder

# Import basic shapes
from .box import Box
from .rectangle import Rectangle
from .point import Point
from .line import Line
from .plane import Plane
from .sphere import Sphere
from .ellipsoid import Ellipsoid
from .domain import Domain

# Import cylinder shapes
from .circular_cylinder import CircularCylinder
from .elliptic_cylinder import EllipticCylinder
from .sectoral_cylinder import SectoralCylinder
from .segmental_cylinder import SegmentalCylinder
from .circular_shell_cylinder import CircularShellCylinder
from .polygonal_cylinder import PolygonalCylinder

# Import utilities
from .periodize_shape import periodize_shape

# Export public API
__all__ = [
    # Base classes
    'Shape',
    'Interval',
    'ZeroVolShape',
    'GenericCylinder',
    # Basic shapes
    'Box',
    'Rectangle',
    'Point',
    'Line',
    'Plane',
    'Sphere',
    'Ellipsoid',
    'Domain',
    # Cylinder shapes
    'CircularCylinder',
    'EllipticCylinder',
    'SectoralCylinder',
    'SegmentalCylinder',
    'CircularShellCylinder',
    'PolygonalCylinder',
    # Utilities
    'periodize_shape',
]

__version__ = '0.1.0'