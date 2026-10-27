# MaxwellFDFD Shape Module

## Overview

The shape module provides geometric shapes for defining simulation domains and objects in MaxwellFDFD. Shapes are defined using level set functions, where a level set function takes a position (x, y, z) and returns a real number. For points in the interior, boundary, and exterior of the shape, LSF > 0, LSF == 0, and LSF < 0 respectively.

## Conversion Summary

This module was converted from MATLAB to Python following the architecture design documented in `ARCHITECTURE.md`.

### Files Created

**Base Classes (3 files):**
- `shape.py` - Base Shape class (ABC) with level set function support
- `interval.py` - Interval class for 1D intervals
- `zero_vol_shape.py` - ZeroVolShape base class for shapes without volume

**Basic Shapes (6 files):**
- `point.py` - Point shape
- `line.py` - Line shape
- `plane.py` - Plane shape
- `box.py` - Box shape
- `rectangle.py` - Rectangle shape
- `sphere.py` - Sphere shape

**Cylinder Shapes (7 files):**
- `generic_cylinder.py` - Generic cylinder base class (ABC)
- `circular_cylinder.py` - Circular cylinder
- `elliptic_cylinder.py` - Elliptic cylinder
- `sectoral_cylinder.py` - Sectoral cylinder
- `segmental_cylinder.py` - Segmental cylinder
- `circular_shell_cylinder.py` - Circular shell cylinder
- `polygonal_cylinder.py` - Polygonal cylinder

**Other Shapes (2 files):**
- `ellipsoid.py` - Ellipsoid shape
- `domain.py` - Domain shape (special Box subclass)

**Utilities (1 file):**
- `periodize_shape.py` - Periodize shape utility

**Module Files (1 file):**
- `__init__.py` - Module initialization with exports

## Class Hierarchy

```
Shape (ABC)
├── Interval
├── Box
│   └── Domain
├── Ellipsoid
│   └── Sphere
├── ZeroVolShape (ABC)
│   ├── Point
│   ├── Line
│   ├── Plane
│   └── Rectangle
└── GenericCylinder (ABC)
    ├── EllipticCylinder
    │   └── CircularCylinder
    ├── SectoralCylinder
    ├── SegmentalCylinder
    ├── CircularShellCylinder
    └── PolygonalCylinder
```

## Key Conversion Decisions

### 1. Level Set Functions
- MATLAB function handles became Python callables
- Type hint: `Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]`
- All level set functions are vectorized to handle array inputs

### 2. Cell Arrays
- MATLAB cell arrays became Python dictionaries with enum keys
- Example: `lprim_cell` is `Dict[Axis, np.ndarray]`

### 3. Properties
- MATLAB properties became `@property` decorators
- Immutable properties use read-only properties (no setter)

### 4. Indexing
- Converted 1-based indexing to 0-based indexing
- `Axis.elems(i)` became `Axis.elems()[i]` (already 0-based in Python)
- `Sign.n` maps to index 0, `Sign.p` maps to index 1

### 5. Enumerations
- Used enums from `maxwellfdfd.base`:
  - `Axis` for x, y, z axes
  - `Sign` for negative, positive
  - `Dir` for horizontal, vertical

### 6. Validation
- Used `chkarg()` from `maxwellfdfd.base.utils`
- Converted MATLAB validation messages to Python format

### 7. NumPy Integration
- Used NumPy arrays for all numerical operations
- Replaced MATLAB functions:
  - `Inf` → `np.inf`
  - `NaN` → `np.nan`
  - `ndgrid` → `np.meshgrid` with `indexing='ij'`
  - `gradient` → `np.gradient`
  - `linspace` → `np.linspace`
  - `mean` → `np.mean`
  - `diff` → `np.diff`

### 8. Type Hints
- Added comprehensive type hints to all functions and methods
- Used `Union` for multiple possible types
- Used `Optional` for optional parameters

### 9. Docstrings
- Used NumPy docstring format
- Included Parameters, Returns, Raises, and Examples sections

### 10. Abstract Methods
- Used `@abstractmethod` decorator for abstract methods
- Imported from `abc` module

## Dependencies

### From Base Module
- `maxwellfdfd.base.Axis` - Axis enumeration
- `maxwellfdfd.base.Sign` - Sign enumeration
- `maxwellfdfd.base.Dir` - Direction enumeration
- `maxwellfdfd.base.utils.chkarg` - Argument validation
- `maxwellfdfd.base.utils.expand2row` - Expand to row vector

### Standard Library
- `abc` - Abstract base classes
- `typing` - Type hints
- `functools` - Function utilities

### Third-Party Libraries
- `numpy` - Numerical operations

## Usage Examples

### Creating a Box

```python
import numpy as np
from maxwellfdfd.shape import Box

# Create a box with bounds [-100, 100] x [-50, 50] x [0, 20]
box = Box(np.array([[-100, 100], [-50, 50], [0, 20]]))

# Check if a point is in the box
print(box.contains(0, 0, 10))  # True
print(box.contains(200, 0, 10))  # False

# Get box properties
print(box.bound)  # [[-100, 100], [-50, 50], [0, 20]]
print(box.cb_center)  # [0, 0, 10]
print(box.L)  # [200, 100, 20]
```

### Creating a Sphere

```python
import numpy as np
from maxwellfdfd.shape import Sphere

# Create a sphere centered at origin with radius 100
sphere = Sphere(np.array([0, 0, 0]), 100)

# Check if a point is in the sphere
print(sphere.contains(0, 0, 0))  # True
print(sphere.contains(100, 0, 0))  # False
```

### Creating a Circular Cylinder

```python
import numpy as np
from maxwellfdfd.base import Axis
from maxwellfdfd.shape import CircularCylinder

# Create a circular cylinder along z-axis
cylinder = CircularCylinder(
    Axis.Z,  # axis
    100,    # height
    np.array([0, 0, 50]),  # center
    50      # radius
)

# Get cylinder properties
print(cylinder.radius)  # 50.0
print(cylinder.height)  # 100.0
```

### Creating a Periodic Array of Shapes

```python
import numpy as np
from maxwellfdfd.base import Axis
from maxwellfdfd.shape import Box, CircularCylinder, periodize_shape

# Create a slab
t = 500  # thickness
slab = Box(np.array([[-5000, 5000], [-3000, 3000], [0, t]]))

# Create an array of circular cylinders
r = 100  # radius
rod = CircularCylinder(Axis.Z, t, np.array([0, 0, t/2]), r)
primitive_vecs = [
    np.array([400, 0, 0]),
    np.array([0, 400, 0]),
    np.array([0, 0, t])
]
rod_array = periodize_shape(rod, primitive_vecs, slab)
print(f"Created {len(rod_array)} rods")
```

## Running Tests

To test the shape module, run:

```bash
python3 -m pytest tests/test_shape/
```

## Notes

1. The Pylance errors about unknown imports from `maxwellfdfd.base` are expected during development and will be resolved when the base module is properly installed.

2. All shapes use level set functions for geometric representation, which allows for:
   - Efficient point-in-shape testing
   - Smooth boundary representation
   - Easy combination of shapes through level set operations

3. Zero-volume shapes (Point, Line, Plane, Rectangle) use a modified level set function to help with visualization.

4. The `periodize_shape` function creates periodic arrays of shapes by translating a base shape according to primitive lattice vectors.

5. All shapes support the following methods:
   - `contains(x, y, z)` - Check if point(s) are in the shape
   - `circumbox_contains(x, y, z)` - Check if point(s) are in the circumbox
   - Properties: `bound`, `cb_center`, `L`, `dl_max`, `lprim`

## Version History

- 0.1.0 (2026-07-11): Initial conversion from MATLAB to Python