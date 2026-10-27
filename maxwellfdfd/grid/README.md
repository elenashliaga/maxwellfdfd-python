# Grid Module

This module provides staggered grid structures for finite-difference frequency-domain (FDFD) simulations of Maxwell's equations. It implements the Yee grid scheme with support for 1D, 2D, and 3D grids, including PML regions and various boundary conditions.

## Overview

The grid module is responsible for:
- Defining staggered Yee grids (primary and dual grid points)
- Generating adaptive grids that respect material boundaries and source locations
- Supporting PML (Perfectly Matched Layer) regions
- Handling various boundary conditions (PEC, PMC, periodic)
- Providing grid locations and cell sizes for field discretization

## Files

### Grid Classes

- **[`grid1d.py`](grid1d.py)**: [`Grid1d`](grid1d.py:Grid1d) class for 1D staggered grids
  - Primary and dual grid points
  - PML region support
  - Boundary condition handling
  - Ghost vertices for boundary calculations

- **[`grid2d.py`](grid2d.py)**: [`Grid2d`](grid2d.py:Grid2d) class for 2D staggered grids
  - Created by slicing a [`Grid3d`](grid3d.py:Grid3d) along a normal axis
  - Horizontal and vertical components
  - Inherits properties from component 1D grids

- **[`grid3d.py`](grid3d.py)**: [`Grid3d`](grid3d.py:Grid3d) class for 3D staggered grids
  - Composed of three [`Grid1d`](grid1d.py:Grid1d) instances (x, y, z axes)
  - Supports different PML thicknesses per axis and per end
  - Supports different boundary conditions per axis

### Grid Generation Utilities

- **[`complete_lprim1d.py`](complete_lprim1d.py)**: Complete 1D primary grid by filling gaps
  - [`complete_lprim1d()`](complete_lprim1d.py:complete_lprim1d): Main function to complete partial grids
  - Uses geometric progression for smooth transitions
  - Preserves provided subgrids for material/source alignment

- **[`generate_lprim1d_part.py`](generate_lprim1d_part.py)**: Generate 1D primary grid part
  - [`generate_lprim1d_part()`](generate_lprim1d_part.py:generate_lprim1d_part): Generates partial grids from domain and intervals
  - Respects PML regions and material boundaries
  - Creates consistent cell sizes on both sides of primary nodes

- **[`generate_lprim3d.py`](generate_lprim3d.py)**: Generate 3D grid
  - [`generate_lprim3d()`](generate_lprim3d.py:generate_lprim3d): Main 3D grid generation function
  - Supports uniform and adaptive grid generation
  - Collects intervals from shapes to avoid redundancy
  - Preserves source grid points

## Key Concepts

### Staggered Yee Grid

The grid implements a staggered Yee scheme where:
- **Primary grid points** (GT.PRIM): Store electric field components (E_x, E_y, E_z)
- **Dual grid points** (GT.DUAL): Store magnetic field components (H_x, H_y, H_z)

Primary grid points are at integer indices, while dual grid points are at half-integer indices.

### Grid Locations

Grid locations are stored as dictionaries with enum keys:
- For 1D: `Dict[GT, np.ndarray]` - maps grid type to location array
- For 2D: `Dict[Tuple[Dir, GT], np.ndarray]` - maps (direction, grid type) to location array
- For 3D: `Dict[Tuple[Axis, GT], np.ndarray]` - maps (axis, grid type) to location array

### PML Regions

PML (Perfectly Matched Layer) regions can be specified at both ends of each axis:
- `Npml`: Number of primary grid cells inside PML
- `lpml`: Location of PML interface
- `Lpml`: Thickness of PML region

### Boundary Conditions

Supported boundary conditions:
- **PEC** (Perfect Electric Conductor): E_tangential = 0 at boundary
- **PMC** (Perfect Magnetic Conductor): H_tangential = 0 at boundary
- **P** (Periodic): Fields wrap around at boundary

## Usage Examples

### Creating a 1D Grid

```python
from maxwellfdfd.base import Axis, BC, PhysUnit
from maxwellfdfd.grid import Grid1d
import numpy as np

unit = PhysUnit()
lprim = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
Npml = np.array([5, 5])  # 5 cells at each end
bc = BC.P  # Periodic boundary condition

grid1d = Grid1d(Axis.X, unit, lprim, Npml, bc)
print(f"Number of cells: {grid1d.N}")
print(f"Domain length: {grid1d.L}")
print(f"Primary grid locations: {grid1d.l[0]}")  # GT.PRIM = 0
print(f"Dual grid locations: {grid1d.l[1]}")    # GT.DUAL = 1
```

### Creating a 3D Grid

```python
from maxwellfdfd.base import Axis, BC, PhysUnit
from maxwellfdfd.grid import Grid3d
import numpy as np

unit = PhysUnit()
lprim_cell = {
    Axis.X: np.array([0.0, 0.5, 1.0]),
    Axis.Y: np.array([0.0, 0.5, 1.0]),
    Axis.Z: np.array([0.0, 0.5, 1.0])
}

grid3d = Grid3d(unit, lprim_cell)
print(f"Cell counts: {grid3d.N}")
print(f"Total cells: {grid3d.Ntot}")
print(f"Domain size: {grid3d.L}")
```

### Generating a 3D Grid from Domain

```python
from maxwellfdfd.base import Axis
from maxwellfdfd.shape import Domain, Box
from maxwellfdfd.grid import generate_lprim3d

# Create domain
domain = Domain(Box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1))
Lpml = 0.1
shape_array = []
src_array = []

# Generate uniform grid
lprim_cell, Npml = generate_lprim3d(
    domain, Lpml, shape_array, src_array, withuniform=True
)

print(f"X-axis grid: {lprim_cell[Axis.X]}")
print(f"PML cell counts: {Npml}")
```

### Completing a Partial Grid

```python
from maxwellfdfd.grid import complete_lprim1d
import numpy as np

# Define partial grid with subgrids and target cell sizes
lprim_part_cell = [
    np.array([0.0, 0.5]),  # Subgrid 1
    1.0,                   # Target dl between subgrids
    np.array([10.0, 11.0, 12.0]),  # Subgrid 2
    1.5,                   # Target dl
    np.array([20.5, 21.0]),  # Subgrid 3
    2.0,                   # Target dl
    np.array([30.0, 31.0])   # Subgrid 4
]

# Complete the grid
lprim = complete_lprim1d(lprim_part_cell)
print(f"Completed grid: {lprim}")
print(f"Grid length: {len(lprim)}")
```

## Conversion Notes

### MATLAB to Python Conversion

This module was converted from MATLAB with the following key changes:

1. **Indexing**: All 1-based MATLAB indexing converted to 0-based Python indexing
2. **Cell Arrays**: MATLAB cell arrays converted to Python dictionaries with enum keys
3. **Properties**: MATLAB properties converted to Python `@property` decorators
4. **Validation**: MATLAB's `chkarg` replaced with Python `ValueError` exceptions
5. **Root Finding**: MATLAB's `fzero` replaced with `scipy.optimize.brentq`
6. **Array Operations**: MATLAB functions converted to NumPy equivalents

### Key Conversion Decisions

- **Grid Storage**: MATLAB cell arrays like `l{GT.prim}` become Python dictionaries with GT enum keys
- **Enum Access**: MATLAB's `Axis.X` becomes Python's `Axis.X` (same syntax)
- **Iteration**: MATLAB's `for w = Axis.elems` becomes Python's `for axis in Axis.elems()`
- **Array Slicing**: MATLAB's `lprim(1:end-1)` becomes Python's `lprim[:-1]`
- **Boolean Indexing**: MATLAB's `find(lprim < val)` becomes Python's `lprim[lprim < val]`

## Dependencies

### From maxwellfdfd.base
- [`Axis`](../base/axis.py): Cartesian axes enumeration
- [`BC`](../base/bc.py): Boundary conditions enumeration
- [`Sign`](../base/sign.py): Signs enumeration
- [`GT`](../base/gt.py): Grid types enumeration
- [`Dir`](../base/dir.py): Directions enumeration
- [`PhysUnit`](../base/physunit.py): Physical unit class
- [`expand2mat`](../base/utils.py): Expand to matrix
- [`expand2row`](../base/utils.py): Expand to row vector

### From maxwellfdfd.shape
- [`Interval`](../shape/interval.py): 1D interval class
- [`Domain`](../shape/domain.py): Domain class
- [`Shape`](../shape/shape.py): Base shape class (for type hints)

### External Libraries
- `numpy`: Numerical operations
- `scipy.optimize`: Root finding (brentq)

## Testing

To test the grid module:

```python
from maxwellfdfd.base import Axis, BC, PhysUnit
from maxwellfdfd.grid import Grid1d, Grid3d
import numpy as np

# Test 1D grid
unit = PhysUnit()
lprim = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
Npml = np.array([5, 5])
grid1d = Grid1d(Axis.X, unit, lprim, Npml, BC.P)
assert grid1d.N == 4
assert grid1d.L == 2.0

# Test 3D grid
lprim_cell = {
    Axis.X: np.array([0.0, 0.5, 1.0]),
    Axis.Y: np.array([0.0, 0.5, 1.0]),
    Axis.Z: np.array([0.0, 0.5, 1.0])
}
grid3d = Grid3d(unit, lprim_cell)
assert np.all(grid3d.N == [2, 2, 2])
assert grid3d.Ntot == 8

print("All tests passed!")
```

## References

- Yee, K. S. (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media." IEEE Transactions on Antennas and Propagation.
- Berenger, J. P. (1994). "A perfectly matched layer for the absorption of electromagnetic waves." Journal of Computational Physics.

## See Also

- [Base Module](../base/README.md): Fundamental enumerations and utilities
- [Shape Module](../shape/README.md): Geometric shapes for defining simulation domains
- [Source Module](../source/README.md): Electromagnetic sources for exciting fields