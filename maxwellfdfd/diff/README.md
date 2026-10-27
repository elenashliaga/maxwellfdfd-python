# MaxwellFDFD Diff Module

## Overview

The diff module provides finite-difference operators and equation solvers for the finite-difference frequency-domain (FDFD) method. This module implements the core numerical operators used to discretize Maxwell's equations on a staggered Yee grid.

## Purpose

The diff module is responsible for:

1. **Creating finite-difference operators**: Curl and divergence operators that discretize the spatial derivatives in Maxwell's equations
2. **Handling boundary conditions**: PEC, PMC, and periodic boundary conditions through mask operators
3. **Implementing PML**: Perfectly Matched Layers using stretched-coordinate formulation
4. **Building matrix equations**: Constructing the system matrix A and right-hand side b for A*f = b
5. **Solving equations**: Both direct (sparse LU) and iterative (BiCGSTAB) solvers

## Module Structure

```
diff/
├── __init__.py              # Public API exports
├── _utils.py                # Utility functions (mult_vec, div_vec, reordering_indices)
├── create_Dw.py             # Forward difference matrix
├── create_Ds.py             # Stretched coordinate PML operators
├── create_masks.py          # Mask operators for boundary conditions
├── create_curls.py          # Curl operators for Maxwell's equations
├── create_divs.py           # Divergence operators
├── eh_from_eh.py            # Field conversion
├── matrix_equation.py       # Matrix equation class
├── solve_eq_direct.py       # Direct solver
├── solve_eq_iterative.py    # Iterative solver
└── README.md                # This file
```

## Files and Their Purpose

### Utility Functions

#### `_utils.py`
- `mult_vec()`: Element-wise multiplication of cell arrays
- `div_vec()`: Element-wise division of cell arrays
- `reordering_indices()`: Generate reordering indices for bandwidth reduction

### FDFD Operators

#### `create_Dw.py`
- `create_Dw()`: Create forward difference matrix without dl division
- Handles both 3D (Axis) and 2D (Dir) directions
- Implements symmetry and Bloch boundary factors

#### `create_Ds.py`
- `create_Ds()`: Create forward/backward derivative matrices with PML stretching
- Supports stretched-coordinate PML through dl_factor_cell
- Handles periodic boundary conditions with Bloch phase factors

#### `create_masks.py`
- `create_masks()`: Create mask indices for boundary conditions
- Returns indices rather than diagonal matrices for efficiency
- Handles PEC and PMC boundary masking

#### `create_curls.py`
- `create_curls()`: Create curl operators Ce and Cm for E and H fields
- Constructs curl matrices from derivative matrices
- Applies boundary condition masking

#### `create_divs.py`
- `create_divs()`: Create divergence operators Dive and Divm
- Constructs divergence matrices from derivative matrices
- Used for enforcing divergence-free conditions

### Field Conversion

#### `eh_from_eh.py`
- `eh_from_eh()`: Convert flattened field vectors to structured arrays
- Reshapes vectors into 3D arrays with Axis enum keys
- Useful for visualization and post-processing

### Matrix Equation

#### `matrix_equation.py`
- `MatrixEquation` class: Main class for FDFD system
- Properties: Ce, Cm, eps, mu, j, m, omega, pm, r, s, ft
- Methods:
  - `matrix_op()`: Get matrix operation A and RHS b
  - `matrixfree_op()`: Get matrix-free operators for iterative solving

### Solvers

#### `solve_eq_direct.py`
- `solve_eq_direct()`: Direct solver using sparse LU decomposition
- Suitable for small to medium-sized problems
- Returns E and H fields as structured arrays

#### `solve_eq_iterative.py`
- `solve_eq_iterative()`: Iterative solver using BiCGSTAB
- Suitable for large problems
- Supports zero or random initial guess
- Returns E, H fields, residual, iteration count, and residual history

## Key Conversion Decisions

### 1. Cell Arrays to Dictionaries
MATLAB cell arrays with enum keys were converted to Python dictionaries with enum keys:
```python
# MATLAB: Ds_cell{Axis.x}
# Python: Ds_cell[Axis.X]
```

### 2. 1-based to 0-based Indexing
All array indices were converted from 1-based to 0-based indexing.

### 3. Sparse Matrices
Uses `scipy.sparse` for sparse matrix operations:
```python
import scipy.sparse as sp
Dw = sp.coo_matrix((data, (rows, cols)), shape=(M, M))
```

### 4. Type Hints
Comprehensive type hints were added to all function signatures.

### 5. NumPy Docstrings
All functions and classes use NumPy docstring format.

### 6. Property Decorators
The MatrixEquation class uses `@property` decorators for read-only properties.

### 7. Enum Methods
Enum class methods like `elems()` and `count()` are used as defined in the base module.

### 8. Grid Type Handling
Uses `isinstance()` to check for Grid2d vs Grid3d:
```python
if isinstance(gridnd, Grid2d):
    v = Dir
else:
    v = Axis
```

### 9. Boundary Condition Handling
Uses BC enum values for boundary condition checks:
```python
if bc[w] == BC.P:
    fg = np.exp(-1j * kBloch[w] * L[w])
else:
    fg = 0.0
```

## Dependencies

The diff module depends on:

- `maxwellfdfd.base`: Base enumerations (Axis, BC, GT, Sign, FT, PML) and utilities
- `maxwellfdfd.grid`: Grid classes (Grid2d, Grid3d)
- `numpy`: Numerical operations
- `scipy.sparse`: Sparse matrix operations
- `scipy.sparse.linalg`: Sparse linear algebra

## Usage Examples

### Creating Curl Operators

```python
import numpy as np
from maxwellfdfd.base import Axis, BC, GT
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.base import PhysUnit
from maxwellfdfd.diff import create_curls

unit = PhysUnit()
lprim = [np.array([0.0, 0.5, 1.0]), 
         np.array([0.0, 0.5, 1.0]), 
         np.array([0.0, 0.5, 1.0])]
Npml = [[1, 1], [1, 1], [1, 1]]
bc = {Axis.X: BC.E, Axis.Y: BC.E, Axis.Z: BC.E}
grid3d = Grid3d(unit, lprim, Npml, bc)

Ce, Cm = create_curls(GT.Prim, None, grid3d)
```

### Solving FDFD Equations

```python
from maxwellfdfd.base import FT, PML, EquationType
from maxwellfdfd.diff import solve_eq_direct

eqtype = EquationType(FT.E, GT.Prim)
eps_cell = {Axis.X: np.ones((2, 2, 2)), 
            Axis.Y: np.ones((2, 2, 2)), 
            Axis.Z: np.ones((2, 2, 2))}
mu_cell = {Axis.X: np.ones((2, 2, 2)), 
           Axis.Y: np.ones((2, 2, 2)), 
           Axis.Z: np.ones((2, 2, 2))}
s_factor_cell = {}
J_cell = {Axis.X: np.zeros((2, 2, 2)), 
          Axis.Y: np.zeros((2, 2, 2)), 
          Axis.Z: np.zeros((2, 2, 2))}
M_cell = {Axis.X: np.zeros((2, 2, 2)), 
          Axis.Y: np.zeros((2, 2, 2)), 
          Axis.Z: np.zeros((2, 2, 2))}
omega = 2 * np.pi * 3e8 / 1.55e-6

E, H = solve_eq_direct(eqtype, PML.SC, omega, eps_cell, mu_cell, 
                       s_factor_cell, J_cell, M_cell, grid3d)
```

### Using Matrix Equation Class

```python
from maxwellfdfd.diff import MatrixEquation

eq = MatrixEquation(eqtype, PML.SC, omega, eps_cell, mu_cell,
                   s_factor_cell, J_cell, M_cell, grid3d)

# Get matrix for direct solving
A, b = eq.matrix_op()

# Get matrix-free operators for iterative solving
Op, b, GfromF = eq.matrixfree_op()
```

## Testing Considerations

1. **Curl operators**: Verify that div(curl(F)) ≈ 0
2. **Divergence operators**: Verify appropriate identities
3. **Boundary conditions**: Test PEC, PMC, and periodic boundaries
4. **Matrix equation**: Test for both FT.E and FT.H field types
5. **Solvers**: Compare direct and iterative solver results
6. **PML**: Test with and without PML stretching

## Notes

- The Pylance errors shown in the project are from other modules (shape, source, grid) and are not related to the diff module conversion
- The diff module follows the architecture design documented in `ARCHITECTURE.md`
- All functions include comprehensive docstrings following NumPy format
- Type hints are provided for all function signatures
- The module is designed to work with the existing base, grid, and material modules

## Conversion Status

This module has been successfully converted from MATLAB to Python following the conversion rules in `MATLAB_to_Python_conversion_guide.md`. All 9 MATLAB files have been converted to 10 Python files (including the utility functions file).

### Files Converted

1. ✅ `create_curls.m` → `create_curls.py`
2. ✅ `create_divs.m` → `create_divs.py`
3. ✅ `create_Ds.m` → `create_Ds.py`
4. ✅ `create_Dw.m` → `create_Dw.py`
5. ✅ `create_masks.m` → `create_masks.py`
6. ✅ `EH_from_eh.m` → `eh_from_eh.py`
7. ✅ `MatrixEquation.m` → `matrix_equation.py`
8. ✅ `solve_eq_direct.m` → `solve_eq_direct.py`
9. ✅ `solve_eq_iterative.m` → `solve_eq_iterative.py`
10. ✅ `__init__.py` (new file)
11. ✅ `_utils.py` (new file for utility functions)
12. ✅ `README.md` (this file)

## Version

Current version: 0.1.0