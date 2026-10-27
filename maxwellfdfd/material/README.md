# Material Module

This module provides classes and functions for defining and assigning electromagnetic materials to grid nodes in MaxwellFDFD simulations.

## Overview

The material module is responsible for:
- Defining electromagnetic materials with electric permittivity and magnetic permeability
- Loading material data from files
- Assigning material properties to grid nodes
- Computing mean material properties at grid nodes

## Module Structure

```
maxwellfdfd/material/
├── __init__.py              # Public API exports
├── material.py              # Material class
├── assign_material_node.py  # Material assignment function
├── mean_material_node.py    # Mean material computation
├── _emobject.py            # EMObject stub (private)
└── README.md               # This file
```

## Public API

### Classes

#### `Material`

Represents an electromagnetic material with electric permittivity and magnetic permeability.

**Constructor:**
```python
Material(name, color, eps, mu=None, islossless=False)
```

**Parameters:**
- `name` (str): Material name
- `color` (str or tuple): Color for visualization (e.g., 'r', 'b', or (0.5, 0.5, 0.5))
- `eps` (complex or array_like): Electric permittivity (scalar, vector, or tensor)
- `mu` (complex or array_like, optional): Magnetic permeability. Default is 1.0
- `islossless` (bool, optional): If True, ignore loss/gain. Default is False

**Properties:**
- `eps` (numpy.ndarray): Electric permittivity tensor (3x3, immutable)
- `mu` (numpy.ndarray): Magnetic permeability tensor (3x3, immutable)
- `name` (str): Material name (mutable)
- `color` (str or tuple): Material color (mutable)
- `has_iso_eps` (bool): True if permittivity is isotropic
- `has_iso_mu` (bool): True if permeability is isotropic
- `is_iso` (bool): True if fully isotropic

**Static Methods:**
- `fromtable(osc, name, color, islossless=False)`: Create material from data file

**Examples:**
```python
from maxwellfdfd.material import Material

# Create vacuum
vacuum = Material('vacuum', 'none', 1.0)

# Create dielectric
dielectric = Material('dielectric', 'b', 4.0)

# Create lossy material
lossy = Material('lossy', 'r', 2.0 - 0.1j)

# Create lossless material from lossy input
lossless = Material('lossless', 'g', 2.0 - 0.1j, islossless=True)

# Create anisotropic material
anisotropic = Material('anisotropic', 'm', [4.0, 4.0, 4.5])

# Load material from data file
from maxwellfdfd.base import Oscillation
osc = Oscillation(wvlen=1.55)  # 1.55 um wavelength
gold = Material.fromtable(osc, 'Au', 'y')
```

### Functions

#### `assign_material_node`

Assign material properties to grid nodes based on object shapes and materials.

**Signature:**
```python
assign_material_node(grid3d, object_array, eps_node_cell=None, mu_node_cell=None)
```

**Parameters:**
- `grid3d` (Grid3d): 3D grid object
- `object_array` (list of EMObject): List of electromagnetic objects
- `eps_node_cell` (dict, optional): Dictionary mapping axes to permittivity arrays
- `mu_node_cell` (dict, optional): Dictionary mapping axes to permeability arrays

**Returns:**
- `eps_node_cell` (dict): Dictionary mapping axes to permittivity arrays
- `mu_node_cell` (dict): Dictionary mapping axes to permeability arrays

**Examples:**
```python
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.shape import Box
from maxwellfdfd.material import Material, assign_material_node
from maxwellfdfd.material._emobject import EMObject

# Create grid
grid = Grid3d(...)

# Create material and shape
mat = Material('dielectric', 'b', 4.0)
shape = Box(...)

# Create object and assign materials
obj = EMObject(shape, mat)
eps, mu = assign_material_node(grid, [obj])
```

#### `mean_material_node`

Compute mean material at grid nodes using arithmetic or harmonic averaging.

**Signature:**
```python
mean_material_node(grid3d, gt, material_node)
```

**Parameters:**
- `grid3d` (Grid3d): 3D grid object
- `gt` (GT): Grid type (GT.PRIM for primary, GT.DUAL for dual)
- `material_node` (dict): Dictionary mapping axes to material arrays

**Returns:**
- `material_cell` (dict): Dictionary mapping axes to mean material arrays

**Notes:**
- Uses arithmetic mean for primary grid (GT.PRIM)
- Uses harmonic mean for dual grid (GT.DUAL)

**Examples:**
```python
from maxwellfdfd.grid import Grid3d
from maxwellfdfd.base import GT
from maxwellfdfd.material import mean_material_node

# Compute mean material for primary grid
material_cell = mean_material_node(grid, GT.PRIM, material_node)

# Compute mean material for dual grid
material_cell = mean_material_node(grid, GT.DUAL, material_node)
```

## Material Data Files

The `Material.fromtable()` method loads material data from `.mat` files located in the `dielconst` directory. Each data file should contain:

- `eV`: Array of photon energies in eV
- `n`: Array of refractive indices
- `k`: Array of extinction coefficients

The permittivity is computed as: `epsilon = (n - 1j*k)^2`

**File naming convention:**
- Use '/' as path separator in the `name` parameter
- The function converts '/' to the OS-specific file separator
- Example: `Material.fromtable(osc, 'metals/Au', 'y')` loads from `dielconst/metals/Au.mat`

## Conversion Notes

### Key Conversion Decisions

1. **Tensor Representation**: Material tensors are stored as 3x3 numpy arrays, matching the MATLAB implementation.

2. **Immutable Properties**: The `eps` and `mu` properties are immutable (stored as private attributes `_eps` and `_mu`), matching MATLAB's `SetAccess = immutable`.

3. **Dependent Properties**: Properties like `has_iso_eps`, `has_iso_mu`, and `is_iso` are computed on-demand using `@property` decorators.

4. **Material Data Loading**: Uses `scipy.io.loadmat` to load `.mat` files and `scipy.interpolate.interp1d` for interpolation.

5. **Grid Node Assignment**: The `assign_material_node` function returns dictionaries with `Axis` enum keys instead of MATLAB cell arrays.

6. **Mean Computation**: Arithmetic mean is used for primary grid, harmonic mean for dual grid, preserving the physics of field continuity.

### Deviations from Architecture

1. **EMObject Stub**: A stub EMObject class (`_emobject.py`) was created as a placeholder for the io module. This is a private module and should be replaced when the io module is converted.

2. **Helper Functions**: The `mean_material_node` module includes private helper functions (`_expand_node_array`, `_arithmetic_mean_material_node`, `_harmonic_mean_material_node`) that were nested functions in the MATLAB implementation.

3. **Axis Iteration**: Due to Python's enum implementation, `Axis.elems()` returns a list rather than being directly iterable. This requires storing the result in a variable before iteration.

## Dependencies

The material module depends on:

- `maxwellfdfd.base`: Axis, GT, Sign, utils (chkarg, expand2row, isexpandable2row), Oscillation
- `maxwellfdfd.shape`: Shape, Box
- `maxwellfdfd.grid`: Grid3d
- `numpy`: Array operations
- `scipy.io`: Load .mat files
- `scipy.interpolate`: Interpolation

## Testing

To test the material module:

```python
import numpy as np
from maxwellfdfd.material import Material

# Test basic material creation
mat = Material('test', 'r', 2.0)
assert mat.is_iso
assert np.allclose(mat.eps, np.eye(3) * 2.0)

# Test anisotropic material
mat = Material('test', 'r', [2.0, 2.0, 3.0])
assert not mat.is_iso
assert mat.has_iso_eps
assert mat.has_iso_mu

# Test lossless conversion
mat = Material('test', 'r', 2.0 - 0.1j, islossless=True)
assert np.allclose(mat.eps, np.eye(3) * 2.0)
assert mat.name == 'test (lossless)'
```

## Future Work

1. Replace the EMObject stub with the full implementation from the io module
2. Add support for frequency-dependent materials (dispersion)
3. Add support for nonlinear materials
4. Add material library with common materials
5. Add material validation and consistency checks

## References

- MATLAB original: `/Users/elenashliaga/science/maxwellfdfd/material/`
- Architecture: `/Users/elenashliaga/science/maxwellfdfd-python/ARCHITECTURE.md`
- Conversion guide: `/Users/elenashliaga/science/MATLAB_to_Python_conversion_guide.md`