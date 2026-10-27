# MaxwellFDFD Base Module

## Overview

The base module provides fundamental enumerations, classes, and utility functions used throughout the MaxwellFDFD package. This module was converted from the MATLAB base module located in `/Users/elenashliaga/science/maxwellfdfd/base/`.

## Files

### Core Files

- `__init__.py` - Package initialization and public API exports
- `README.md` - This file

### Enumerations

- `_enumerated.py` - Base enumeration class (internal)
- `axis.py` - Cartesian axes (X, Y, Z)
- `bc.py` - Boundary conditions (E=PEC, M=PMC, P=periodic)
- `dir.py` - Directions (H=horizontal, V=vertical)
- `gt.py` - Grid types (PRIM=primary, DUAL=dual)
- `physq.py` - Physical quantities (14 quantities with symbols and SI units)
- `pml.py` - PML kinds (SC=stretched-coordinate, U=uniaxial)
- `sign.py` - Signs (N=negative, P=positive)
- `ft.py` - Field types (E=electric, H=magnetic)

### Classes

- `oscillation.py` - Oscillation parameters (wavelength, angular frequency, energy)
- `equation_type.py` - Equation type definition (field type and grid type)
- `physc.py` - Physical constants (c0, mu0, eps0, eta0, h, hbar)
- `physunit.py` - Physical unit handling (normalized units based on L0)

### Utilities

- `utils.py` - Utility functions for validation and array manipulation

## Public API

### Enumerations

#### Axis
```python
from maxwellfdfd.base import Axis

Axis.count()  # 3
Axis.elems()  # [Axis.X, Axis.Y, Axis.Z]
int(Axis.X)   # 0
str(Axis.X)   # 'x'
p, q, r = Axis.Y.cycle()  # Cyclic permutation
```

#### BC (Boundary Conditions)
```python
from maxwellfdfd.base import BC

BC.count()  # 3
BC.elems()  # [BC.E, BC.M, BC.P]
# E: PEC (tangential E-field = 0)
# M: PMC (normal E-field = 0)
# P: periodic
```

#### Dir (Directions)
```python
from maxwellfdfd.base import Dir

Dir.count()  # 2
Dir.elems()  # [Dir.H, Dir.V]
Dir.H.alter()  # Dir.V
```

#### GT (Grid Types)
```python
from maxwellfdfd.base import GT

GT.count()  # 2
GT.elems()  # [GT.PRIM, GT.DUAL]
GT.PRIM.alter()  # GT.DUAL
```

#### PhysQ (Physical Quantities)
```python
from maxwellfdfd.base import PhysQ

PhysQ.count()  # 14
PhysQ.L.symbol  # 'L'
PhysQ.L.SIunit  # 'm'
```

#### PML (PML Kinds)
```python
from maxwellfdfd.base import PML

PML.count()  # 2
PML.elems()  # [PML.SC, PML.U]
# SC: stretched-coordinate
# U: uniaxial
```

#### Sign
```python
from maxwellfdfd.base import Sign

Sign.count()  # 2
Sign.elems()  # [Sign.N, Sign.P]
Sign.N.alter()  # Sign.P
```

#### FT (Field Types)
```python
from maxwellfdfd.base import FT

FT.count()  # 2
FT.elems()  # [FT.E, FT.H]
FT.E.alter()  # FT.H
```

### Classes

#### Oscillation
```python
from maxwellfdfd.base import Oscillation, PhysUnit

unit = PhysUnit(1e-6)  # 1 micrometer reference
osc = Oscillation(1.0, unit)  # Wavelength of 1 L0

osc.in_L0()      # Wavelength in L0 units
osc.in_omega0()  # Angular frequency (2*pi/wvlen)
osc.in_eV()      # Photon energy in eV
```

#### EquationType
```python
from maxwellfdfd.base import EquationType, FT, GT

eqtype = EquationType(FT.E, GT.PRIM)
eqtype.f   # FT.E
eqtype.ge  # GT.PRIM
```

#### PhysC (Physical Constants)
```python
from maxwellfdfd.base import PhysC

PhysC.c0    # 2.99792458e8 (m/s)
PhysC.mu0   # 4*pi*1e-7 (H/m)
PhysC.eps0  # 1/(c0^2*mu0) (F/m)
PhysC.eta0  # sqrt(mu0/eps0) (Ohm)
PhysC.h     # 4.135667662e-15 (eV·sec)
PhysC.hbar  # h/(2*pi) (eV·sec)
```

#### PhysUnit
```python
from maxwellfdfd.base import PhysUnit, PhysQ

unit = PhysUnit(1e-6)  # 1 micrometer reference
unit.value(PhysQ.L)  # 1e-6 (meters)
unit.value([[PhysQ.L, 1], [PhysQ.OMEGA, -1]])  # L/omega
```

### Utility Functions

#### chkarg
```python
from maxwellfdfd.base import chkarg

chkarg(condition, "Error message %s", "arg")
# Raises ValueError if condition is False
```

#### create_spdiag
```python
from maxwellfdfd.base import create_spdiag
import numpy as np

D = create_spdiag(np.array([1, 2, 3]))
# Returns sparse diagonal matrix
```

#### expand2mat
```python
from maxwellfdfd.base import expand2mat

expand2mat(5, 2, 3)  # 2x3 matrix of 5s
expand2mat([1, 2], 2, 3)  # 2x3 matrix with columns [1, 2]
```

#### expand2row
```python
from maxwellfdfd.base import expand2row

expand2row(5, 3)  # [5, 5, 5]
expand2row([1, 2, 3], 3)  # [1, 2, 3]
```

#### is2dlike
```python
from maxwellfdfd.base import is2dlike
import numpy as np

truth, normal_axis = is2dlike(np.array([100, 100, 5]))
# Returns (True, Axis.Z) if one dimension is < 10% of others
```

#### isexpandable2mat
```python
from maxwellfdfd.base import isexpandable2mat

isexpandable2mat(5, 2, 3)  # True
isexpandable2mat([1, 2], 2, 3)  # True
isexpandable2mat([1, 2], 3, 3)  # False
```

#### isexpandable2row
```python
from maxwellfdfd.base import isexpandable2row

isexpandable2row(5, 3)  # True
isexpandable2row([1, 2, 3], 3)  # True
isexpandable2row([1, 2], 3)  # False
```

#### isphysQcell
```python
from maxwellfdfd.base import isphysQcell, PhysQ

isphysQcell([[PhysQ.L, 1], [PhysQ.OMEGA, -1]])  # True
isphysQcell([[1, 2], [3, 4]])  # False
```

#### istypeof
```python
from maxwellfdfd.base import istypeof

istypeof(5, 'int')  # True
istypeof(5.0, 'real')  # True
istypeof(3+4j, 'complex')  # True
```

#### istypesizeof
```python
from maxwellfdfd.base import istypesizeof
import numpy as np

istypesizeof(5, 'int')  # True (scalar)
istypesizeof(np.array([[1, 2], [3, 4]]), 'int', (2, 2))  # True
istypesizeof(np.array([1, 2, 3]), 'int', (1, 0))  # True (row vector)
```

## Conversion Notes

### Key Changes from MATLAB

1. **Indexing**: All 1-based MATLAB indexing converted to 0-based Python indexing
2. **Naming**: Functions use snake_case, classes use PascalCase
3. **Enumerations**: Use Python's `enum.Enum` instead of MATLAB classdef enumeration
4. **Properties**: Use `@property` decorator for read-only properties
5. **Error Handling**: Use Python exceptions instead of MATLAB MException
6. **Sparse Matrices**: Use `scipy.sparse` for sparse diagonal matrices
7. **Type Hints**: Added to all function signatures
8. **Docstrings**: NumPy-style docstrings for all public APIs

### Architecture Decisions

1. **Base Enumeration Class**: Created `_enumerated.py` as base class for all enumerations
2. **Separate Files**: Each enumeration in its own file for better organization
3. **Internal Functions**: Prefixed with underscore (`_`) to indicate internal use
4. **Type Safety**: Used type hints throughout for better IDE support and error checking

### Dependencies

- `enum` - For enumerations
- `numpy` - For numerical operations
- `scipy.sparse` - For sparse matrices
- `typing` - For type hints

## Testing

To test the base module:

```bash
cd maxwellfdfd-python
python3 -m pytest tests/test_base/
```

## Examples

See the `examples/` directory for usage examples.

## References

- Original MATLAB code: `/Users/elenashliaga/science/maxwellfdfd/base/`
- Architecture design: `maxwellfdfd-python/ARCHITECTURE.md`
- Conversion guide: `MATLAB_to_Python_conversion_guide.md`