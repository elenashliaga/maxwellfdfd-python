# MaxwellFDFD Source Module

This module provides electromagnetic sources for exciting fields in MaxwellFDFD simulations. It has been converted from the MATLAB source module to Python following the architecture design documented in `ARCHITECTURE.md`.

## Module Contents

### Base Class

- **`Source`** ([`source.py`](source.py)): Abstract base class for all electromagnetic sources. Defines the interface that all concrete source implementations must follow.

### Concrete Source Classes

- **`PointSrc`** ([`point_src.py`](point_src.py)): Point source representing an electric point dipole. The most basic source type that places an oscillating electric point dipole at a specified location.

- **`LineSrc`** ([`line_src.py`](line_src.py)): Line source representing a constant electric dipole distribution along a line. Can generate cylindrical waves with support for oblique incidence via Bloch boundary conditions.

- **`RectSrc`** ([`rect_src.py`](rect_src.py)): Rectangular source representing constant electric dipole distribution over a rectangular patch. Similar to PlaneSrc but limited to a rectangular region.

- **`PlaneSrc`** ([`plane_src.py`](plane_src.py)): Plane wave source representing constant electric dipole distribution over an entire plane. Generates plane waves with support for oblique incidence via Bloch boundary conditions.

- **`ModalSrc`** ([`modal_src.py`](modal_src.py)): Modal source for generating waveguide modes. Stores the transverse J-field distribution on a plane to generate a mode of a waveguide. Requires an external mode solver.

- **`TFSFPlaneSrc`** ([`tfsf_plane_src.py`](tfsf_plane_src.py)): Total-field/scattered-field plane wave source. Generates a plane wave inside a total field box for the TF/SF method. Supports arbitrary propagation and polarization directions.

### Mixin

- **`WithBloch`** ([`with_bloch.py`](with_bloch.py)): Mixin class for sources that support Bloch boundary conditions. Provides an abstract interface for specifying Bloch wave vectors.

### Utility Functions

- **`assign_source`** ([`utils.py`](utils.py)): Assigns multiple sources to the grid by superposing their contributions.

- **`ind_for_loc`** ([`utils.py`](utils.py)): Finds the index of a grid location closest to a given location. Issues a warning if an exact match is not found.

## Conversion Notes

### Key Conversion Decisions

1. **Indexing**: All MATLAB 1-based indexing has been converted to Python 0-based indexing.

2. **Naming Conventions**: 
   - Classes use PascalCase (e.g., `PointSrc`, `LineSrc`)
   - Functions and methods use snake_case (e.g., `assign_source`, `ind_for_loc`)
   - Properties use snake_case (e.g., `polarization`, `intercept`)

3. **Abstract Methods**: The `Source` base class uses `@abstractmethod` decorator for the `generate_kernel` method, following Python ABC patterns.

4. **Type Hints**: All function signatures include type hints following Python 3.9+ syntax.

5. **Docstrings**: All classes and functions have comprehensive docstrings following NumPy docstring format.

6. **Cell Arrays**: MATLAB cell arrays have been converted to Python lists or dictionaries with enum keys.

7. **Mixin Pattern**: The `WithBloch` mixin is implemented as a Python class with multiple inheritance support.

### Source Generation Methods

The source generation methods (`generate_kernel`) have been converted with the following considerations:

1. **Index Handling**: MATLAB's cell array indexing has been converted to Python list indexing with support for slice objects and numpy arrays.

2. **Phase Calculations**: Bloch phase calculations use numpy array operations for efficiency.

3. **Grid Access**: Grid properties are accessed via dictionary-like syntax (e.g., `grid3d.dl[int(axis), gt]`).

4. **Return Types**: Methods return tuples of (index_cell, JMw_patch) where index_cell is a list of indices and JMw_patch is a numpy array.

### Dependencies

The source module depends on:

- `maxwellfdfd.base`: Base enumerations (Axis, GT, Dir, Sign) and utilities (chkarg)
- `maxwellfdfd.shape`: Shape classes (Point, Line, Plane, Rectangle, Box)
- `numpy`: Numerical operations
- `typing`: Type hints

### Known Limitations

1. **Grid Module**: Some classes reference Grid3d and Grid2d that are not yet implemented. These are handled with TYPE_CHECKING to avoid circular imports.

2. **Oscillation Module**: ModalSrc and TFSFPlaneSrc reference Oscillation class that is not yet implemented.

3. **Material Module**: TFSFPlaneSrc references Material class for background material.

4. **Validation**: Some validation checks are commented out pending implementation of dependent modules.

## Usage Examples

### Point Source

```python
import numpy as np
from maxwellfdfd.base import Axis, GT
from maxwellfdfd.source import PointSrc

# Create a z-polarized point source at the origin
src = PointSrc(Axis.Z, np.array([0, 0, 0]))

# Set grid type
src.set_gridtype(GT.PRIM)

# Generate source for z-axis
index_cell, JMz_patch = src.generate(Axis.Z, grid3d)
```

### Plane Wave Source

```python
import numpy as np
from maxwellfdfd.base import Axis
from maxwellfdfd.source import PlaneSrc

# Create a z-polarized plane wave on y = 0 plane
src = PlaneSrc(Axis.Y, 0, Axis.Z)

# Create with oblique incidence
src = PlaneSrc(Axis.Y, 0, Axis.Z, theta=np.pi/6, wvlen=1.0)
```

### Line Source with Bloch Boundary Condition

```python
import numpy as np
from maxwellfdfd.base import Axis
from maxwellfdfd.source import LineSrc

# Create a z-axis line source at origin
src = LineSrc(Axis.Z, np.array([0, 0]))

# Create with oblique incidence
src = LineSrc(Axis.Z, np.array([0, 0]), theta=np.pi/6, wvlen=1.0)
```

### Assigning Multiple Sources

```python
from maxwellfdfd.source import PointSrc, assign_source

# Create multiple sources
src1 = PointSrc(Axis.Z, np.array([0, 0, 0]))
src2 = PointSrc(Axis.Z, np.array([1, 1, 1]))

# Assign sources to grid
JM_cell = assign_source(grid3d, [src1, src2])
```

## Testing

The source module should be tested with the following considerations:

1. **Unit Tests**: Test each source class independently with mock grid objects.

2. **Integration Tests**: Test source assignment with actual grid objects when available.

3. **Indexing Tests**: Verify correct 0-based indexing conversion.

4. **Phase Tests**: Verify Bloch phase calculations for LineSrc and PlaneSrc.

5. **Edge Cases**: Test boundary conditions and edge cases for each source type.

## Future Work

1. **Grid Integration**: Complete integration with Grid3d and Grid2d classes when implemented.

2. **Mode Solver Integration**: Complete ModalSrc integration with mode solver.

3. **Material Integration**: Complete TFSFPlaneSrc integration with Material class.

4. **Performance Optimization**: Optimize phase calculations for large grids.

5. **Additional Validation**: Add more comprehensive input validation.

## References

- MATLAB source module: `/Users/elenashliaga/science/maxwellfdfd/source/`
- Architecture design: `/Users/elenashliaga/science/maxwellfdfd-python/ARCHITECTURE.md`
- Conversion guide: `MATLAB_to_Python_conversion_guide.md`