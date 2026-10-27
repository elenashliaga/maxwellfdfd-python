# MaxwellFDFD IO Module

## Overview

The `io` module is the **top-level orchestration layer** of MaxwellFDFD. It
provides the main user-facing entry point `maxwell_run()`, data structures
for field representation, system building from flexible parameter groups,
and utility functions for array manipulation, grid generation, and file I/O.

This module depends on **all other modules**: `base`, `shape`, `source`,
`grid`, `material`, and `diff`.

## Files

### Main Interface

| File | Description |
|------|-------------|
| [`maxwell_run.py`](maxwell_run.py) | Main solver entry point — constructs domain and launches FDFD solver |
| [`build_system.py`](build_system.py) | Builds the FDFD system (grid, materials, sources, PML) from parameter groups |

### Data Structures

| File | Description |
|------|-------------|
| [`em_object.py`](em_object.py) | `EMObject` — combination of `Shape` and `Material` |
| [`scalar2d.py`](scalar2d.py) | `Scalar2d` — 2D scalar field with grid metadata |
| [`scalar3d.py`](scalar3d.py) | `Scalar3d` — 3D scalar field with grid metadata |
| [`progmark.py`](progmark.py) | `ProgMark` — progress/timing marker |

### Utility Functions

| File | Functions | Description |
|------|-----------|-------------|
| [`vec_utils.py`](vec_utils.py) | `flip_array`, `flip_vec`, `div_vec`, `mult_vec`, `neg_vec` | Cell-vector element-wise operations |
| [`utils.py`](utils.py) | `array2scalar`, `cell2array`, `collapse_complex`, `expand_complex`, `expand_node_array`, `real_cell`, `reordering_indices`, `slice_scalar3d`, `transform_gt`, `vec2amp` | Array/data manipulation |
| [`grid_utils.py`](grid_utils.py) | `generate_s_factor`, `read_grid` | PML s-factor generation and grid file reading |
| [`io_utils.py`](io_utils.py) | `read_output`, `write_input`, `fds` | HDF5 file I/O and server-based solver |

## Key Conversion Decisions

### 1. Flexible Parameter Parsing (`varargin` → `*args`)

MATLAB's `maxwell_run(varargin)` accepts positional arguments with string
group markers (`'OSC'`, `'DOM'`, `'OBJ'`, `'SOBJ'`, `'SRCJ'`, `'SRCM'`).

**Python approach:**
```python
def maxwell_run(*args, inspect_only=False, solveropts=None):
```

This preserves the MATLAB-style positional calling convention while making
`inspect_only` and `solveropts` clean keyword-only arguments.

### 2. Cell Arrays → Dictionaries with Enum Keys

| MATLAB | Python |
|--------|--------|
| `E_cell{Axis.x}` | `E_cell[Axis.X]` |
| `s_factor_cell{w, g}` | `s_factor_cell[(w, g)]` |

### 3. `subsindex()` → `int()`

MATLAB `subsindex(GT.prim)` returns 0; Python `int(GT.PRIM)` returns 0.

### 4. Immutable Properties

MATLAB `properties (SetAccess = immutable)` → Python `@property` (read-only,
no setter). Values set once in `__init__` as private `_attr`.

### 5. HDF5 I/O

MATLAB `h5read`/`h5write`/`h5create` → Python `h5py`. Complex arrays use
`expand_complex`/`collapse_complex` for interleaved real/imaginary storage.

### 6. Interpolation

MATLAB `interpn` → Python `scipy.interpolate.interpn` for field value
extraction and grid-type transformation.

### 7. `fds.m` — Server-Based Solver

The `fds` function communicates with an external **lightlabsFDS** server
via FTP. The Python conversion uses `ftplib` and `h5py`. If the server is
unavailable, a `NotImplementedError` is raised with guidance to use
`solve_eq_direct` or `solve_eq_iterative` instead.

### 8. EMObject and the Material Module Stub

The `material` module has a private `_emobject.py` stub used for type
checking. The **authoritative** `EMObject` implementation lives in
`io/em_object.py`. The material stub should eventually be updated to
re-export from the io module.

### 9. Visualization

Visualization calls in `maxwell_run` (MATLAB `figure`/`vis2d`/`drawnow`)
are guarded behind a try/except import of the `vis` module, which has not
been converted yet. A warning is printed if visualization is requested but
unavailable.

## MATLAB → Python File Mapping

| MATLAB (26 files) | Python (11 files) |
|-------------------|-------------------|
| `maxwell_run.m` | `maxwell_run.py` |
| `build_system.m` | `build_system.py` |
| `EMObject.m` | `em_object.py` |
| `Scalar2d.m` | `scalar2d.py` |
| `Scalar3d.m` | `scalar3d.py` |
| `ProgMark.m` | `progmark.py` |
| `flip_array.m`, `flip_vec.m`, `div_vec.m`, `mult_vec.m`, `neg_vec.m` | `vec_utils.py` |
| `array2scalar.m`, `cell2array.m`, `collapse_complex.m`, `expand_complex.m`, `expand_node_array.m`, `real_cell.m`, `reordering_indices.m`, `slice_scalar3d.m`, `transform_gt.m`, `vec2amp.m` | `utils.py` |
| `generate_s_factor.m`, `read_grid.m` | `grid_utils.py` |
| `read_output.m`, `write_input.m`, `fds.m` | `io_utils.py` |

## Usage

```python
from maxwellfdfd.base import Axis, BC
from maxwellfdfd.shape import Box
from maxwellfdfd.source import PointSrc
from maxwellfdfd.io import maxwell_run

# Run a simulation
E, H, objs, srcs, extra = maxwell_run(
    'OSC', 1e-9, 1550,
    'DOM', ['Palik/SiO2', 'none'],
        [[-700, 700], [-600, 600], [-200, 1700]], 20, BC.P, 200,
    'OBJ',
        ['Palik/SiO2', 'none'],
        Box([[-50, 50], [-50, 50], [-200, 1700]], [2, 2, 20]),
    'SRCJ', PointSrc(Axis.X, [0, 0, 200]),
)

# Access results
E_x = E[Axis.X]          # Scalar3d for E_x
grid = extra['grid3d']   # Grid3d
eps = extra['eps']       # {Axis: Scalar3d} permittivity
```

## Dependencies

- **Internal:** `base`, `shape`, `source`, `grid`, `material`, `diff`
- **External:** `numpy`, `scipy` (sparse, interpolate), `h5py`, `ftplib` (for `fds`)

## Notes

- All indexing is 0-based (Python convention), unlike MATLAB's 1-based indexing.
- Cell arrays are represented as `dict` with enum keys.
- The `fds` server-based solver requires network access to a configured
  lightlabsFDS instance.
- The PETSc binary format in the original `write_input.m` is documented as
  future work; the HDF5 path is the primary implementation.
