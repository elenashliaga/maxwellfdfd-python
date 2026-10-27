# MaxwellFDFD-Python

A pure **NumPy/SciPy** finite-difference frequency-domain (FDFD) solver for Maxwell's equations — an independent Python port of the MATLAB [MaxwellFDFD](https://github.com/wsshin/maxwellfdfd) library by Wonseok Shin.

## What it does

Computes electromagnetic fields, waveguide eigenmodes, scattering, absorption, and facet/junction reflectivities of integrated-photonics structures: semiconductor lasers, SiN/Si photonics, hybrid-integration junctions, Bragg mirrors, optical Tamm states.

- Frequency-domain Maxwell on staggered Yee grids; sparse direct (LU) or iterative (BiCGSTAB/GMRES + ILU) solvers
- Stretched-coordinate PML with per-axis/end thickness control
- Sources: point, line, rect, plane (oblique, Bloch-periodic), TFSF, and modal (built on the eigenmode solver)
- 2D waveguide eigenmode solver (H-field eigenoperator; shift-invert and mode-order strategies)
- Dispersive material tables (Johnson & Christy, Palik) or constant ε/μ; direct complex-ε 3D array input for layered structures
- Arbitrary length unit `L0` — e.g. run everything in nanometers
- Time convention exp(+iωt) ⇒ passive media have Im(ε) < 0

## Installation

```bash
python3 -m pip install numpy scipy matplotlib h5py
```

No MATLAB required: the package is fully self-contained, and the material data tables are included in `dielconst/`.

## Usage

```python
from maxwellfdfd.base import Axis, BC
from maxwellfdfd.source import PointSrc
from maxwellfdfd.io import maxwell_run
import numpy as np

E, H, obj, src, extra = maxwell_run(
    'OSC', 1e-9, 20,                    # unit 1 nm, wavelength 20 nm
    'DOM', ['vacuum', 'none', 1.0],     # background
    np.array([[-60, 60]] * 2 + [[0, 1]]),  # domain box
    np.array([1, 1, 1]), BC.P,          # grid steps, periodic BC
    np.array([[5, 5], [5, 5], [0, 0]]), # PML cells per axis/end
    'SRCJ', PointSrc(Axis.Z, [0, 0, 0.5]),
)

ez = E[Axis.Z].array                    # complex field on the grid
```

Run from the repository root:

```bash
python3 -m pytest              # 51 tests, < 1 s
python3 examples/slit_2d.py    # Ag metal slit demo (≈ 10 s)
```

## Validation

- Analytic slab waveguide TE₀/TM₀ dispersion: |Δn_eff| < 5·10⁻⁴, |Im n_eff| ~ 10⁻⁸
- Operator-level discrete identities div(curl) = 0 on random fields
- Poynting-flux balance through closed boxes: 0.00 % for smooth fields
- TFSF pure-vacuum injection: exact plane wave inside the TF box, zero leakage outside

## References

- Original MATLAB solver: [wsshin/maxwellfdfd](https://github.com/wsshin/maxwellfdfd) — this package is an independent Python reimplementation; the module-by-module API mapping is in [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Mode-solver eigenformulation: G. Veronis and S. Fan, "Modes of subwavelength plasmonic slot waveguides," *J. Lightwave Technol.* **25**, 2511–2521 (2007), Sec. II

## License

MIT — see [LICENSE](LICENSE), © Elena Shliaga. The original MATLAB MaxwellFDFD is copyright its respective authors.
