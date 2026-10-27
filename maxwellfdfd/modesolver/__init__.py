"""
Waveguide mode solver for MaxwellFDFD (Python port).

Ported from the MATLAB package `maxwellfdfd/modesolver/`:
- `wgmode_matrix.m`  -> `wgmode_matrix.py` (this package)
- `solve_mode.m`, `solve_mode_real.m` -> `solve_mode.py`
- `prep_modalsrc.m`  -> `ModalSrc.prep()` (see `maxwellfdfd/source/modal_src.py`)

The operator ``A`` built by :func:`wgmode_matrix` acts on the transverse
H-field vector ``[Hh; Hv]`` of the 2D cross-section orthogonal to the
waveguide axis and satisfies the eigenvalue equation ``A x = lambda x`` with
``lambda = gamma^2 = -beta^2`` (``gamma = 1j*beta``), so that
``beta = sqrt(-lambda)``.

Reference: Sec. II of G. Veronis and S. Fan, Journal of Lightwave Technology,
vol. 25, no. 9, pp. 2511-2521 (2007).
"""

from .wgmode_matrix import wgmode_matrix
from .solve_mode import solve_mode, solve_mode_order

__all__ = ['wgmode_matrix', 'solve_mode', 'solve_mode_order']
