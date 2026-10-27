"""
Metal slit (Ag) — 2D FDFD simulation.

Converted from MATLAB: maxwellfdfd/example/2d/slit_2d.m

Two silver blocks separated by a 160 nm vacuum gap form a subwavelength
metallic slit. A z-polarized *magnetic* plane source (SRCM) is injected at
y = -500 nm, exciting the TM slit mode. The power transmitted through the
slit is measured at y = 500 nm.

This is the magnetic-source (TM) counterpart of ``ag_slot_2d.py`` (which uses
an electric SRCJ). The MATLAB original also includes commented-out variants
(PEC slit, no metal, modal source, point source, rectangular source); only
the active silver-slit case is translated here.

Material note: MATLAB uses ``{'CRC/Ag', gray}``. We use a constant
permittivity eps ~ -129.2 - 3.3j (Ag at 1550 nm). Replace with
``['CRC/Ag', gray]`` if the data table is available.

Run from the maxwellfdfd-python directory:
    python3 examples/slit_2d.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from maxwellfdfd.base import Axis, BC
from maxwellfdfd.shape import Box
from maxwellfdfd.source import PlaneSrc
from maxwellfdfd.io import maxwell_run

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_utils import vis2d_full, powerflux_patch


def main():
    inspect_only = False

    gray = [0.5, 0.5, 0.5]
    flux_y = 500
    flux_x1, flux_x2 = -200, 200

    # Ag at 1550 nm (fallback for 'CRC/Ag')
    ag_material = ['Ag', gray, -129.2 - 3.3j]

    # ---- Solve (magnetic plane source, z-polarized) ----
    E, H, obj_array, src_array, extra = maxwell_run(
        'OSC', 1e-9, 1550,
        'DOM', ['vacuum', 'none', 1.0],
            np.array([[-1100, 1100], [-1100, 2600], [0, 10]]),
            10, BC.P, [100, 100, 0],
        'OBJ',
            ag_material,
            Box(np.array([[-1100, -80], [0, 1000], [0, 10]])),
            Box(np.array([[80, 1100], [0, 1000], [0, 10]])),
        'SRCM', PlaneSrc(Axis.Y, -500, Axis.Z),
        inspect_only=inspect_only,
    )

    if not inspect_only:
        # ---- Visualize |Ex| (TM slit mode) with structure overlay ----
        # MATLAB: vis2d(E{Axis.x}, Axis.z, z_loc, obj_array, src_array, opts)
        #         with opts.withabs=true → abs(Ex) with 'hot' colormap
        z_loc = 5
        fig = vis2d_full(
            E[Axis.X],
            obj_array=obj_array,
            src_array=src_array,
            extra=extra,
            z_index=0,
            withabs=True,
            withpml=False,
            title='|E_x| — Ag metal slit (TM, 1550 nm)',
            savepath=os.path.join(os.path.dirname(__file__), 'slit_2d_Ex.png'),
        )
        plt.close(fig)

        # ---- Power flux through the slit ----
        # NOTE: the MATLAB original limits the integration to the slit
        # aperture x in [flux_x1, flux_x2]. The Python powerflux_patch does
        # not yet support a sub-range, so the full y-plane flux is returned.
        grid3d = extra['grid3d']
        power = powerflux_patch(E, H, Axis.Y, flux_y, grid3d)

        print('\nMetal slit simulation complete.')
        print(f'  transmitted power = {power:.6e}')


if __name__ == '__main__':
    main()
