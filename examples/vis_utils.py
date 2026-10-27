"""
Visualization utilities for MaxwellFDFD Python examples.

This module provides matplotlib-based replacements for the MATLAB ``vis2d``
visualization function, along with basic power-flux calculation helpers
(``powerflux_box`` and ``powerflux_patch``) that are not yet available in
the core ``maxwellfdfd`` package.

Functions
---------
vis2d_field : Plot a 2D slice of a 3D Scalar3d field component.
vis2d_real  : Plot the real part of a 2D field slice.
powerflux_box   : Integrate the Poynting flux through a rectangular box.
powerflux_patch : Integrate the Poynting flux through a planar patch.
"""

from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from maxwellfdfd.base import Axis, GT


# -----------------------------------------------------------------------
# Colormaps and field preparation (matching MATLAB's Painter2d)
# -----------------------------------------------------------------------

def _b2r_colormap():
    """Blue-to-white-to-red diverging colormap matching MATLAB's ``b2r.m``.

    Goes from blue [0,0,1] through white [1,1,1] to red [1,0,0].
    """
    return LinearSegmentedColormap.from_list(
        'b2r',
        [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
        N=256,
    )


def _prep_field_2d(
    scalar3d,
    z_index: int = 0,
    *,
    withinterp: bool = True,
    withpml: bool = False,
):
    """
    Prepare a 2D field slice for plotting, matching MATLAB's ``Painter2d.prep_data``.

    Uses ``data_expanded()`` (includes ghost/boundary points) and interpolates
    to the plotting grid returned by ``grid3d.lplot()``.

    Returns
    -------
    C : ndarray
        2D interpolated field array, shape ``(len(x_plot), len(y_plot))``.
    x_plot : ndarray
        1D x plotting coordinates.
    y_plot : ndarray
        1D y plotting coordinates.
    """
    # Expanded data (includes ghost points) — matches MATLAB data_expanded()
    array_3d, l_cell = scalar3d.data_expanded()

    grid3d = scalar3d.grid3d
    gt_array = scalar3d.gt_array

    # Expanded grid locations — truncate to match array dimensions
    # (lall may include an extra periodic-boundary point not in the array)
    x_exp = l_cell[Axis.X][:array_3d.shape[0]]
    y_exp = l_cell[Axis.Y][:array_3d.shape[1]]

    # Plotting locations from grid3d.lplot()
    l_plot = grid3d.lplot(gt_array, withinterp, withpml)
    x_plot = np.asarray(l_plot[0])
    y_plot = np.asarray(l_plot[1])

    # Extract 2D slice at z_index
    field_2d = array_3d[:, :, z_index]

    # Interpolate from expanded grid to plotting grid (matches MATLAB interpn)
    need_interp = (
        len(x_plot) != len(x_exp)
        or len(y_plot) != len(y_exp)
        or not np.allclose(x_plot, x_exp)
        or not np.allclose(y_plot, y_exp)
    )

    if withinterp and need_interp:
        try:
            from scipy.interpolate import RegularGridInterpolator
            rgi = RegularGridInterpolator(
                (x_exp, y_exp), field_2d,
                bounds_error=False, fill_value=None,
            )
            Xp, Yp = np.meshgrid(x_plot, y_plot, indexing='ij')
            pts = np.stack([Xp, Yp], axis=-1)
            C = rgi(pts)
        except ImportError:
            C = field_2d
            x_plot = x_exp
            y_plot = y_exp
    else:
        C = field_2d

    return C, x_plot, y_plot


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

def _extract_2d_slice(
    scalar3d,
    z_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a 2D (x-y) slice from a 3D Scalar3d at a given z-index.

    Parameters
    ----------
    scalar3d : Scalar3d
        The 3D scalar field.
    z_index : int, optional
        Index along z to slice (default 0).

    Returns
    -------
    field_2d : ndarray
        2D complex array (Nx x Ny).
    x : ndarray
        1D x-coordinates.
    y : ndarray
        1D y-coordinates.
    """
    data, l_cell = scalar3d.data_original()
    field_2d = data[:, :, z_index]
    x = l_cell[Axis.X]
    y = l_cell[Axis.Y]
    return field_2d, x, y


def vis2d_field(
    scalar3d,
    z_index: int = 0,
    *,
    title: str = '',
    cmap: str = 'viridis',
    cmax: Optional[float] = None,
    cscale: float = 1.0,
    savepath: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the magnitude of a 2D field slice from a Scalar3d.

    This replaces MATLAB's ``vis2d(field, Axis.z, z_location, ...)``
    for 2D simulations where the z-axis is trivial.

    Parameters
    ----------
    scalar3d : Scalar3d
        The 3D scalar field to visualize.
    z_index : int, optional
        Z-slice index (default 0).
    title : str, optional
        Plot title.
    cmap : str, optional
        Matplotlib colormap (default 'viridis').
    cmax : float, optional
        Maximum color scale value. If None, auto-scaled.
    cscale : float, optional
        Scale factor applied to the field before plotting (default 1.0).
    savepath : str, optional
        If given, save the figure to this path as PNG.
    show : bool, optional
        If True, call ``plt.show()`` (default False).

    Returns
    -------
    fig : matplotlib Figure
        The figure object.
    """
    field_2d, x, y = _extract_2d_slice(scalar3d, z_index)
    magnitude = np.abs(field_2d) * cscale

    if cmax is None:
        cmax = np.max(magnitude) if magnitude.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        x, y, magnitude.T,
        cmap=cmap,
        shading='auto',
        vmin=0,
        vmax=cmax,
    )
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label='|Field|')

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f'Figure saved to {savepath}')

    if show:
        plt.show()

    return fig


def vis2d_real(
    scalar3d,
    z_index: int = 0,
    *,
    title: str = '',
    cmap: str = 'RdBu_r',
    savepath: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the real part of a 2D field slice from a Scalar3d.

    Parameters
    ----------
    scalar3d : Scalar3d
        The 3D scalar field to visualize.
    z_index : int, optional
        Z-slice index (default 0).
    title : str, optional
        Plot title.
    cmap : str, optional
        Matplotlib colormap (default 'RdBu_r' for diverging data).
    savepath : str, optional
        If given, save the figure to this path as PNG.
    show : bool, optional
        If True, call ``plt.show()`` (default False).

    Returns
    -------
    fig : matplotlib Figure
        The figure object.
    """
    field_2d, x, y = _extract_2d_slice(scalar3d, z_index)
    real_part = np.real(field_2d)

    vmax = np.max(np.abs(real_part)) if real_part.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        x, y, real_part.T,
        cmap=cmap,
        shading='auto',
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Re(Field)')

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f'Figure saved to {savepath}')

    if show:
        plt.show()

    return fig


# -----------------------------------------------------------------------
# Full visualization with structure overlay
# -----------------------------------------------------------------------

def vis2d_full(
    scalar3d,
    obj_array=None,
    src_array=None,
    extra=None,
    z_index: int = 0,
    *,
    withabs: bool = False,
    phase: float = 0.0,
    cscale: float = 1.0,
    cmax: Optional[float] = None,
    withinterp: bool = True,
    withpml: bool = False,
    title: str = '',
    savepath: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a 2D field slice matching MATLAB's ``vis2d`` / ``Painter2d``.

    This faithfully reproduces MATLAB's visualization pipeline:

    1. **``data_expanded()``** — uses the full array including ghost/boundary
       points (not ``data_original()``).
    2. **``interpn`` interpolation** — interpolates from the expanded grid to
       the plotting grid returned by ``grid3d.lplot()``.
    3. **Field transform** (matching ``Painter2d.prep_data``):

       * ``withabs=True``  → ``abs(field)``,  ``hot``  colormap, range ``[0, cscale*cmax]``
       * ``withabs=False`` → ``real(e^{i*phase} * field)``, ``b2r`` colormap,
         range ``[-cscale*cmax, cscale*cmax]``

    4. **Structure overlay** — material regions from ``extra['eps']`` are
       filled semi-transparent gray with white boundary contours.
    5. **Source markers** — red stars at source locations.

    Parameters
    ----------
    scalar3d : Scalar3d
        The 3D scalar field to visualize.
    obj_array : list, optional
        List of EMObject (unused; structure comes from eps).
    src_array : list, optional
        List of Source objects for marker placement.
    extra : dict, optional
        The ``extra`` dict returned by :func:`maxwell_run`. Must contain
        ``'eps'`` (``{Axis: Scalar3d}``) and optionally ``'grid3d'``.
    z_index : int, optional
        Z-slice index into the expanded array (default 0).
    withabs : bool, optional
        If True, plot ``abs(field)`` with ``hot`` colormap.
        If False (default), plot ``real(field)`` with ``b2r`` colormap.
    phase : float, optional
        Phase angle (radians) applied before taking real part when
        ``withabs=False`` (default 0).
    cscale : float, optional
        Scale factor for the color range (default 1.0).
    cmax : float, optional
        Maximum color value. If None, defaults to ``max(abs(field))``.
    withinterp : bool, optional
        Whether to interpolate to plotting grid (default True).
    withpml : bool, optional
        If True, include PML regions in the plot (default False).
    title : str, optional
        Plot title.
    savepath : str, optional
        If given, save the figure to this path as PNG.
    show : bool, optional
        If True, call ``plt.show()`` (default False).

    Returns
    -------
    fig : matplotlib Figure
        The figure object.
    """
    # --- Prepare field data (data_expanded + interpn, matching Painter2d) ---
    C, x, y = _prep_field_2d(scalar3d, z_index, withinterp=withinterp, withpml=withpml)

    # --- Compute cmax (defaults to max(abs(C)) if None, matching MATLAB) ---
    maxamp = np.max(np.abs(C)) if C.size > 0 else 1.0
    if cmax is None or (isinstance(cmax, float) and np.isnan(cmax)):
        cmax = maxamp

    # --- Field transform + colormap (matching Painter2d.set_caxis) ---
    if withabs:
        plot_data = np.abs(C)
        cmap = 'hot'
        vmin = 0.0
        vmax = cscale * cmax
        cb_label = '|E|'
    else:
        plot_data = np.real(np.exp(1j * phase) * C)
        cmap = _b2r_colormap()
        vmin = -cscale * cmax
        vmax = cscale * cmax
        cb_label = 'Re(E)'

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Field colormap ---
    # Use imshow for symmetric rendering.  pcolormesh with non-uniform
    # grid spacing (dx=5 at boundaries, dx=10 interior) introduces
    # rendering asymmetry.  imshow treats the data as a regular image.
    # Compute cell widths so the extent exactly matches the data positions.
    dx_cell = (x[-1] - x[0]) / max(len(x) - 1, 1)
    dy_cell = (y[-1] - y[0]) / max(len(y) - 1, 1)
    im = ax.imshow(
        plot_data.T,
        extent=(float(x[0] - dx_cell / 2), float(x[-1] + dx_cell / 2),
                float(y[0] - dy_cell / 2), float(y[-1] + dy_cell / 2)),
        origin='lower',
        aspect='equal',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='bilinear',
    )

    # --- Structure overlay from eps ---
    if extra is not None and 'eps' in extra:
        eps_dict = extra['eps']
        # Use the X-component as representative structure map
        eps_scalar3d = eps_dict.get(Axis.X)
        if eps_scalar3d is not None:
            eps_C, eps_x, eps_y = _prep_field_2d(
                eps_scalar3d, z_index, withinterp=withinterp, withpml=withpml,
            )
            eps_real = np.real(eps_C)

            # Binary mask: True where material differs from vacuum (eps≈1)
            struct_mask = (np.abs(eps_real - 1.0) > 0.1).astype(float)

            if struct_mask.max() > 0:
                # Draw crisp white boundary lines around material regions.
                # No fill — the field must remain visible inside materials.
                ax.contour(
                    eps_x, eps_y, struct_mask.T,
                    levels=[0.5],
                    colors='white',
                    linewidths=1.0,
                    alpha=0.5,
                )

    # --- Source markers ---
    if src_array is not None:
        for src in src_array:
            try:
                loc = getattr(src, 'location', None)
                if loc is not None and hasattr(loc, '__len__') and len(loc) >= 2:
                    ax.plot(loc[0], loc[1], 'r*', markersize=12,
                            markeredgecolor='white', markeredgewidth=0.5,
                            zorder=10)
            except Exception:
                pass

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cb_label)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f'Figure saved to {savepath}')

    if show:
        plt.show()

    return fig


# -----------------------------------------------------------------------
# Power Flux Calculations
# -----------------------------------------------------------------------

def _get_field_array_and_locs(scalar3d_or_array):
    """Extract the expanded numpy array and per-axis coordinates.

    Uses ``data_expanded()`` (includes ghost/boundary points) so that the
    coordinates match the array indices exactly, as in MATLAB's
    ``data_expanded``.
    """
    if hasattr(scalar3d_or_array, 'data_expanded'):
        arr, l_cell = scalar3d_or_array.data_expanded()
        return arr, {a: np.asarray(l_cell[a], dtype=float) for a in l_cell}
    raise ValueError('Field should be a Scalar3d (with data_expanded).')


def _patch_flux(E, H, normal_axis: Axis, location: float, grid3d) -> float:
    """Poynting flux through an axis-normal plane, with dl weights (2D sim).

    Interpolates all needed E and H components onto the plane (MATLAB-style
    ``interpn``) and integrates ``S = 0.5 Re(E x H*)`` over the patch area,
    including the z-thickness (``t``), which is required for 2D simulations.
    """
    from scipy.interpolate import RegularGridInterpolator

    def comp_grid(comp):
        arr, locs = _get_field_array_and_locs(comp)
        return (arr,
                locs[Axis.X][:arr.shape[0]],
                locs[Axis.Y][:arr.shape[1]],
                locs[Axis.Z][:arr.shape[2]])


    # Determine tangential coordinate arrays from the E-field prim grid
    # via data_expanded of E components.
    _, xg0, _, _ = comp_grid(E[Axis.Z])
    _, _, yg0, _ = comp_grid(E[Axis.Z])
    _, _, _, zg0 = comp_grid(E[Axis.Z])

    # z is the trivial 2D axis: use a single layer at the center of the
    # physical slab with weight equal to the slab thickness t. (The
    # expanded z-grid contains ghost points, e.g. [-0.5, 0.5, 1.5] for
    # z in [0, 1]; integrating over them would multiply the flux.)
    lz = grid3d.l[(Axis.Z, GT.PRIM)] if hasattr(grid3d, 'l') else None
    lz = np.asarray(lz, dtype=float)
    if lz.size >= 2:
        z_center = 0.5 * (lz[0] + lz[-1])
        t_slab = float(lz[-1] - lz[0])
    else:
        z_center = float(np.mean(zg0))
        t_slab = 1.0
    z_mesh = np.array([z_center])

    def interp(comp, fixed_axis, fixed_val, t1_grid, t2_grid):
        arr, xg, yg, zg = comp_grid(comp)
        rgi = RegularGridInterpolator((xg, yg, zg), arr, bounds_error=False,
                                      fill_value=None)
        T1, T2 = np.meshgrid(t1_grid, t2_grid, indexing='ij')
        if fixed_axis == Axis.X:
            F = np.full_like(T1, fixed_val)
            pts = np.stack([F, T1, T2], axis=-1)
        else:  # Axis.Y fixed
            F = np.full_like(T1, fixed_val)
            pts = np.stack([T1, F, T2], axis=-1)
        return rgi(pts)

    def dl_weights(g):
        g = np.asarray(g, dtype=float)
        w = np.empty_like(g)
        w[1:-1] = (g[2:] - g[:-2]) / 2.0
        w[0] = g[1] - g[0]
        w[-1] = g[-1] - g[-2]
        return w

    if normal_axis == Axis.X:
        # Sx = 0.5 Re(Ey Hz* - Ez Hy*) on the plane x = location
        Ey = interp(E[Axis.Y], Axis.X, location, yg0, z_mesh)
        Ez = interp(E[Axis.Z], Axis.X, location, yg0, z_mesh)
        Hy = interp(H[Axis.Y], Axis.X, location, yg0, z_mesh)
        Hz = interp(H[Axis.Z], Axis.X, location, yg0, z_mesh)
        S = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
        wy = dl_weights(yg0)
        return float(np.sum(S * wy[:, None]) * t_slab)
    elif normal_axis == Axis.Y:
        # Sy = 0.5 Re(Ez Hx* - Ex Hz*) on the plane y = location
        Ez = interp(E[Axis.Z], Axis.Y, location, xg0, z_mesh)
        Ex = interp(E[Axis.X], Axis.Y, location, xg0, z_mesh)
        Hx = interp(H[Axis.X], Axis.Y, location, xg0, z_mesh)
        Hz = interp(H[Axis.Z], Axis.Y, location, xg0, z_mesh)
        S = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
        wx = dl_weights(xg0)
        return float(np.sum(S * wx[:, None]) * t_slab)
    else:
        raise NotImplementedError('powerflux for z-normal planes is not '
                                  'needed in 2D examples.')


def powerflux_box(
    E: Dict[Axis, object],
    H: Dict[Axis, object],
    box_bounds: np.ndarray,
    grid3d=None,
) -> float:
    """
    Calculate the net Poynting flux through a rectangular box (2D: 4 faces).

    Integrates the outward Poynting vector ``S = 0.5 Re(E x H*)`` through
    each side face of the axis-aligned box and returns the net outward flux.
    For a lossless scatterer inside the box this is ~0.

    Parameters
    ----------
    E : dict
        ``{Axis: Scalar3d}`` E-field components.
    H : dict
        ``{Axis: Scalar3d}`` H-field components.
    box_bounds : ndarray
        Box bounds as shape (3, 2): ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.
    grid3d : Grid3d, optional
        Grid object (kept for backward compatibility; the grid is taken
        from the fields themselves).

    Returns
    -------
    float
        Net outward power flux through the box surface.
    """
    if grid3d is None:
        grid3d = E[Axis.X].grid3d

    xmin, xmax = box_bounds[0]
    ymin, ymax = box_bounds[1]

    flux = 0.0
    flux -= _patch_flux(E, H, Axis.X, xmin, grid3d)   # left face: outward = -x
    flux += _patch_flux(E, H, Axis.X, xmax, grid3d)   # right face: outward = +x
    flux -= _patch_flux(E, H, Axis.Y, ymin, grid3d)   # bottom face: outward = -y
    flux += _patch_flux(E, H, Axis.Y, ymax, grid3d)   # top face: outward = +y
    return flux


def powerflux_patch(
    E: Dict[Axis, object],
    H: Dict[Axis, object],
    normal_axis: Axis,
    location: float,
    grid3d=None,
) -> float:
    """
    Calculate the Poynting flux through a planar patch.

    Interpolates the field components onto the plane at ``location`` and
    integrates the Poynting vector component along ``normal_axis`` over the
    patch area, including proper grid-cell (dl) weights and the z-thickness.

    Parameters
    ----------
    E : dict
        ``{Axis: Scalar3d}`` E-field components.
    H : dict
        ``{Axis: Scalar3d}`` H-field components.
    normal_axis : Axis
        Axis normal to the patch (the propagation direction).
    location : float
        Position of the patch along ``normal_axis``.
    grid3d : Grid3d, optional
        Grid object for coordinate lookup. If None, uses the grid from
        the first E-field component.

    Returns
    -------
    float
        Power flux through the patch.
    """
    if grid3d is None:
        grid3d = E[Axis.X].grid3d
    return _patch_flux(E, H, normal_axis, location, grid3d)
