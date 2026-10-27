"""
Mean material computation at grid nodes.

This module provides functions for computing mean material properties
at grid nodes using arithmetic or harmonic averaging.
"""

from typing import Dict

import numpy as np

from ..base import Axis, GT
from ..base.utils import chkarg
from ..grid.grid3d import Grid3d


def mean_material_node(
    grid3d: Grid3d,
    gt: GT,
    material_node: Dict[Axis, np.ndarray]
) -> Dict[Axis, np.ndarray]:
    """
    Compute mean material at grid nodes.
    
    Computes material parameters at grid nodes by averaging the material
    properties at surrounding nodes. Uses arithmetic mean for primary grid
    and harmonic mean for dual grid.
    
    Parameters
    ----------
    grid3d : Grid3d
        3D grid object.
    gt : GT
        Grid type (GT.PRIM for primary, GT.DUAL for dual).
    material_node : dict of Axis to numpy.ndarray
        Dictionary mapping axes to material arrays at grid nodes.
        Each array should have shape (Nx, Ny, Nz).
    
    Returns
    -------
    material_cell : dict of Axis to numpy.ndarray
        Dictionary mapping axes to mean material arrays.
        Each array has shape (Nx, Ny, Nz).
    
    Notes
    -----
    - For primary grid (GT.PRIM), arithmetic mean is used.
    - For dual grid (GT.DUAL), harmonic mean is used.
    - The material_node arrays are expanded with ghost nodes before averaging.
    
    Examples
    --------
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.base import GT
    >>> grid = Grid3d(...)
    >>> material_node = {Axis.X: eps_x, Axis.Y: eps_y, Axis.Z: eps_z}
    >>> material_cell = mean_material_node(grid, GT.PRIM, material_node)
    
    See Also
    --------
    assign_material_node : Assign material to grid nodes
    """
    # Validate grid3d
    chkarg(isinstance(grid3d, Grid3d),
           '"grid3d" should be instance of Grid3d.')
    
    # Validate gt
    chkarg(isinstance(gt, GT),
           '"gt" should be instance of GT.')
    
    # Get grid dimensions
    N = grid3d.N  # ndarray of (Nx, Ny, Nz)
    # NOTE: comparing .shape == N (ndarray) yields an array, ambiguous in all().
    N_shape = tuple(int(n) for n in N)
    
    # Validate material_node
    axis_list = Axis.elems()
    chkarg(isinstance(material_node, dict) and
           all(axis in material_node for axis in axis_list) and
           all(isinstance(material_node[axis], np.ndarray) for axis in axis_list) and
           all(material_node[axis].shape == N_shape for axis in axis_list),
           '"material_node" should be a dict with Axis keys and Nx-by-Ny-by-Nz arrays.')
    
    # Expand material_node arrays with ghost nodes
    material_node_expanded = {}
    for w in axis_list:
        material_node_expanded[w] = _expand_node_array(grid3d, material_node[w])
    
    # Compute mean based on grid type
    if gt == GT.PRIM:
        # Arithmetic mean for primary grid
        material_cell = _arithmetic_mean_material_node(material_node_expanded)
    else:  # gt == GT.DUAL
        # Harmonic mean for dual grid
        material_cell = _harmonic_mean_material_node(material_node_expanded)
    
    return material_cell


def _expand_node_array(
    grid3d: Grid3d,
    arr: np.ndarray
) -> np.ndarray:
    """
    Expand array with ghost nodes.
    
    Expands an array by adding one layer of ghost nodes on each side.
    The ghost nodes are filled with the values from the nearest interior nodes.
    
    Parameters
    ----------
    grid3d : Grid3d
        3D grid object.
    arr : numpy.ndarray
        Input array with shape (Nx, Ny, Nz).
    
    Returns
    -------
    numpy.ndarray
        Expanded array with shape (Nx+2, Ny+2, Nz+2).
    """
    Nx, Ny, Nz = grid3d.N
    
    # Create expanded array
    expanded = np.full((Nx + 2, Ny + 2, Nz + 2), np.nan, dtype=arr.dtype)
    
    # Copy interior values
    expanded[1:-1, 1:-1, 1:-1] = arr
    
    # Fill ghost nodes with nearest interior values
    # x-direction
    expanded[0, 1:-1, 1:-1] = arr[0, :, :]  # x = 0
    expanded[-1, 1:-1, 1:-1] = arr[-1, :, :]  # x = Nx+1
    
    # y-direction
    expanded[1:-1, 0, 1:-1] = arr[:, 0, :]  # y = 0
    expanded[1:-1, -1, 1:-1] = arr[:, -1, :]  # y = Ny+1
    
    # z-direction
    expanded[1:-1, 1:-1, 0] = arr[:, :, 0]  # z = 0
    expanded[1:-1, 1:-1, -1] = arr[:, :, -1]  # z = Nz+1
    
    # Fill corners and edges (average of adjacent faces)
    # Corners (8)
    expanded[0, 0, 0] = (arr[0, 0, 0] + arr[0, 0, 0] + arr[0, 0, 0]) / 3
    expanded[0, 0, -1] = (arr[0, 0, -1] + arr[0, 0, -1] + arr[0, 0, -1]) / 3
    expanded[0, -1, 0] = (arr[0, -1, 0] + arr[0, -1, 0] + arr[0, -1, 0]) / 3
    expanded[0, -1, -1] = (arr[0, -1, -1] + arr[0, -1, -1] + arr[0, -1, -1]) / 3
    expanded[-1, 0, 0] = (arr[-1, 0, 0] + arr[-1, 0, 0] + arr[-1, 0, 0]) / 3
    expanded[-1, 0, -1] = (arr[-1, 0, -1] + arr[-1, 0, -1] + arr[-1, 0, -1]) / 3
    expanded[-1, -1, 0] = (arr[-1, -1, 0] + arr[-1, -1, 0] + arr[-1, -1, 0]) / 3
    expanded[-1, -1, -1] = (arr[-1, -1, -1] + arr[-1, -1, -1] + arr[-1, -1, -1]) / 3
    
    # Edges (12)
    # x-edges
    expanded[0, 0, 1:-1] = (arr[0, 0, :] + arr[0, 0, :]) / 2
    expanded[0, -1, 1:-1] = (arr[0, -1, :] + arr[0, -1, :]) / 2
    expanded[-1, 0, 1:-1] = (arr[-1, 0, :] + arr[-1, 0, :]) / 2
    expanded[-1, -1, 1:-1] = (arr[-1, -1, :] + arr[-1, -1, :]) / 2
    
    # y-edges
    expanded[0, 1:-1, 0] = (arr[0, :, 0] + arr[0, :, 0]) / 2
    expanded[0, 1:-1, -1] = (arr[0, :, -1] + arr[0, :, -1]) / 2
    expanded[-1, 1:-1, 0] = (arr[-1, :, 0] + arr[-1, :, 0]) / 2
    expanded[-1, 1:-1, -1] = (arr[-1, :, -1] + arr[-1, :, -1]) / 2
    
    # z-edges
    expanded[1:-1, 0, 0] = (arr[:, 0, 0] + arr[:, 0, 0]) / 2
    expanded[1:-1, 0, -1] = (arr[:, 0, -1] + arr[:, 0, -1]) / 2
    expanded[1:-1, -1, 0] = (arr[:, -1, 0] + arr[:, -1, 0]) / 2
    expanded[1:-1, -1, -1] = (arr[:, -1, -1] + arr[:, -1, -1]) / 2
    
    return expanded


def _arithmetic_mean_material_node(
    material_node: Dict[Axis, np.ndarray]
) -> Dict[Axis, np.ndarray]:
    """
    Compute arithmetic mean of material at grid nodes.
    
    Computes the arithmetic mean of material properties at grid nodes
    for the primary grid. This is used for field components on the
    primary grid.
    
    Parameters
    ----------
    material_node : dict of Axis to numpy.ndarray
        Dictionary mapping axes to expanded material arrays.
        Each array should have shape (Nx+2, Ny+2, Nz+2).
    
    Returns
    -------
    material_edge_cell : dict of Axis to numpy.ndarray
        Dictionary mapping axes to arithmetic mean material arrays.
        Each array has shape (Nx, Ny, Nz).
    
    Notes
    -----
    The arithmetic mean is computed by averaging the material properties
    at the four nodes surrounding each edge center. For example, for
    the x-component, the mean at (i, j, k) is the average of the values
    at (i, j, k), (i, j-1, k), (i, j, k-1), and (i, j-1, k-1).
    """
    material_edge_cell = {}
    
    # X component: average of (i,j,k), (i,j-1,k), (i,j,k-1), (i,j-1,k-1)
    material_edge_cell[Axis.X] = (
        material_node[Axis.X][1:-1, 1:-1, 1:-1] +
        material_node[Axis.X][1:-1, :-2, 1:-1] +
        material_node[Axis.X][1:-1, 1:-1, :-2] +
        material_node[Axis.X][1:-1, :-2, :-2]
    ) / 4.0
    
    # Y component: average of (i,j,k), (i-1,j,k), (i,j,k-1), (i-1,j,k-1)
    material_edge_cell[Axis.Y] = (
        material_node[Axis.Y][1:-1, 1:-1, 1:-1] +
        material_node[Axis.Y][1:-1, 1:-1, :-2] +
        material_node[Axis.Y][:-2, 1:-1, 1:-1] +
        material_node[Axis.Y][:-2, 1:-1, :-2]
    ) / 4.0
    
    # Z component: average of (i,j,k), (i-1,j,k), (i,j-1,k), (i-1,j-1,k)
    material_edge_cell[Axis.Z] = (
        material_node[Axis.Z][1:-1, 1:-1, 1:-1] +
        material_node[Axis.Z][:-2, 1:-1, 1:-1] +
        material_node[Axis.Z][1:-1, :-2, 1:-1] +
        material_node[Axis.Z][:-2, :-2, 1:-1]
    ) / 4.0
    
    return material_edge_cell


def _harmonic_mean_material_node(
    material_node: Dict[Axis, np.ndarray]
) -> Dict[Axis, np.ndarray]:
    """
    Compute harmonic mean of material at grid nodes.
    
    Computes the harmonic mean of material properties at grid nodes
    for the dual grid. This is used for field components on the
    dual grid.
    
    Parameters
    ----------
    material_node : dict of Axis to numpy.ndarray
        Dictionary mapping axes to expanded material arrays.
        Each array should have shape (Nx+2, Ny+2, Nz+2).
    
    Returns
    -------
    material_face_cell : dict of Axis to numpy.ndarray
        Dictionary mapping axes to harmonic mean material arrays.
        Each array has shape (Nx, Ny, Nz).
    
    Notes
    -----
    The harmonic mean is computed by averaging the reciprocals of the
    material properties at the two nodes on either side of each face.
    For example, for the x-component, the mean at (i, j, k) is the
    harmonic mean of the values at (i-1, j, k) and (i, j, k).
    
    The harmonic mean is used for the dual grid because it preserves
    the continuity of the normal component of the displacement field
    across material interfaces.
    """
    material_face_cell = {}
    
    # X component: harmonic mean of (i-1,j,k) and (i,j,k)
    material_face_cell[Axis.X] = 2.0 / (
        1.0 / material_node[Axis.X][:-2, 1:-1, 1:-1] +
        1.0 / material_node[Axis.X][1:-1, 1:-1, 1:-1]
    )
    
    # Y component: harmonic mean of (i,j-1,k) and (i,j,k)
    material_face_cell[Axis.Y] = 2.0 / (
        1.0 / material_node[Axis.Y][1:-1, :-2, 1:-1] +
        1.0 / material_node[Axis.Y][1:-1, 1:-1, 1:-1]
    )
    
    # Z component: harmonic mean of (i,j,k-1) and (i,j,k)
    material_face_cell[Axis.Z] = 2.0 / (
        1.0 / material_node[Axis.Z][1:-1, 1:-1, :-2] +
        1.0 / material_node[Axis.Z][1:-1, 1:-1, 1:-1]
    )
    
    return material_face_cell