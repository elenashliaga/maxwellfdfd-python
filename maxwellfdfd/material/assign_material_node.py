"""
Material assignment to grid nodes.

This module provides functions for assigning material properties to grid nodes
based on object shapes and materials.
"""

from typing import List, Dict, Optional, Tuple

import numpy as np

from ..base import Axis, GT, Sign
from ..base.utils import chkarg
from ..grid.grid3d import Grid3d
from ..shape.shape import Shape
from ..shape.box import Box
from .material import Material
from ._emobject import EMObject


def assign_material_node(
    grid3d: Grid3d,
    object_array: List[EMObject],
    eps_node_cell: Optional[Dict[Axis, np.ndarray]] = None,
    mu_node_cell: Optional[Dict[Axis, np.ndarray]] = None
) -> Tuple[Dict[Axis, np.ndarray], Dict[Axis, np.ndarray]]:
    """
    Assign material properties to grid nodes.
    
    Assigns electric permittivity and magnetic permeability to grid nodes
    based on the shapes and materials of the given objects.
    
    Parameters
    ----------
    grid3d : Grid3d
        3D grid object.
    object_array : list of EMObject
        List of electromagnetic objects (shape + material combinations).
    eps_node_cell : dict of Axis to numpy.ndarray, optional
        Dictionary mapping axes to permittivity arrays. If not provided,
        initialized with NaN arrays. Default is None.
    mu_node_cell : dict of Axis to numpy.ndarray, optional
        Dictionary mapping axes to permeability arrays. If not provided,
        initialized with NaN arrays. Default is None.
    
    Returns
    -------
    eps_node_cell : dict of Axis to numpy.ndarray
        Dictionary mapping axes to permittivity arrays at grid nodes.
    mu_node_cell : dict of Axis to numpy.ndarray
        Dictionary mapping axes to permeability arrays at grid nodes.
    
    Notes
    -----
    - For Box shapes, material is assigned using direct indexing.
    - For non-Box shapes, material is assigned using point containment checks.
    - Material properties are assigned as diagonal elements of the tensor.
    
    Examples
    --------
    >>> from maxwellfdfd.grid import Grid3d
    >>> from maxwellfdfd.shape import Box
    >>> from maxwellfdfd.material import Material
    >>> from ._emobject import EMObject
    >>> grid = Grid3d(...)
    >>> mat = Material('dielectric', 'b', 4.0)
    >>> shape = Box(...)
    >>> obj = EMObject(shape, mat)
    >>> eps, mu = assign_material_node(grid, [obj])
    
    See Also
    --------
    mean_material_node : Compute mean material at grid nodes
    """
    # Validate grid3d
    chkarg(isinstance(grid3d, Grid3d),
           '"grid3d" should be instance of Grid3d.')
    
    # Validate object_array
    # NOTE: EMObject lives in the io module, which depends on material, so we
    # cannot import it here without a circular import. Duck-type instead: an
    # EMObject exposes both a .shape and a .material attribute.
    chkarg(isinstance(object_array, list) and
           all(hasattr(obj, 'shape') and hasattr(obj, 'material') for obj in object_array),
           '"object_array" should be a list of EMObject instances.')
    
    # Get grid dimensions
    N = grid3d.N  # Tuple of (Nx, Ny, Nz)
    
    # Initialize eps_node_cell if not provided
    if eps_node_cell is None:
        eps_node_cell = {
            Axis.X: np.full(N, np.nan, dtype=complex),
            Axis.Y: np.full(N, np.nan, dtype=complex),
            Axis.Z: np.full(N, np.nan, dtype=complex)
        }
    
    # Validate eps_node_cell
    axis_list = Axis.elems()
    # NOTE: grid3d.N may be an ndarray; comparing .shape == N yields an array,
    # which is ambiguous inside all(). Compare against a plain tuple instead.
    N_shape = tuple(int(n) for n in N)
    chkarg(isinstance(eps_node_cell, dict) and
           all(axis in eps_node_cell for axis in axis_list) and
           all(isinstance(eps_node_cell[axis], np.ndarray) for axis in axis_list) and
           all(eps_node_cell[axis].shape == N_shape for axis in axis_list),
           '"eps_node_cell" should be a dict with Axis keys and Nx-by-Ny-by-Nz arrays.')
    
    # Initialize mu_node_cell if not provided
    if mu_node_cell is None:
        mu_node_cell = {
            Axis.X: np.full(N, np.nan, dtype=complex),
            Axis.Y: np.full(N, np.nan, dtype=complex),
            Axis.Z: np.full(N, np.nan, dtype=complex)
        }
    
    # Validate mu_node_cell
    chkarg(isinstance(mu_node_cell, dict) and
           all(axis in mu_node_cell for axis in axis_list) and
           all(isinstance(mu_node_cell[axis], np.ndarray) for axis in axis_list) and
           all(mu_node_cell[axis].shape == N_shape for axis in axis_list),
           '"mu_node_cell" should be a dict with Axis keys and Nx-by-Ny-by-Nz arrays.')
    
    # Get dual grid locations (cell centers)
    ldual = {}
    for w in axis_list:
        # NOTE: Grid3d.l is keyed by (Axis, GT) tuples, not nested dicts.
        ldual[w] = grid3d.l[(w, GT.DUAL)]
    
    # Process each object
    for obj in object_array:
        shape = obj.shape
        material = obj.material
        
        # Find indices for each axis
        ind = {}
        for w in axis_list:
            # NOTE: Shape.bound is a property (ndarray), not a method.
            bn = shape.bound[int(w), int(Sign.N)]
            bp = shape.bound[int(w), int(Sign.P)]
            
            # Find first index where ldual >= bn
            in_idx = np.argmax(ldual[w] >= bn)
            if not (ldual[w][in_idx] >= bn):
                in_idx = 0
            
            # Find last index where ldual <= bp
            ip_idx = len(ldual[w]) - 1 - np.argmax(ldual[w][::-1] <= bp)
            if not (ldual[w][ip_idx] <= bp):
                ip_idx = len(ldual[w]) - 1
            
            ind[w] = slice(in_idx, ip_idx + 1)
        
        # Assign material properties
        if isinstance(shape, Box):
            # For Box shapes, use direct indexing
            for w in axis_list:
                # NOTE: Axis.value is the axis *name* (e.g. 'x'), not an int.
                w_idx = int(w)
                eps_node_cell[w][ind[Axis.X], ind[Axis.Y], ind[Axis.Z]] = material.eps[w_idx, w_idx]
                mu_node_cell[w][ind[Axis.X], ind[Axis.Y], ind[Axis.Z]] = material.mu[w_idx, w_idx]
        else:
            # For non-Box shapes, use point containment checks
            # Create coordinate grid
            X, Y, Z = np.meshgrid(
                ldual[Axis.X][ind[Axis.X]],
                ldual[Axis.Y][ind[Axis.Y]],
                ldual[Axis.Z][ind[Axis.Z]],
                indexing='ij'
            )
            
            # Check which points are inside the shape
            is_in = shape.contains(X, Y, Z)
            
            # Create logical index array
            ind_tf = np.zeros(N, dtype=bool)
            ind_tf[ind[Axis.X], ind[Axis.Y], ind[Axis.Z]] = is_in
            
            # Assign material properties
            for w in axis_list:
                # NOTE: Axis.value is the axis *name* (e.g. 'x'), not an int.
                w_idx = int(w)
                eps_node_cell[w][ind_tf] = material.eps[w_idx, w_idx]
                mu_node_cell[w][ind_tf] = material.mu[w_idx, w_idx]
    
    return eps_node_cell, mu_node_cell