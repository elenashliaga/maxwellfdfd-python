"""
Periodize shape utility function.

This module provides the periodize_shape function, which constructs
a periodic array of a given shape.
"""

from typing import Dict, List, Union
import numpy as np
from maxwellfdfd.base import Axis, chkarg
from .shape import Shape


def periodize_shape(
    shape: Shape,
    primitive_vec_cell: List[np.ndarray],
    boundshape: Shape
) -> List[Shape]:
    """
    Construct a periodic array of a given shape.
    
    This function creates a periodic array of shapes by translating the
    input shape according to primitive lattice vectors. The extent of
    the array is controlled by a bounding shape.
    
    Parameters
    ----------
    shape : Shape
        Instance of Shape to be repeated.
    primitive_vec_cell : list of ndarray
        Three primitive lattice vectors in the format of
        [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]].
    boundshape : Shape
        Shape that contains the centers of the translated shapes.
        Used to control the overall extent of the periodic array.
    
    Returns
    -------
    list of Shape
        List of Shape instances forming the periodic array.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base import Axis
    >>> from maxwellfdfd.shape import Box, CircularCylinder, periodize_shape
    >>> 
    >>> # Create a slab
    >>> t = 500  # thickness
    >>> slab = Box(np.array([[-5000, 5000], [-3000, 3000], [0, t]]))
    >>> 
    >>> # Create an array of circular cylinders
    >>> r = 100  # radius
    >>> rod = CircularCylinder(Axis.Z, t, np.array([0, 0, t/2]), r)
    >>> primitive_vecs = [
    ...     np.array([400, 0, 0]),
    ...     np.array([0, 400, 0]),
    ...     np.array([0, 0, t])
    ... ]
    >>> rod_array = periodize_shape(rod, primitive_vecs, slab)
    >>> len(rod_array)
    75
    """
    # Validate shape
    chkarg(isinstance(shape, Shape), '"shape" should be instance of Shape.')
    
    # Validate primitive_vec_cell
    chkarg(
        isinstance(primitive_vec_cell, list) and len(primitive_vec_cell) == Axis.count(),
        f'"primitive_vec_cell" should be list with {Axis.count()} elements.'
    )
    for vec in primitive_vec_cell:
        vec_arr = np.asarray(vec)
        chkarg(
            vec_arr.shape == (Axis.count(),) and np.isrealobj(vec_arr),
            f'elements of "primitive_vec_cell" should be length-{Axis.count()} arrays with real elements.'
        )
    
    # Validate boundshape
    chkarg(isinstance(boundshape, Shape), '"boundshape" should be instance of Shape.')
    
    # Get shape parameters
    lsf0 = shape.lsf  # level set function
    lprim0 = shape.lprim  # primary grid planes
    c0 = shape.cb_center  # center of circumbox
    dl_max0 = shape.dl_max  # dl_max
    
    # Get bounding box of boundshape
    box_bound = boundshape.bound
    
    # Create corners of bounding box
    Xbox, Ybox, Zbox = np.meshgrid(
        box_bound[int(Axis.X), :],
        box_bound[int(Axis.Y), :],
        box_bound[int(Axis.Z), :],
        indexing='ij'
    )
    corners = np.column_stack([Xbox.flatten(), Ybox.flatten(), Zbox.flatten()]).T
    
    # Create matrix of primitive vectors (columns are primitive vectors)
    A = np.column_stack([primitive_vec_cell[0], primitive_vec_cell[1], primitive_vec_cell[2]])
    
    # Calculate all n = [nx, ny, nz] that make A * n + c0 corners
    # A * n + c0 = corner => n = A^{-1} * (corner - c0)
    try:
        n_ext = np.linalg.solve(A, corners - c0.reshape(-1, 1))
    except np.linalg.LinAlgError:
        raise ValueError('Primitive vectors are linearly dependent.')
    
    # Calculate extreme n values
    nmax = np.floor(np.max(n_ext, axis=1) - np.finfo(float).eps)
    nmin = np.ceil(np.min(n_ext, axis=1) + np.finfo(float).eps)
    
    # Create shape array
    shape_array: List[Shape] = []
    
    # Iterate over all possible translations
    for nz in range(int(nmin[int(Axis.Z)]), int(nmax[int(Axis.Z)]) + 1):
        for ny in range(int(nmin[int(Axis.Y)]), int(nmax[int(Axis.Y)]) + 1):
            for nx in range(int(nmin[int(Axis.X)]), int(nmax[int(Axis.X)]) + 1):
                n = np.array([nx, ny, nz])
                R = A @ n  # lattice vector
                
                # Check if center is in boundshape
                loc = [c0[int(Axis.X)] + R[int(Axis.X)],
                       c0[int(Axis.Y)] + R[int(Axis.Y)],
                       c0[int(Axis.Z)] + R[int(Axis.Z)]]
                
                if boundshape.contains(loc[0], loc[1], loc[2]):
                    # Create translated lprim
                    lprim: Dict[Axis, np.ndarray] = {}
                    for axis in Axis.elems():
                        lprim[axis] = lprim0[axis] + R[int(axis)]
                    
                    # Create translated level set function
                    def lsf_translated(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
                        return lsf0(x - R[int(Axis.X)], y - R[int(Axis.Y)], z - R[int(Axis.Z)])
                    
                    # Create shape and add to array
                    translated_shape = Shape(lprim, lsf_translated, dl_max0)
                    shape_array.append(translated_shape)
    
    return shape_array