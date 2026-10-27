"""
Utility functions for MaxwellFDFD base module.

This module provides utility functions for argument validation, type checking,
and array manipulation used throughout MaxwellFDFD.
"""

import numpy as np
from typing import Any, List, Tuple, Union
from scipy.sparse import spdiags, spmatrix


def chkarg(truth: bool, errmsg: str, *args) -> None:
    """
    Throw an exception if the argument is False.
    
    Parameters
    ----------
    truth : bool
        Condition to check. If False, an exception is raised.
    errmsg : str
        Error message (can be a format string).
    *args
        Additional arguments for format string.
    
    Raises
    ------
    ValueError
        If truth is False.
    
    Examples
    --------
    >>> chkarg(True, 'No error')
    >>> chkarg(False, 'Value %d is invalid', 42)
    Traceback (most recent call last):
        ...
    ValueError: Wrong argument: Value 42 is invalid
    """
    if args:
        errmsg = errmsg % args
    
    if not truth:
        raise ValueError(f"Wrong argument: {errmsg}")


def create_spdiag(vec: np.ndarray) -> spmatrix:
    """
    Create a sparse diagonal matrix.
    
    Parameters
    ----------
    vec : array_like
        Vector of diagonal elements.
    
    Returns
    -------
    scipy.sparse.spmatrix
        Sparse diagonal matrix.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base.utils import create_spdiag
    >>> D = create_spdiag(np.array([1, 2, 3]))
    >>> D.toarray()
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    """
    vec_flat = np.asarray(vec).flatten()
    n = len(vec_flat)
    return spdiags(vec_flat, 0, n, n)


def expand2mat(obj: Any, nrow: int, ncol: int) -> np.ndarray:
    """
    Expand object to a matrix of specified size.
    
    Parameters
    ----------
    obj : scalar, array_like, or matrix
        Object to expand.
    nrow : int
        Number of rows in output matrix.
    ncol : int
        Number of columns in output matrix.
    
    Returns
    -------
    ndarray
        Expanded matrix of shape (nrow, ncol).
    
    Raises
    ------
    ValueError
        If obj cannot be expanded to the specified size.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base.utils import expand2mat
    >>> expand2mat(5, 2, 3)
    array([[5, 5, 5],
           [5, 5, 5]])
    >>> expand2mat([1, 2], 2, 3)
    array([[1, 1, 1],
           [2, 2, 2]])
    """
    obj_arr = np.asarray(obj)
    
    # NOTE: after np.asarray(), a Python scalar becomes a 0-d array, for which
    # np.isscalar() returns False. Use ndim == 0 to detect scalars robustly.
    if obj_arr.ndim == 0:
        return np.full((nrow, ncol), obj_arr.item())
    elif obj_arr.ndim == 1:
        chkarg(len(obj_arr) == nrow, 'length of "obj" array should be "nrow".')
        if obj_arr.shape[0] == 1:  # row vector
            obj_arr = obj_arr.reshape(-1, 1)
        return np.tile(obj_arr.reshape(-1, 1), (1, ncol))
    elif obj_arr.ndim == 2:
        chkarg(obj_arr.shape == (nrow, ncol),
               'size of "obj" should be "nrow"-by-"ncol".')
        return obj_arr
    else:
        raise ValueError('obj must be scalar, vector, or 2D matrix')


def expand2row(obj: Any, n: int) -> np.ndarray:
    """
    Expand object to a row vector of specified length.
    
    Parameters
    ----------
    obj : scalar or array_like
        Object to expand.
    n : int
        Length of output row vector.
    
    Returns
    -------
    ndarray
        Row vector of shape (1, n).
    
    Raises
    ------
    ValueError
        If obj cannot be expanded to the specified length.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base.utils import expand2row
    >>> expand2row(5, 3)
    array([[5, 5, 5]])
    >>> expand2row([1, 2, 3], 3)
    array([[1, 2, 3]])
    """
    obj_arr = np.asarray(obj)
    
    # NOTE: after np.asarray(), a Python scalar becomes a 0-d array, for which
    # np.isscalar() returns False. Use ndim == 0 to detect scalars robustly.
    if obj_arr.ndim == 0:
        return np.full((1, n), obj_arr.item())
    # Flatten 2-D row/column vectors (e.g. shape (1,n) or (n,1)) to 1-D, as
    # MATLAB does when expanding to a row vector.
    if obj_arr.ndim == 2 and 1 in obj_arr.shape:
        obj_arr = obj_arr.reshape(-1)
    if obj_arr.ndim == 1:
        if obj_arr.size == 1:
            return np.full((1, n), obj_arr[0])
        chkarg(obj_arr.size == n, '"obj" should be length-"n" vector.')
        return obj_arr.reshape(1, -1)
    raise ValueError('obj must be scalar or 1D vector')


def is2dlike(N: np.ndarray) -> Tuple[bool, Any]:
    """
    Check if the grid is 2D-like (one dimension is much smaller than others).
    
    A grid is considered 2D-like if one dimension is less than 10% of the
    other two dimensions.
    
    Parameters
    ----------
    N : array_like
        Array of grid dimensions (length 3 for 3D).
    
    Returns
    -------
    truth : bool
        True if grid is 2D-like.
    normal_axis : Axis
        The axis corresponding to the smallest dimension.
    
    Raises
    ------
    ValueError
        If N is not a length-3 array with integer elements.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base.utils import is2dlike
    >>> is2dlike(np.array([100, 100, 5]))
    (True, <Axis.Z: 'z'>)
    >>> is2dlike(np.array([100, 100, 100]))
    (False, <Axis.X: 'x'>)
    """
    from .axis import Axis
    
    chkarg(bool(isinstance(N, np.ndarray) and N.shape == (3,) and
           np.all(N == N.astype(int))),
           '"N" should be length-3 array with integer elements.')
    
    imin = int(np.argmin(N))
    all_elems = [Axis.X, Axis.Y, Axis.Z]
    n = all_elems[imin]
    p, q, _ = n.cycle()
    
    truth = bool((N[int(n)] / N[int(p)] < 0.1) and (N[int(n)] / N[int(q)] < 0.1))
    return truth, n


def isexpandable2mat(obj: Any, nrow: int, ncol: int) -> bool:
    """
    Check if object can be expanded to a matrix of specified size.
    
    Parameters
    ----------
    obj : Any
        Object to check.
    nrow : int
        Number of rows in target matrix.
    ncol : int
        Number of columns in target matrix.
    
    Returns
    -------
    bool
        True if obj can be expanded to (nrow, ncol) matrix.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base.utils import isexpandable2mat
    >>> isexpandable2mat(5, 2, 3)
    True
    >>> isexpandable2mat([1, 2], 2, 3)
    True
    >>> isexpandable2mat([1, 2], 3, 3)
    False
    """
    obj_arr = np.asarray(obj)
    
    # NOTE: after np.asarray(), a Python scalar becomes a 0-d array, for which
    # np.isscalar() returns False. Use ndim == 0 to detect scalars robustly.
    if obj_arr.ndim == 0:
        return True
    elif obj_arr.ndim == 1:
        return len(obj_arr) == nrow
    elif obj_arr.ndim == 2:
        return obj_arr.shape == (nrow, ncol)
    else:
        return False


def isexpandable2row(obj: Any, n: int) -> bool:
    """
    Check if object can be expanded to a row vector of specified length.
    
    Parameters
    ----------
    obj : Any
        Object to check.
    n : int
        Length of target row vector.
    
    Returns
    -------
    bool
        True if obj can be expanded to length-n row vector.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base.utils import isexpandable2row
    >>> isexpandable2row(5, 3)
    True
    >>> isexpandable2row([1, 2, 3], 3)
    True
    >>> isexpandable2row([1, 2], 3)
    False
    """
    obj_arr = np.asarray(obj)
    
    # NOTE: after np.asarray(), a Python scalar becomes a 0-d array, for which
    # np.isscalar() returns False. Use ndim == 0 to detect scalars robustly.
    if obj_arr.ndim == 0:
        return True
    # Flatten 2-D row/column vectors (e.g. shape (1,n) or (n,1)) to 1-D, as
    # MATLAB does when expanding to a row vector.
    if obj_arr.ndim == 2 and 1 in obj_arr.shape:
        obj_arr = obj_arr.reshape(-1)
    if obj_arr.ndim == 1:
        return obj_arr.size == 1 or obj_arr.size == n
    return False


def isphysQcell(cell_array: Any) -> bool:
    """
    Check if object is a physical quantity cell array.
    
    A physical quantity cell array is a 2D array where each row contains
    a PhysQ instance and an integer power.
    
    Parameters
    ----------
    cell_array : Any
        Object to check.
    
    Returns
    -------
    bool
        True if cell_array is a valid physical quantity cell array.
    
    Examples
    --------
    >>> from maxwellfdfd.base import PhysQ
    >>> from maxwellfdfd.base.utils import isphysQcell
    >>> isphysQcell([[PhysQ.L, 1], [PhysQ.OMEGA, -1]])
    True
    >>> isphysQcell([[1, 2], [3, 4]])
    False
    """
    from .physq import PhysQ
    
    if not isinstance(cell_array, (list, np.ndarray)):
        return False
    
    cell_arr = np.asarray(cell_array)
    
    if cell_arr.ndim != 2 or cell_arr.shape[1] != 2:
        return False
    
    n = cell_arr.shape[0]
    for i in range(n):
        if not isinstance(cell_arr[i, 0], PhysQ):
            return False
        if not isinstance(cell_arr[i, 1], (int, np.integer)):
            return False
    
    return True


def istypeof(obj: Any, typename: str) -> bool:
    """
    Check the type of an object.
    
    Parameters
    ----------
    obj : Any
        Object to check.
    typename : str
        Type name to check against. Can be:
        - 'int': integer type
        - 'real': real number type
        - 'complex': complex number type
        - 'arbitrary': always returns True
        - Class name: checks if obj is instance of that class
        - Type name + 'cell': checks if obj is a list/array of that type
    
    Returns
    -------
    bool
        True if obj matches the specified type.
    
    Examples
    --------
    >>> from maxwellfdfd.base.utils import istypeof
    >>> istypeof(5, 'int')
    True
    >>> istypeof(5.0, 'real')
    True
    >>> istypeof(3+4j, 'complex')
    True
    >>> istypeof([1, 2, 3], 'int')
    True
    """
    iscelltype = typename.endswith('cell')
    if iscelltype:
        typename = typename[:-4]
    
    # Handle empty objects
    if obj is None or (isinstance(obj, (list, np.ndarray)) and len(obj) == 0):
        if iscelltype:
            return isinstance(obj, (list, np.ndarray))
        else:
            return True
    
    if iscelltype:
        if not isinstance(obj, (list, np.ndarray)):
            return False
        obj_flat = np.asarray(obj).flatten()
        for item in obj_flat:
            if not istypeof(item, typename):
                return False
        return True
    elif typename == 'int':
        return bool(isinstance(obj, (int, np.integer)) and np.isreal(obj))
    elif typename == 'real':
        return bool(isinstance(obj, (int, float, np.number)) and np.isreal(obj))
    elif typename == 'complex':
        return isinstance(obj, (int, float, complex, np.number))
    elif typename == 'arbitrary':
        return True
    else:
        # Check for class type by matching the class name (or any base class
        # name in the MRO). The previous implementation used eval(typename),
        # which fails for any class not imported into this module's namespace
        # (e.g. 'Shape', 'Material', 'Source'), silently returning False.
        for klass in type(obj).__mro__:
            if klass.__name__ == typename:
                return True
        return False


def istypesizeof(obj: Any, typename: str, 
                 dims: Tuple[int, ...] | None = None,
                 dimselem: Tuple[int, ...] | None = None) -> bool:
    """
    Check the type and size of an object.
    
    Parameters
    ----------
    obj : Any
        Object to check.
    typename : str
        Type name to check against.
    dims : tuple of int, optional
        Expected dimensions. A value of 0 means "any size" for that dimension.
        If None, checks for scalar (1, 1).
    dimselem : tuple of int, optional
        Expected dimensions of each element (only for cell types).
    
    Returns
    -------
    bool
        True if obj matches the specified type and size.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.base.utils import istypesizeof
    >>> istypesizeof(5, 'int')
    True
    >>> istypesizeof(np.array([[1, 2], [3, 4]]), 'int', (2, 2))
    True
    >>> istypesizeof(np.array([1, 2, 3]), 'int', (1, 0))
    True
    """
    iscelltype = typename.endswith('cell')
    if iscelltype:
        typename = typename[:-4]
    
    if dims is None:
        dims = (1, 1)
    
    chkarg(isinstance(dims, tuple) and all(isinstance(d, int) for d in dims),
           '"dims" should be tuple with integer elements.')
    
    # Get object shape
    if isinstance(obj, np.ndarray):
        size_obj = obj.shape
    elif isinstance(obj, list):
        # Try to convert to array for shape checking
        try:
            obj_arr = np.asarray(obj)
            size_obj = obj_arr.shape
        except:
            size_obj = (len(obj),)
    else:
        size_obj = ()  # scalar
    
    # Pad size_obj to match dims length
    if len(size_obj) < len(dims):
        size_obj = tuple(list(size_obj) + [1] * (len(dims) - len(size_obj)))
    
    # Handle empty objects with some zero dimensions
    if (len(obj) == 0 if isinstance(obj, (list, np.ndarray)) else False) and not all(dims):
        return True
    
    if len(size_obj) != len(dims):
        return False
    
    # Check dimensions (0 means any size)
    for i, (s, d) in enumerate(zip(size_obj, dims)):
        if d != 0 and s != d:
            return False
    
    if iscelltype:
        if not isinstance(obj, (list, np.ndarray)):
            return False
        obj_flat = np.asarray(obj).flatten()
        if dimselem is None:
            for item in obj_flat:
                if not istypeof(item, typename):
                    return False
        else:
            for item in obj_flat:
                if not istypesizeof(item, typename, dimselem):
                    return False
        return True
    else:
        return istypeof(obj, typename)