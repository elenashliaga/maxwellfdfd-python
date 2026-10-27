"""
Domain class for computation domains.

This module defines the Domain class, which is a special Box subclass
for specifying a computation domain.
"""

import numpy as np
from maxwellfdfd.base import Axis, Sign, chkarg
from .box import Box


class Domain(Box):
    """
    Special Box subclass for specifying a computation domain.
    
    Unlike other Shape classes, Domain must take dl_max as an argument.
    
    Parameters
    ----------
    bound : ndarray
        Bounds of the domain as shape (3, 2) array:
        [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
    dl_max : float or array_like
        Maximum grid cell size allowed. Can be a scalar or length-3 vector.
    
    Examples
    --------
    >>> import numpy as np
    >>> from maxwellfdfd.shape import Domain
    >>> domain = Domain(np.array([[-5000, 5000], [-3000, 3000], [0, 500]]), np.array([10, 10, 10]))
    >>> domain.bound
    array([[-5000.,  5000.],
           [-3000.,  3000.],
           [    0.,   500.]])
    >>> domain.dl_max
    array([10., 10., 10.])
    """
    
    def __init__(self, bound: np.ndarray, dl_max):
        """
        Initialize a Domain.
        
        Parameters
        ----------
        bound : ndarray
            Domain bounds.
        dl_max : float or array_like
            Maximum grid cell size.
        """
        # Validate bound
        chkarg(
            bound.shape == (Axis.count(), Sign.count()) and np.isrealobj(bound),
            f'"bound" should be {Axis.count()}-by-{Sign.count()} array with real elements.'
        )
        
        # Call parent constructor with dl_max (required for Domain)
        super().__init__(bound, dl_max)