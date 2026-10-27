"""
Equation type class for MaxwellFDFD.

This module defines the equation type used in MaxwellFDFD simulations.
"""

from .ft import FT
from .gt import GT
from .utils import chkarg, istypesizeof


class EquationType:
    """
    Class defining the type of the equation.
    
    The EquationType class specifies which field type (E or H) the equation
    solves for, and the grid type of the E-field grid.
    
    Parameters
    ----------
    f : FT
        Field type (E or H). Can be a single FT or a row vector of FT instances.
    ge : GT
        Grid type of the E-field grid (primary or dual).
    
    Attributes
    ----------
    f : FT or list of FT
        Field type(s) that this equation solves for.
    ge : GT
        Grid type of the E-field grid.
    
    Examples
    --------
    >>> from maxwellfdfd.base import EquationType, FT, GT
    >>> eqtype = EquationType(FT.E, GT.PRIM)
    >>> eqtype.f
    <FT.E: 'e'>
    >>> eqtype.ge
    <GT.PRIM: 'prim'>
    """
    
    def __init__(self, f: FT, ge: GT):
        """
        Initialize EquationType with field type and grid type.
        
        Parameters
        ----------
        f : FT
            Field type (E or H). Can be a single FT or a row vector of FT instances.
        ge : GT
            Grid type of the E-field grid (primary or dual).
        
        Raises
        ------
        ValueError
            If f is not a valid FT instance or row vector, or if ge is not a GT instance.
        """
        # Check if f is a single FT or a row vector of FT instances
        if isinstance(f, FT):
            self._f = f
        elif isinstance(f, (list, tuple)):
            # Check if it's a row vector of FT instances
            chkarg(all(isinstance(item, FT) for item in f),
                   '"f" should be row vector with FT as elements.')
            self._f = list(f)
        else:
            raise ValueError('"f" should be FT instance or row vector with FT as elements.')
        
        chkarg(isinstance(ge, GT), '"ge" should be instance of GT.')
        self._ge = ge
    
    @property
    def f(self) -> FT | list[FT]:
        """
        Get the field type.
        
        Returns
        -------
        FT or list of FT
            Field type(s) that this equation solves for.
        """
        return self._f
    
    @property
    def ge(self) -> GT:
        """
        Get the grid type of the E-field grid.
        
        Returns
        -------
        GT
            Grid type (primary or dual).
        """
        return self._ge