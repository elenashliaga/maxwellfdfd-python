"""
Physical units for MaxwellFDFD.

This module defines the physical units used in MaxwellFDFD simulations.
"""

import numpy as np
from typing import Union, List, Tuple
from .physq import PhysQ
from .physc import PhysC
from .utils import chkarg, isphysQcell


class PhysUnit:
    """
    Class defining the units of physical quantities used in MaxwellFDFD.
    
    The PhysUnit class provides a way to work with normalized units based on
    a reference length L0. All quantities are expressed in terms of this
    reference length.
    
    Parameters
    ----------
    L0 : float
        Reference length in meters.
    
    Attributes
    ----------
    va : ndarray
        Array of unit values indexed by PhysQ enumeration.
    
    Examples
    --------
    >>> from maxwellfdfd.base import PhysUnit, PhysQ
    >>> unit = PhysUnit(1e-6)  # Reference length of 1 micrometer
    >>> unit.value(PhysQ.L)
    1e-06
    >>> unit.value(PhysQ.OMEGA)
    299792458.0
    >>> unit.value([[PhysQ.L, 1], [PhysQ.OMEGA, -1]])
    299792458.0
    """
    
    def __init__(self, L0: float):
        """
        Initialize PhysUnit with reference length.
        
        Parameters
        ----------
        L0 : float
            Reference length in meters.
        """
        self._va = np.full(PhysQ.count(), np.nan)
        
        # Set unit values for each physical quantity
        self._va[int(PhysQ.ARBITRARY)] = 1.0
        self._va[int(PhysQ.L)] = L0  # length in m (L0-dependent)
        self._va[int(PhysQ.OMEGA)] = PhysC.c0 / self._va[int(PhysQ.L)]  # frequency in rad/s
        self._va[int(PhysQ.EPS)] = PhysC.eps0  # permittivity in eps0
        self._va[int(PhysQ.MU)] = PhysC.mu0  # permeability in mu0
        self._va[int(PhysQ.E)] = 1.0  # E-field in V/m
        self._va[int(PhysQ.H)] = self._va[int(PhysQ.E)] / PhysC.eta0  # H-field in A/m
        self._va[int(PhysQ.J)] = self._va[int(PhysQ.H)] / self._va[int(PhysQ.L)]  # current density in A/m^2
        self._va[int(PhysQ.M)] = self._va[int(PhysQ.E)] / self._va[int(PhysQ.L)]  # magnetic current density in V/m^2
        self._va[int(PhysQ.I)] = self._va[int(PhysQ.J)] * self._va[int(PhysQ.L)]**2  # current in A
        self._va[int(PhysQ.V)] = self._va[int(PhysQ.E)] * self._va[int(PhysQ.L)]  # voltage in V
        self._va[int(PhysQ.S)] = self._va[int(PhysQ.E)] * self._va[int(PhysQ.H)]  # Poynting vector in W/m^2
        self._va[int(PhysQ.P)] = self._va[int(PhysQ.S)] * self._va[int(PhysQ.L)]**2  # power in W
        self._va[int(PhysQ.U)] = self._va[int(PhysQ.S)] / self._va[int(PhysQ.L)]  # power density in W/m^3
    
    @property
    def va(self) -> np.ndarray:
        """
        Get the array of unit values.
        
        Returns
        -------
        ndarray
            Array of unit values indexed by PhysQ enumeration.
        """
        return self._va
    
    def value(self, physQcell: Union[PhysQ, List, Tuple, np.ndarray]) -> float:
        """
        Calculate the unit value for a physical quantity or combination.
        
        Parameters
        ----------
        physQcell : PhysQ or list/tuple/array
            Either a single PhysQ instance, or a 2D array where each row
            contains [PhysQ, power] pairs. For example, [[PhysQ.L, 1], [PhysQ.OMEGA, -1]]
            represents L/omega.
        
        Returns
        -------
        float
            The calculated unit value.
        
        Raises
        ------
        ValueError
            If physQcell is not a valid PhysQ instance or physical quantity cell array.
        
        Examples
        --------
        >>> from maxwellfdfd.base import PhysUnit, PhysQ
        >>> unit = PhysUnit(1e-6)
        >>> unit.value(PhysQ.L)
        1e-06
        >>> unit.value([[PhysQ.L, 1], [PhysQ.OMEGA, -1]])
        3.3356409519815204e-15
        """
        # Check if physQcell is a single PhysQ instance
        if isinstance(physQcell, PhysQ):
            physQcell = [[physQcell, 1]]

        # Normalize a single 1-D row [PhysQ, int] to a 2-D cell (MATLAB
        # {PhysQ, int} is 1x2). This keeps value() robust to 1-D input.
        if isinstance(physQcell, (list, np.ndarray)):
            _arr = np.asarray(physQcell, dtype=object)
            if _arr.ndim == 1 and _arr.shape[0] == 2:
                physQcell = _arr.reshape(1, 2).tolist()
        
        # Check if physQcell is a valid physical quantity cell array
        chkarg(isphysQcell(physQcell),
               '"physQcell" should be instance of PhysQ, or cell array {PhysQ, int; PhysQ, int; ...}.')
        
        physQ_arr = np.asarray(physQcell)
        n = physQ_arr.shape[0]
        
        v0 = 1.0
        for i in range(n):
            physQ = physQ_arr[i, 0]
            physDim = int(physQ_arr[i, 1])
            v0 = v0 * self._va[int(physQ)]**physDim
        
        return v0