"""
Oscillation class for MaxwellFDFD.

This module defines the oscillation parameters used in MaxwellFDFD simulations.
"""

import numpy as np
from .physunit import PhysUnit
from .physc import PhysC
from .utils import chkarg, istypesizeof


class Oscillation:
    """
    Class containing information on the operation frequency.
    
    The Oscillation class represents the oscillation parameters of the
    electromagnetic fields, including wavelength, angular frequency, and energy.
    
    Parameters
    ----------
    wvlen : complex
        Vacuum wavelength (normalized to L0 units).
    unit : PhysUnit
        Physical unit instance for unit conversions.
    
    Attributes
    ----------
    wvlen : complex
        Vacuum wavelength (normalized).
    unit : PhysUnit
        Physical unit instance.
    
    Examples
    --------
    >>> from maxwellfdfd.base import Oscillation, PhysUnit
    >>> unit = PhysUnit(1e-6)  # 1 micrometer reference
    >>> osc = Oscillation(1.0, unit)  # Wavelength of 1 L0
    >>> osc.in_L0()
    1.0
    >>> osc.in_omega0()
    6.283185307179586
    >>> osc.in_eV()
    1.2398419843320026
    """
    
    def __init__(self, wvlen: complex, unit: PhysUnit):
        """
        Initialize Oscillation with wavelength and unit.
        
        Parameters
        ----------
        wvlen : complex
            Vacuum wavelength (normalized to L0 units).
        unit : PhysUnit
            Physical unit instance for unit conversions.
        
        Raises
        ------
        ValueError
            If wvlen is not complex or unit is not a PhysUnit instance.
        """
        chkarg(istypesizeof(wvlen, 'complex'), '"wvlen" should be complex.')
        chkarg(isinstance(unit, PhysUnit), '"unit" should be instance of PhysUnit.')
        
        self._wvlen = complex(wvlen)
        self._unit = unit
    
    @property
    def wvlen(self) -> complex:
        """
        Get the vacuum wavelength.
        
        Returns
        -------
        complex
            Vacuum wavelength (normalized to L0 units).
        """
        return self._wvlen
    
    @property
    def unit(self) -> PhysUnit:
        """
        Get the physical unit.
        
        Returns
        -------
        PhysUnit
            Physical unit instance.
        """
        return self._unit
    
    def in_L0(self) -> complex:
        """
        Return wavelength in L0 units.
        
        Returns
        -------
        complex
            Wavelength normalized to L0.
        """
        return self._wvlen
    
    def in_omega0(self) -> complex:
        """
        Return angular frequency normalized to omega0.
        
        The angular frequency is calculated as omega = 2*pi / wvlen.
        
        Returns
        -------
        complex
            Angular frequency (normalized).
        """
        return 2 * np.pi / self._wvlen
    
    def in_eV(self) -> complex:
        """
        Return photon energy in electron-volts.
        
        The energy is calculated as E = h*c / lambda, where lambda is
        the wavelength in meters.
        
        Returns
        -------
        complex
            Photon energy in eV.
        """
        from .physq import PhysQ
        lambda_m = self._wvlen * self._unit.value(PhysQ.L)
        return PhysC.h * PhysC.c0 / lambda_m