"""
Physical constants for MaxwellFDFD.

This module defines fundamental physical constants used in MaxwellFDFD simulations.
"""

import numpy as np


class PhysC:
    """
    Class defining fundamental physical constants.
    
    All constants are in SI units.
    
    Attributes
    ----------
    c0 : float
        Speed of light in vacuum (m/s).
    mu0 : float
        Permeability of free space (H/m).
    eps0 : float
        Permittivity of free space (F/m).
    eta0 : float
        Impedance of free space (Ohm).
    h : float
        Planck constant (eV·sec).
    hbar : float
        Reduced Planck constant (eV·sec).
    
    Examples
    --------
    >>> from maxwellfdfd.base import PhysC
    >>> PhysC.c0
    299792458.0
    >>> PhysC.eps0
    8.854187817620389e-12
    >>> PhysC.hbar
    6.582119569509306e-16
    """
    
    # Fundamental constants
    c0 = 2.99792458e8  # Speed of light in vacuum (m/s)
    mu0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    eps0 = 1 / (c0**2 * mu0)  # Permittivity of free space (F/m)
    eta0 = np.sqrt(mu0 / eps0)  # Impedance of free space (Ohm)
    h = 4.135667662e-15  # Planck constant (eV·sec)
    hbar = h / (2 * np.pi)  # Reduced Planck constant (eV·sec)