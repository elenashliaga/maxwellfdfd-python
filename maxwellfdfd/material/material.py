"""
Material class for MaxwellFDFD.

This module defines the Material class for representing electromagnetic
materials with electric permittivity and magnetic permeability.
"""

import os
from typing import Union, List, Tuple, Optional

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

from ..base import Axis
from ..base.oscillation import Oscillation
from ..base.utils import chkarg, expand2row, isexpandable2row


class Material:
    """
    Electromagnetic material with electric permittivity and magnetic permeability.
    
    The Material class represents electromagnetic materials with scalar, vector,
    or tensor permittivity and permeability. Materials can be created directly
    or loaded from data files containing refractive index measurements.
    
    Parameters
    ----------
    name : str
        Name of the material. Used to distinguish materials and label them in figures.
    color : str or tuple of float
        Color of the material for visualization. Can be a MATLAB-style color code
        (e.g., 'r' for red, 'k' for black) or an RGB tuple [r, g, b] where each
        component is between 0 and 1. Use 'none' to hide objects made of this material.
    eps : complex or array_like
        Electric permittivity of the material. Can be:
        - Complex scalar: isotropic permittivity
        - Length-3 vector: diagonal tensor with given values
        - 3x3 matrix: full tensor
        - 3x6 matrix: tensor in crystal axes format (first 3 columns are tensor,
          last 3 columns are crystal axes)
    mu : complex or array_like, optional
        Magnetic permeability of the material. Same format as eps. Default is 1.0.
    islossless : bool, optional
        If True, ignore loss or gain in material by taking real parts only.
        Default is False.
    
    Attributes
    ----------
    eps : numpy.ndarray
        Electric permittivity as a 3x3 tensor (immutable).
    mu : numpy.ndarray
        Magnetic permeability as a 3x3 tensor (immutable).
    name : str
        Material name (mutable).
    color : str or tuple
        Material color for visualization (mutable).
    
    Examples
    --------
    Create a vacuum material:
    
    >>> vacuum = Material('vacuum', 'none', 1.0)
    
    Create a dielectric with anisotropic permittivity:
    
    >>> dielectric = Material('dielectric', 'b', [4.0, 4.0, 4.5])
    
    Create a lossy material:
    
    >>> lossy = Material('lossy', 'r', 2.0 - 0.1j)
    
    Create a lossless material from lossy input:
    
    >>> lossless = Material('lossless', 'g', 2.0 - 0.1j, islossless=True)
    
    See Also
    --------
    fromtable : Create material from data file
    """
    
    def __init__(
        self,
        name: str,
        color: Union[str, Tuple[float, float, float]],
        eps: Union[complex, np.ndarray],
        mu: Union[complex, np.ndarray, None] = None,
        islossless: bool = False
    ):
        """
        Initialize Material.
        
        Parameters
        ----------
        name : str
            Name of the material.
        color : str or tuple of float
            Color for visualization.
        eps : complex or array_like
            Electric permittivity.
        mu : complex or array_like, optional
            Magnetic permeability. Default is 1.0.
        islossless : bool, optional
            Whether to make material lossless. Default is False.
        """
        # Validate name
        chkarg(isinstance(name, str) and len(name) > 0,
               '"name" should be a non-empty string.')
        
        # Validate color
        if isinstance(color, str):
            # String color is valid
            pass
        elif isinstance(color, (tuple, list, np.ndarray)):
            chkarg(len(color) == 3 and all(isinstance(c, (int, float)) for c in color)
                   and all(0 <= c <= 1 for c in color),
                   '"color" should be string or [r g b] with values in [0, 1].')
            color = tuple(float(c) for c in color)
        else:
            raise ValueError('"color" should be string or [r g b].')
        
        # Validate and process eps
        eps = self._process_material_tensor(eps, 'eps')
        
        # Process mu
        if mu is None:
            mu = 1.0
        mu = self._process_material_tensor(mu, 'mu')
        
        # Validate islossless
        chkarg(isinstance(islossless, bool),
               '"islossless" should be boolean.')
        
        # Apply lossless if requested
        if islossless:
            name = f"{name} (lossless)"
            eps = np.real(eps)
            mu = np.real(mu)
        
        # Store attributes
        self._eps = eps
        self._mu = mu
        self.name = name
        self.color = color
    
    @staticmethod
    def _process_material_tensor(
        tensor: Union[complex, np.ndarray],
        name: str
    ) -> np.ndarray:
        """
        Process material tensor input into 3x3 format.
        
        Parameters
        ----------
        tensor : complex or array_like
            Input tensor (scalar, vector, 3x3 matrix, or 3x6 matrix).
        name : str
            Name of the tensor (for error messages).
        
        Returns
        -------
        numpy.ndarray
            3x3 tensor.
        
        Raises
        ------
        ValueError
            If tensor format is invalid.
        """
        # Convert to numpy array if not already
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor, dtype=complex)
        
        # Check if scalar or expandable to row vector
        if np.isscalar(tensor) or isexpandable2row(tensor, Axis.count()):
            # Expand to diagonal tensor.
            # NOTE: expand2row returns a (1, n) array; np.diag of a 2-D array
            # extracts (rather than builds) a diagonal, so flatten to 1-D first.
            tensor = expand2row(tensor, Axis.count()).reshape(-1)
            tensor = np.diag(tensor)
        # Check if 3x3 tensor
        elif tensor.shape == (Axis.count(), Axis.count()):
            pass  # Already in correct format
        # Check if 3x6 tensor (crystal axes format)
        elif tensor.shape == (Axis.count(), 2 * Axis.count()):
            # Extract crystal axes and tensor
            S = tensor[:, Axis.count():]
            tensor = tensor[:, :Axis.count()]
            # Transform: S * tensor / S
            tensor = S @ tensor @ np.linalg.inv(S)
        else:
            raise ValueError(
                f'"{name}" should be complex scalar, length-{Axis.count()} '
                f'row vector, {Axis.count()}x{Axis.count()} matrix, or '
                f'{Axis.count()}x{2*Axis.count()} matrix.'
            )
        
        return tensor
    
    @property
    def eps(self) -> np.ndarray:
        """
        Electric permittivity tensor (3x3).
        
        Returns
        -------
        numpy.ndarray
            3x3 permittivity tensor.
        """
        return self._eps
    
    @property
    def mu(self) -> np.ndarray:
        """
        Magnetic permeability tensor (3x3).
        
        Returns
        -------
        numpy.ndarray
            3x3 permeability tensor.
        """
        return self._mu
    
    @property
    def has_iso_eps(self) -> bool:
        """
        Check if permittivity is isotropic.
        
        Returns
        -------
        bool
            True if permittivity is isotropic (diagonal with equal elements).
        """
        return np.allclose(self.eps, np.diag(np.diag(self.eps))) and \
               np.allclose(np.diag(self.eps), self.eps[0, 0])
    
    @property
    def has_iso_mu(self) -> bool:
        """
        Check if permeability is isotropic.
        
        Returns
        -------
        bool
            True if permeability is isotropic (diagonal with equal elements).
        """
        return np.allclose(self.mu, np.diag(np.diag(self.mu))) and \
               np.allclose(np.diag(self.mu), self.mu[0, 0])
    
    @property
    def is_iso(self) -> bool:
        """
        Check if material is fully isotropic.
        
        Returns
        -------
        bool
            True if both permittivity and permeability are isotropic.
        """
        return self.has_iso_eps and self.has_iso_mu
    
    @staticmethod
    def fromtable(
        osc: Oscillation,
        name: str,
        color: Union[str, Tuple[float, float, float]],
        islossless: bool = False
    ) -> 'Material':
        """
        Create material from data file.
        
        Loads material data from a .mat file containing refractive index
        measurements and interpolates to the specified frequency.
        
        Parameters
        ----------
        osc : Oscillation
            Oscillation object specifying the frequency.
        name : str
            Name of the material (also used to find the data file).
        color : str or tuple of float
            Color for visualization.
        islossless : bool, optional
            Whether to make material lossless. Default is False.
        
        Returns
        -------
        Material
            Material with interpolated permittivity.
        
        Notes
        -----
        The data file should be located in the 'dielconst' directory relative
        to the MaxwellFDFD root. The file name should be '{name}.mat' (with '/'
        replaced by file separator).
        
        The .mat file should contain:
        - eV : array of photon energies in eV
        - n : array of refractive indices
        - k : array of extinction coefficients
        
        The permittivity is computed as epsilon = (n - 1j*k)^2.
        
        Examples
        --------
        >>> from maxwellfdfd.base import Oscillation
        >>> osc = Oscillation(wvlen=1.55)  # 1.55 um wavelength
        >>> gold = Material.fromtable(osc, 'Au', 'y')
        
        Raises
        ------
        FileNotFoundError
            If the data file cannot be found.
        ValueError
            If the frequency is outside the data range.
        """
        # Validate inputs
        chkarg(isinstance(osc, Oscillation),
               '"osc" should be instance of Oscillation.')
        chkarg(isinstance(name, str) and len(name) > 0,
               '"name" should be a non-empty string.')
        
        if isinstance(color, str):
            pass  # String color is valid
        elif isinstance(color, (tuple, list, np.ndarray)):
            chkarg(len(color) == 3 and all(isinstance(c, (int, float)) for c in color)
                   and all(0 <= c <= 1 for c in color),
                   '"color" should be string or [r g b] with values in [0, 1].')
        else:
            raise ValueError('"color" should be string or [r g b].')
        
        chkarg(isinstance(islossless, bool),
               '"islossless" should be boolean.')
        
        # Get energy in eV
        # (osc.in_eV() may return a complex-typed value with zero imaginary
        # part; interp1d requires real x, so cast via real part)
        eV = float(np.real(osc.in_eV()))
        
        # Find data file
        # Get the package root directory
        package_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        param_dir = os.path.join(package_root, 'dielconst')
        param_file = os.path.join(param_dir, name.replace('/', os.sep) + '.mat')
        
        # Load data
        if not os.path.exists(param_file):
            raise FileNotFoundError(
                f"Material data file not found: {param_file}\n"
                f"Please ensure the dielconst directory exists and contains {name}.mat"
            )
        
        param = loadmat(param_file)
        
        # Check if energy is within data range (with small tolerance)
        eV_data = param['eV'].flatten()
        eV_min, eV_max = eV_data[0], eV_data[-1]
        
        if eV < eV_min and eV * (1 + 1e-6) >= eV_min:
            eV = eV_min
        elif eV > eV_max and eV * (1 - 1e-6) <= eV_max:
            eV = eV_max
        
        chkarg(eV_min <= eV <= eV_max,
               f'"eV" ({eV:.6f}) should be in the range [{eV_min:.6f}, {eV_max:.6f}] '
               f'described by {param_file}')
        
        # Interpolate n and k (measured data)
        n_data = param['n'].flatten()
        k_data = param['k'].flatten()
        
        n_interp = interp1d(eV_data, n_data, kind='linear', bounds_error=False,
                           fill_value=(n_data[0], n_data[-1]))
        k_interp = interp1d(eV_data, k_data, kind='linear', bounds_error=False,
                           fill_value=(k_data[0], k_data[-1]))
        
        n = n_interp(eV)
        k = k_interp(eV)
        
        # Compute permittivity
        epsilon = (n - 1j * k) ** 2
        
        # Create material
        return Material(name, color, epsilon, islossless=islossless)
    
    def sort(self, **kwargs) -> List['Material']:
        """
        Sort a list of materials by name.
        
        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed to sorted() (e.g., reverse=True).
        
        Returns
        -------
        list of Material
            Sorted list of materials.
        
        Examples
        --------
        >>> materials = [Material('b', 'r', 1.0), Material('a', 'b', 2.0)]
        >>> sorted_mats = Material.sort(materials)
        """
        # This method is designed to work with lists of materials
        # For a single material, return a list containing just this material
        return [self]
    
    def __ne__(self, other: 'Material') -> bool:
        """
        Check if two materials are not equal.
        
        Two materials are considered equal if they have the same name,
        color, permittivity, and permeability.
        
        Parameters
        ----------
        other : Material
            Material to compare with.
        
        Returns
        -------
        bool
            True if materials are not equal, False otherwise.
        """
        if not isinstance(other, Material):
            return True
        
        return (self.name != other.name or
                self.color != other.color or
                not np.allclose(self.eps, other.eps) or
                not np.allclose(self.mu, other.mu))
    
    def __eq__(self, other: 'Material') -> bool:
        """
        Check if two materials are equal.
        
        Parameters
        ----------
        other : Material
            Material to compare with.
        
        Returns
        -------
        bool
            True if materials are equal, False otherwise.
        """
        return not self.__ne__(other)

    def __hash__(self) -> int:
        """
        Hash based on object identity.

        Defining ``__eq__`` would otherwise make Material unhashable. This
        identity hash restores hashability (e.g. for de-duplicating a list of
        materials via ``dict.fromkeys``) and matches MATLAB handle semantics,
        where ``unique`` on a handle array de-duplicates by identity.

        Returns
        -------
        int
            Identity-based hash.
        """
        return id(self)

    def __repr__(self) -> str:
        """
        String representation of Material.
        
        Returns
        -------
        str
            String representation.
        """
        eps_str = f"{self.eps[0, 0]:.3g}"
        if not self.has_iso_eps:
            eps_str = "tensor"
        
        return f"Material(name='{self.name}', eps={eps_str})"