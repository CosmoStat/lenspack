# -*- coding: utf-8 -*-

import numpy as np
import astropy.cosmology
from scipy.integrate import quad

from lenspack.utils import convert_units as conv
from lenspack.utils import sigma_critical


class nfw_profile(object):
    def __init__(self, z, c200, m200=None, r200=None, cosmology='default'):
        """Navarro, Frenk, & White (1997) radial halo density profile.

        rho(r; rho0, rs) = rho0 / [(r / rs) * (1 + r / rs)^2],

        where rho0 can be written as the product delta_c * rho_crit. The more
        common (and useful) parameterization is (c200, m200) instead of
        (rho0, rs).

        Parameters
        ----------
        z : float
            Halo redshift. [dimensionless]
        c200 : float
            Halo concentration. [dimensionless]
        m200 : float, optional
            Spherical mass contained within a radius `r200`. [solar mass]
        r200 : float, optional
            Characteristic halo radius. [Mpc]
        cosmology : {'default', instance of `astropy.cosmology`}
            Cosmological model in which to calculate distances. The default
            is a `FlatLambdaCDM` object with parameters H0=70, Om0=0.3,
            Ob0=0.044, and Tcmb0=2.725. Alternatively, a custom cosmology can
            be provided as an instance of `astropy.cosmology`.

        Notes
        -----
        (1) Either `m200` or `r200` must be given to fully define the profile,
            and `m200` takes precedence over `r200` if both are given.
        (2) The reference background density implicitly used in the definition
            of `c200`, `m200`, and `r200` is the critical density at the
            redshift of the halo.
        """
        self.z = z
        self.c200 = c200

        # Background cosmological model
        if cosmology == 'default':
            self.cosmo = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3,
                                                         Ob0=0.044,
                                                         Tcmb0=2.725)
        else:
            if not isinstance(cosmology, astropy.cosmology.core.Cosmology):
                raise TypeError("Invalid cosmology.")
            self.cosmo = cosmology

        # Reference background density
        rho_crit = self.cosmo.critical_density(z)

        # Either m200 or r200 must be provided
        assert (m200 is not None) or (r200 is not None)

        if m200 is not None:
            # Compute r200
            m200 = conv(m200, "solMass")
            r200 = np.power(3. * m200 / (800 * np.pi * rho_crit), 1. / 3)
            r200 = conv(r200, "Mpc")
        else:
            # Compute m200
            r200 = conv(r200, "Mpc")
            m200 = 800. * np.pi * rho_crit * np.power(r200, 3) / 3
            m200 = conv(m200, "solMass")
        self.m200 = m200
        self.r200 = r200

        # Standard NFW parameters
        fc = np.log(1. + self.c200) - self.c200 / (1. + self.c200)
        self.rs = self.r200 / self.c200
        self.delta_c = (200. / 3) * np.power(self.c200, 3) / fc
        self.rho0 = self.delta_c * rho_crit

    def rho(self, r):
        """Density at a given distance from the halo center."""
        r = conv(r, "Mpc")
        return self.rho0 / (r / self.rs) / np.power(1 + r / self.rs, 2)

    def mass_enclosed(self, r):
        """Mass (3D) inside a sphere of a given radius."""
        r = conv(r, "Mpc")
        x = r / self.rs
        prefactor = 4 * np.pi * np.power(self.rs, 3) * self.rho0
        return conv(prefactor * (np.log(1 + x) - x / (1 + x)), "g")

    def sigma_crit(self, zs):
        """Critical surface mass density for a given source redshift.

        Parameters
        ----------
        zs : float or array
            Redshift(s) of the source galaxies.
        """
        return sigma_critical(self.z, zs, self.cosmo)

    def sigma(self, r, r_off=None):
        """Surface mass density (Sigma) at a given radial distance.

        Parameters
        ----------
        r : float or array
            Projected radial distance from the halo center.
        r_off : float, optional
            Projected radial offset of the halo center. If provided, the
            computed Sigma is the mean over all azimuthal angles, i.e.,
            averaged over the circle of radius r_off centered on the halo.

        Notes
        -----
        The calculation is much slower when r_off is given due to the
        numerical integration necessary.

        Value(s) returned in units of solar mass per square parsec.
        """
        prefactor = conv(2 * self.rs * self.rho0, "solMass / pc2")
        r = np.atleast_1d(r)

        # Basic centered calculation (excluding the constant prefactor)
        def _sigma(r):
            # Recast radius in units of rs
            x = (conv(r, self.rs.unit.to_string()) / self.rs).value
            return self._func1(x)

        # Off-centered Sigma as a function of angle and offset radius
        def _sigma_of_theta(theta, r, r_off):
            term1 = np.power(r, 2) + np.power(r_off, 2)
            term2 = 2 * r * r_off * np.cos(theta)
            return _sigma(np.sqrt(term1 - term2))

        if r_off is None or r_off == 0:
            # Simple centered Sigma
            result = _sigma(r)
        else:
            # Average over all azimuthal angles
            r_off = conv(r_off, self.rs.unit.to_string())
            a, b = (0, 2 * np.pi)  # Integration limits
            integrals = [quad(_sigma_of_theta, a, b / 2,
                         args=(rr, r_off))[0] for rr in r]
            result = 2 * np.array(integrals) / (b - a)

        # Clean up as necessary
        if len(result) == 1:
            result = result[0]

        return prefactor * result

    def delta_sigma(self, r, r_off=None):
        """Difference between the mean Sigma within a disk and its boundary.

        Parameters
        ----------
        r : float or array
            Projected radial distance from the halo center.
        r_off : float, optional
            Projected radial offset of the halo center. If provided, the
            computed Delta Sigma is the mean over all azimuthal angles, i.e.,
            averaged over the circle of radius r_off centered on the halo.

        Notes
        -----
        The calculation is much slower when r_off is given due to the
        numerical integration necessary.

        Value(s) returned in units of solar mass per square parsec.
        """
        prefactor = conv(2 * self.rs * self.rho0, "solMass / pc2")
        r = np.atleast_1d(r)

        # Basic centered calculation (excluding the constant prefactor)
        def _delta_sigma(r):
            # Recast radius in units of rs
            x = (conv(r, self.rs.unit.to_string()) / self.rs).value
            return self._func2(x)

        # Off-centered Delta Sigma as a function of angle and offset radius
        def _delta_sigma_of_theta(theta, r, r_off):
            term1 = np.power(r, 2) + np.power(r_off, 2)
            term2 = 2 * r * r_off * np.cos(theta)
            return _delta_sigma(np.sqrt(term1 - term2))

        if r_off is None or r_off == 0:
            # Simple centered Delta Sigma
            result = _delta_sigma(r)
        else:
            # Average over all azimuthal angles
            r_off = conv(r_off, self.rs.unit.to_string())
            a, b = (0, 2 * np.pi)  # Integration limits
            integrals = [quad(_delta_sigma_of_theta, a, b / 2,
                         args=(rr, r_off))[0] for rr in r]
            result = 2 * np.array(integrals) / (b - a)

        # Clean up as necessary
        if len(result) == 1:
            result = result[0]

        return prefactor * result

    def mean_sigma_enclosed(self, r, r_off):
        """Mean surface mass density (Sigma) enclosed within a disk.

        Parameters
        ----------
        r : float or array
            Projected radial distance from the halo center.
        r_off : float, optional
            Projected radial offset of the halo center.

        Notes
        -----
        The calculation is much slower when r_off is given due to the
        numerical integration necessary.

        Value(s) returned in units of solar mass per square parsec.
        """
        return self.delta_sigma(r, r_off) + self.sigma(r, r_off)

    def gamma_t(self, r, zs, r_off=None):
        """Tangential shear magnitude experienced by a source object.

        Parameters
        ----------
        r : float or array
            Projected radial distance of the source from the halo center.
        zs : float or array
            Redshift of the source.
        r_off : float, optional
            Projected radial offset of the halo center.
        """
        r = np.atleast_1d(r)
        zs = np.atleast_1d(zs)
        assert len(r) == len(zs), "Inputs must have the same length."

        ratio = self.delta_sigma(r, r_off) / self.sigma_crit(zs)
        return ratio.decompose().value

    def kappa(self, r, zs, r_off=None):
        """Convergence experienced by a source object.

        Parameters
        ----------
        r : float or array
            Projected radial distance of the source from the halo center.
        zs : float or array
            Redshift of the source.
        r_off : float, optional
            Projected radial offset of the halo center.
        """
        r = np.atleast_1d(r)
        zs = np.atleast_1d(zs)
        assert len(r) == len(zs), "Inputs must have the same length."

        ratio = self.sigma(r, r_off) / self.sigma_crit(zs)
        return ratio.decompose().value

    @staticmethod
    def _func1(x):
        """Dirty work function for calculating sigma (surface mass density).

        See Eq. (11) in Wright & Brainerd, ApJ 534, 34 (2000).

        Parameters
        ----------
        x : float or array
            Dimensionless projected radial distance from the halo center.

        Notes
        -----
        The function diverges as x -> 0, and so we set it to np.inf by hand.
        """
        x = np.atleast_1d(x).astype(float)
        assert (x >= 0).all(), "x must be positive."
        result = np.zeros_like(x)

        # Case 0 : x == 0
        result[x == 0] = np.inf

        # Case 1 : 0 < x < 1
        y = x[(x > 0) & (x < 1)]
        term1 = np.arccosh(1. / y) / np.power(1 - np.power(y, 2), 3. / 2)
        term2 = 1. / (1 - np.power(y, 2))
        result[(x > 0) & (x < 1)] = term1 - term2

        # Case 2 : x == 1
        result[x == 1] = 1. / 3

        # Case 3 : x > 1
        y = x[x > 1]
        term1 = 1. / (np.power(y, 2) - 1)
        term2 = np.arccos(1. / y) / np.power(np.power(y, 2) - 1, 3. / 2)
        result[x > 1] = term1 - term2

        # Clean up
        # if len(result) == 1: result = result[0]

        return result

    @staticmethod
    def _func2(x):
        """Dirty work function for calculating delta_sigma.

        See Eqs. (12)-(14) in Wright & Brainerd, ApJ 534, 34 (2000).

        Parameters
        ----------
        x : float or array
            Dimensionless projected radial distance from the halo center.

        Notes
        -----
        The function approaches 0.5 as x -> 0, which we set by hand due to
        numerical instability.
        """
        x = np.atleast_1d(x).astype(float)
        assert (x >= 0).all(), "x must be positive."
        result = np.zeros_like(x)

        # Case 0 : x == 0
        result[x == 0] = 0.5

        # Case 1 : 0 < x < 1
        y = x[(x > 0) & (x < 1)]
        term1 = ((2. / np.power(y, 2) + 1. / (np.power(y, 2) - 1)) /
                 np.sqrt(1 - np.power(y, 2)))
        ratio = (1 - y) / (1 + y)
        term2 = np.log((1 + np.sqrt(ratio)) / (1 - np.sqrt(ratio)))
        term3 = 2 * np.log(y / 2) / np.power(y, 2)
        term4 = 1. / (np.power(y, 2) - 1)
        result[(x > 0) & (x < 1)] = term1 * term2 + term3 - term4

        # Case 2 : x == 1
        result[x == 1] = 5. / 3 + 2 * np.log(1. / 2)

        # Case 3 : x > 1
        y = x[x > 1]
        term1 = 4. / np.power(y, 2) / np.sqrt(np.power(y, 2) - 1)
        term2 = 2. / np.power(np.power(y, 2) - 1, 3. / 2)
        term3 = np.arctan2(np.sqrt(y - 1), np.sqrt(y + 1))
        term4 = 2 * np.log(y / 2) / np.power(y, 2)
        term5 = 1. / (np.power(y, 2) - 1)
        result[x > 1] = (term1 + term2) * term3 + term4 - term5

        # Clean up
        # if len(result) == 1: result = result[0]

        return result
