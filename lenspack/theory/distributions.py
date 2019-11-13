# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import gamma


class zdist(object):
    """Parameterized redshift probability distribution."""
    def __init__(self, alpha, beta, z0, zmax=None):
        """Continuous model of a redshift probability distribution.

        The parameterized distribution is given by

            n(z) = A * [(z / z0)^alpha] * exp[-(z / z0)^beta],

        where A is the normalization factor.

        Parameters
        ----------
        alpha, beta, z0 : float
            Parameters of the distribution.
        zmax : float, optional
            Maximum redshift of the distribution. If given, the function
            returns zero for redshifts larger than `zmax`, and the
            normalization is computed by numerical integration. If None,
            np.inf is used, and the normalization is given by the gamma
            function. Default is None.

        Notes
        -----
        A good fit to the MICE simulated galaxy catalog are the parameters
        (alpha, beta, z0) = (0.88, 1.4, 0.78).

        Examples
        --------
        >>> nz = zdist(0.88, 1.4, 0.78)
        >>> nz.A
        2.0124103366443378
        >>> nz.pdf([-0.5, 0, 0.5, 1.0, 1.5])
        array([0.        , 0.        , 0.79567909, 0.60772375, 0.29427866])

        >>> nz = zdist(0.88, 1.4, 0.78, zmax=1.4)
        >>> nz.A
        2.4324379840968535
        >>> nz.pdf([-0.5, 0, 0.5, 1.0, 1.5])
        array([0.        , 0.        , 0.96175219, 0.73456706, 0.        ])

        """
        # Check inputs
        assert alpha > 0 and beta > 0 and z0 > 0, "Inputs must be positive."

        self.alpha = alpha
        self.beta = beta
        self.z0 = z0
        self.zmax = zmax
        self.A = self._normalize(self.zmax)

    def _normalize(self, zmax):
        """Compute the normalization factor A."""
        if zmax is None:
            # Use the gamma function
            norm = self.beta / self.z0 / gamma((1. + self.alpha) / self.beta)
            self.zmax = np.inf
        else:
            # Use numerical integration
            integrand = (lambda z: ((z / self.z0)**self.alpha) *
                         np.exp(-((z / self.z0)**self.beta)))
            norm = 1. / quad(integrand, 0, zmax)[0]

        return norm

    def pdf(self, z):
        """Compute the probability density at a given redshift.

        Parameters
        ----------
        z : array_like
            Redshift values.

        Returns
        -------
        float or numpy array
            Probability density at each `z`.

        """
        # Check input
        z = np.atleast_1d(z)
        z[(z < 0) | (z > self.zmax)] = 0

        # Compute density
        x = z / self.z0
        result = self.A * (x**self.alpha) * np.exp(-(x**self.beta))

        # Clean up
        if len(result) == 1:
            result = result[0]

        return result

    def cdf(self, z):
        """Compute the cumulative probability up to a given redshift.

        Parameters
        ----------
        z : array_like
            Redshift values.

        Returns
        -------
        float or numpy array
            Cumulative probability up to each `z`.

        """
        # Check input
        z = np.atleast_1d(z)

        # Compute cumulative probability
        result = np.array([quad(lambda x: self.pdf(x), 0, zz)[0] for zz in z])

        # Clean up
        if len(result) == 1:
            result = result[0]

        return result

    def fit(self, samples, zmax=None):
        """Fit the distribution to a set of samples.

        Parameters
        ----------
        samples : array_like
            Redshift samples.
        zmax : float, optional
            Maximum redshift cutoff of the distribution. Default is None.

        Notes
        -----
        The current parameters (alpha, beta, z0) will be replaced by those
        of the fit.

        """
        # Bin samples
        hist, bin_edges = np.histogram(samples, bins=512, density=True)
        zvals = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find best fit parameters
        fitfunc = (lambda z, A, alpha, beta, z0:
                   A * ((z / z0)**alpha) * np.exp(-((z / z0)**beta)))
        params, cov = curve_fit(fitfunc, zvals, hist, p0=[1., 0.75, 1., 0.75])

        # Set new parameters
        self.alpha = params[1]
        self.beta = params[2]
        self.z0 = params[3]
        self.zmax = zmax
        self.A = self._normalize(self.zmax)
