# -*- coding: utf-8 -*-

"""STATS MODULE

This module contains some common statistical measures useful in weak-lensing
studies. For example, the higher-order moments of filtered convergence maps
can be used to constrain cosmological parameters.

"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, gaussian_kde


def mad(x):
    """Compute median absolute deviation (MAD) of an array.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns the MAD of `x` as a float.

    """
    return np.median(np.abs(x - np.median(x)))


def skew(x):
    """Compute the skewness of an array as the third standardized moment.

    Parameters
    ----------
    x : array_like
        Input array. If `x` is multidimensional, skewness is automatically
        computed over all elements in the array, not just along one axis.

    Returns
    -------
    float
        Skewness of `x`.

    Notes
    -----
    For array x, the calculation is carried out as E[((x - mu) / sigma)^3],
    where E() is the expectation value, mu is the mean of x, and sigma is the
    uncorrected sample standard deviation.

    This is equivalent to
        >>> n = len(x)
        >>> numer = np.power(x - mu, 3).sum() / n
        >>> denom = np.power(np.power(x - mu, 2).sum() / (n - 1), 3 / 2)
        >>> skew = numer / denom
    and to the normalized 3rd-order central moment
        >>> mu_n(x, 3, normed=True)

    This function's output matches that of scipy.stats.skew exactly, but
    the latter is typically faster.

    """
    mean = x.mean(axis=None)
    sigma = x.std(ddof=0, axis=None)
    sigma = 1 if sigma <= 0 else sigma
    return np.mean(np.power((x - mean) / sigma, 3), axis=None)


def kurt(x, fisher=True):
    """Compute the kurtosis of an array as the fourth standardized moment.

    Parameters
    ----------
    x : array_like
        Input array. If `x` is multidimensional, kurtosis is automatically
        computed over all elements in the array, not just along one axis.
    fisher : bool, optional
        If True, use Fisher's normalization, i.e. subtract 3 from the result.

    Returns
    -------
    float
        Kurtosis of `x`.

    Notes
    -----
    For array x, the calculation is carried out as E[((x - mu) / sigma)^4],
    where E() is the expectation value, mu is the mean of x, and sigma is the
    uncorrected sample standard deviation.

    This is equivalent to the normalized 4th-order central moment
        >>> mu_n(x, 4, normed=True) - 3

    This function's output matches that of scipy.stats.kurtosis very well, but
    the latter is typically faster.

    """
    mean = x.mean(axis=None)
    sigma = x.std(ddof=0, axis=None)
    sigma = 1 if sigma <= 0 else sigma
    return np.mean(np.power((x - mean) / sigma, 4), axis=None) - [0, 3][fisher]


def mu_n(x, order, normed=False):
    """Compute the (normalized) nth-order central moment of a distribution.

    Parameters
    ----------
    x : array_like
        Input array. If `x` is multidimensional, the moment is automatically
        computed over all elements in the array, not just along one axis.
    order : int (positive)
        Order of the moment.
    normed : bool, optional
        If True, normalize the result by sigma^order, where sigma is the
        corrected sample standard deviation of `x`. Default is False.

    Returns
    -------
    float
        Nth-order moment of `x`.

    Notes
    -----
    This function's output matches that of scipy.stats.moment very well, but
    the latter is typically faster.

    """
    # Check order
    order = int(order)
    assert order > 0, "Only n > 0 supported."

    x = np.atleast_1d(x).flatten()
    mean = x.mean()

    if normed and order > 2:
        sigma = x.std(ddof=1)
        result = np.power((x - mean) / sigma, order).mean()
    else:
        result = np.power(x - mean, order).mean()

    return result


def kappa_n(x, order):
    """Compute the nth-order cumulant of a distribution.

    Parameters
    ----------
    x : array_like
        Input array. If `x` is multidimensional, the cumulant is automatically
        computed over all elements in the array, not just along one axis.
    order : int between 2 and 6, inclusive
        Order of the cumulant.

    Returns
    -------
    float
        Nth-order cumulant of `x`.

    Notes
    -----
    This function's output matches that of scipy.stats.kstat very well, but
    the latter is typically faster.

    """
    # Check order
    order = int(order)
    assert order in range(2, 7), "Order {} not supported.".format(order)

    if order == 2:
        kappa = mu_n(x, 2, normed=False)
    if order == 3:
        kappa = mu_n(x, 3, normed=False)
    if order == 4:
        mu2 = mu_n(x, 2, normed=False)
        mu4 = mu_n(x, 4, normed=False)
        kappa = mu4 - 3 * mu2**2
    if order == 5:
        mu2 = mu_n(x, 2, normed=False)
        mu3 = mu_n(x, 3, normed=False)
        mu5 = mu_n(x, 5, normed=False)
        kappa = mu5 - 10 * mu3 * mu2
    if order == 6:
        mu2 = mu_n(x, 2, normed=False)
        mu3 = mu_n(x, 3, normed=False)
        mu4 = mu_n(x, 4, normed=False)
        mu6 = mu_n(x, 6, normed=False)
        kappa = mu6 - 15 * mu4 * mu2 - 10 * mu3**2 + 30 * mu2**3

    return kappa


def fdr(x, tail, alpha=0.05, kde=False, n_samples=10000, debug=False):
    """Compute the false discovery rate (FDR) threshold of a distribution.

    Parameters
    ----------
    x : array_like
        Samples of the distribution. If `x` is multidimentional, it will
        first be flattened.
    tail : {'left', 'right'}
        Side of the distribution for which to compute the threshold.
    alpha : float, optional
        Maximum average false discovery rate. Default is 0.05.
    kde : bool, optional
        If True, compute p-values from the distribution smoothed by kernel
        density estimation. Not recommended for large `x`. Default is False.
    n_samples : int, optional
        Number of samples to draw if using KDE. Default is 10000.
    debug : bool, optional
        If True, print the number of detections and the final p-value.
        Default is False.

    Returns
    -------
    float
        FDR threshold.

    Examples
    --------
    ...

    """
    # Check inputs
    assert tail in ('left', 'right'), "Invalid tail."

    # Basic measures
    x = np.atleast_1d(x).flatten()
    N = len(x)
    mean = x.mean()
    sigma = x.std(ddof=1)

    if kde:
        # Approximate the distribution by kernel density estimation
        pdf = gaussian_kde(x, bw_method='silverman')
        amin, amax = x.min(), x.max()
        vmax = np.sign(amax) * np.abs(amax + sigma)
        vmin = np.sign(amin) * np.abs(amin - sigma)

        # Cumulative distribution function
        def cdf(z, tail):
            if tail == 'right':
                return pdf.integrate_box(-np.inf, z)
            else:
                return pdf.integrate_box(z, np.inf)

        # Inverse cumulative distribution function
        def cdfinv(t, tail):
            assert t > 0 and t < 1, "Input to cdfinv must be in (0, 1)."
            try:
                result = brentq(lambda x: cdf(x, tail) - t, vmin, vmax)
            except ValueError:
                print("Warning: bad value. Returning 0.")
                print("threshold = {}".format(t))
                print("vmin = {}".format(vmin))
                print("vmax = {}".format(vmax))
                result = 0
            return result

        # Compute p-values
        if tail == 'right':
            smin = mean + sigma
            smax = vmax
        else:
            smax = mean - sigma
            smin = vmin
        samples = (smax - smin) * np.random.random(n_samples) + smin
        pvals = 1 - np.array([cdf(s, tail) for s in samples])
    else:
        # Compute p-values assuming Gaussian null hypothesis
        z = (x - mean) / sigma
        if tail == 'right':
            pvals = 1 - norm.cdf(z)
        else:
            pvals = norm.cdf(z)

    # Sort p-values and find the last one
    inds = np.argsort(pvals)
    ialpha = alpha * np.arange(1, len(pvals) + 1) / float(len(pvals))
    diff = pvals[inds] - ialpha
    lastindex = np.argmax((diff < 0).cumsum())
    pval = pvals[inds][lastindex]
    if debug:
        print("{} detection(s) / {}".format(lastindex + 1, len(pvals)))
        print("last pval = {}".format(pval))

    # Invert pval to get threshold
    if kde:
        tval = cdfinv(1 - pval, tail)
    else:
        tval = x[inds][lastindex]

    return tval
