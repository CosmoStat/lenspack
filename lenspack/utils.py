# -*- coding: utf-8 -*-

"""UTILS MODULE

This module contains utility functions globally available to lenspack.

"""

import numpy as np
from astropy.units.core import Unit
from astropy.constants import G as G_newton
from astropy.constants import c as c_light


def round_up_to_odd(x):
    """Round up to the nearest odd integer."""
    return np.ceil(x) // 2 * 2 + 1


def convert_units(x, target):
    """Convert or attach units to a variable.

    Parameters
    ----------
    x : float
        Quantity to convert.
    target : str
        Target units given as an acceptable astropy.units string (e.g. 'km').

    An error will be raised if the conversion fails.

    Examples
    --------
    >>> conv(5, 'kpc')
    <Quantity 5. kpc>

    >>> x = 4e14
    >>> x = conv(x, 'solMass')
    >>> conv(x, 'kg')
    <Quantity 7.95390166e+44 kg>

    """
    try:
        x = x.to(Unit(target))
    except AttributeError:
        x = x * Unit(target)
    except Exception as e:
        raise

    return x


def sigma_critical(zl, zs, cosmology):
    """Critical surface mass density between a lens and source galaxy(-ies).

    Sigma_critical = [c^2 / (4 * pi * G)] * D_os / (D_ol * D_ls)

    Angular diameter distances D are calculated in a universe specified by
    an instance of astropy.cosmology.core.Cosmology.

    Parameters
    ----------
    zl : float
        Redshift of the lens.
    zs : array_like
        Redshift(s) of the source galaxies.
    cosmology : astropy.cosmology.core.Cosmology
        Cosmological model.

    Returns
    -------
    astropy.units.quantity.Quantity
        Critical surface mass density between a lens (i.e. cluster or DM halo)
        and each source redshift in units of solar masses per square parsec.
        For sources at the redshift of the halo and below, Sigma_critical is
        set to np.inf.

    Examples
    --------
    ...

    TODO
    ----
    Include the option for source redshift probability distributions.

    """
    # Ensure vectorization
    zs = np.atleast_1d(zs).astype(float)
    assert (zs >= 0).all(), "Redshifts must be positive."
    result = np.zeros_like(zs)

    # Compute distances
    d_ol = cosmology.angular_diameter_distance(zl)
    d_os = cosmology.angular_diameter_distance(zs)
    d_ls = cosmology.angular_diameter_distance_z1z2(zl, zs)

    # Avoid division by zero
    d_ls[d_ls == 0] = np.inf

    # Compute Sigma_crit
    factor = np.power(c_light, 2) / (4 * np.pi * G_newton)
    result = factor * d_os / (d_ol * d_ls)

    # Sources at lower z than the halo are not lensed
    result[result <= 0] = np.inf

    # Clean up
    if len(zs) == 1:
        result = result[0]

    return convert_units(result, "solMass / pc2")


def bin2d(x, y, npix=10, v=None, w=None, extent=None, verbose=False):
    """Bin samples of a spatially varying quantity according to position.

    The (weighted) average is taken of values falling into the same bin. This
    function is relatively general, but it is mainly used within this package
    to produce maps of the two components of shear from a galaxy catalog.

    Parameters
    ----------
    x, y : array_like
        1D position arrays.
    npix : int or list or tuple as (nx, ny), optional
        Number of bins in the `x` and `y` directions. If an int N is given,
        use (N, N). Binning defaults to (10, 10) if not provided.
    v : array_like, optional
        Values at positions (`x`, `y`). This can be given as many arrays
        (v1, v2, ...) of len(`x`) to bin simultaneously. If None, the bin
        count in each pixel is returned.
    w : array_like, optional
        Weights for `v` during averaging. If provided, the same weights are
        applied to each input `v`.
    extent : array_like, optional
        Boundaries of the resulting grid, given as (xmin, xmax, ymin, ymax).
        If None, bin edges are set as the min/max coordinate values of the
        input position arrays.
    verbose : boolean, optional
        If True, print details of the binning.

    Returns
    -------
    ndarray or tuple of ndarray
        2D numpy arrays of values `v` binned into pixels. The number of
        outputs matches the number of input `v` arrays.

    Examples
    --------
    # 100 values at random positions within the ranges -0.5 < x, y < 0.5
    # and binned within -1 < x, y < 1 to a (5, 5) grid.
    >>> x = np.random.random(100) - 0.5
    >>> y = np.random.random(100) - 0.5
    >>> v = np.random.randn(100) * 5
    >>> bin2d(x, y, v=v, npix=5, extent=(-1, 1, -1, 1))
    array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  4.43560619, -2.33308373,  0.48447844,  0.        ],
           [ 0.        ,  1.94903524, -0.29253335,  1.3694618 ,  0.        ],
           [ 0.        , -1.0202718 ,  0.37112266, -1.43062585,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

    """
    # Regroup extent if necessary
    if extent is not None:
        assert len(extent) == 4
        extent = [extent[:2], extent[2:]]

    if v is None:
        # Return the simple bin count map
        bincount, xbins, ybins = np.histogram2d(x, y, bins=npix, range=extent)
        result = bincount.T
    else:
        # Prepare values to bin
        v = np.atleast_1d(v)
        if len(v.shape) == 1:
            v = v.reshape(1, len(v))

        # Prepare weights
        if w is not None:
            w = np.atleast_1d(w)
            has_weights = True
        else:
            w = np.ones_like(x)
            has_weights = False

        # Compute weighted bin count map
        wmap, xbins, ybins = np.histogram2d(x, y, bins=npix, range=extent,
                                            weights=w)
        # Handle division by zero (i.e., empty pixels)
        wmap[wmap == 0] = np.inf
        # Compute mean values per pixel
        result = tuple((np.histogram2d(x, y, bins=npix, range=extent,
                        weights=(vv * w))[0] / wmap).T for vv in v)

        # Clean up
        if len(result) == 1:
            result = result[0]

    if verbose:
        if v is not None:
            print("Binning {} array{} with{} weights.".format(len(v),
                  ['', 's'][(len(v) > 1)], ['out', ''][has_weights]))
        else:
            print("Returning bin count map.")
        print("npix : {}".format(npix))
        print("extent : {}".format([xbins[0], xbins[-1], ybins[0], ybins[-1]]))
        print("(dx, dy) : ({}, {})".format(xbins[1] - xbins[0],
                                           ybins[1] - ybins[0]))

    return result


def radius2d(N, center=None, mode='exact'):
    """Distances from every pixel to a fixed center in a square matrix.

    Parameters
    ----------
    N : int
        Number of pixels to a side.
    center : array_like, optional
        Incides of the central pixel, given as (x0, y0). If not given, the
        center is taken to be (N / 2, N / 2) (though see `mode` description).
    mode : {'exact', 'fft'}
        How to treat the case when N is even. If 'exact', compute distances
        from the true (fractional) central pixel location. If 'fft', use the
        numpy.fft.fftfreq convention such that the central pixel location
        is rounded up to the nearest integer.

    Returns
    -------
    numpy array
        2D matrix of distances.

    Notes
    -----
    Non-integer center coordinates are not supported. If a `center` is
    provided, `mode` is ignored.

    Examples
    --------
    >>> radius2d(4, mode='exact')
    array([[ 2.12132034,  1.58113883,  1.58113883,  2.12132034],
           [ 1.58113883,  0.70710678,  0.70710678,  1.58113883],
           [ 1.58113883,  0.70710678,  0.70710678,  1.58113883],
           [ 2.12132034,  1.58113883,  1.58113883,  2.12132034]])

    >>> radius2d(4, mode='fft')
    array([[ 2.82842712,  2.23606798,  2.        ,  2.23606798],
           [ 2.23606798,  1.41421356,  1.        ,  1.41421356],
           [ 2.        ,  1.        ,  0.        ,  1.        ],
           [ 2.23606798,  1.41421356,  1.        ,  1.41421356]])

    """
    # Verify inputs
    N = int(N)
    assert mode in ('exact', 'fft'), "Mode must be either 'exact' or 'fft'."

    # Generate index grids
    x, y = np.indices((N, N))

    # Determine center
    if center is not None:
        x0, y0 = map(int, center)
    else:
        if mode == 'fft' and N % 2 == 0:
            x0 = N / 2.
            y0 = N / 2.
        else:
            x0 = (N - 1) / 2.
            y0 = (N - 1) / 2.

    # Compute radii
    return np.hypot(x - x0, y - y0)
