# -*- coding: utf-8 -*-

"""UTILS MODULE

This module contains some utility functions globally available to lenspack.

"""

import numpy as np
from astropy.units.core import Unit
from astropy.constants import G as G_newton
from astropy.constants import c as c_light


def convert_units(x, target):
    """Convert or attach units to a variable.

    Parameters
    ----------
    x : float
        Quantity to convert.
    target : str
        Target units given as an acceptable `astropy.units` string (e.g. 'km').

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
    """Critical surface mass density between a lens and source galaxy(ies).

    Sigma_critical = [c^2 / (4 * pi * G)] * D_os / (D_ol * D_ls)

    Angular diameter distances D are calculated in a universe specified by
    an instance of `astropy.cosmology.core.Cosmology`.

    Parameters
    ----------
    zl : float
        Redshift of the lens.
    zs : array_like
        Redshift(s) of the source galaxies.
    cosmology : `astropy.cosmology.core.Cosmology`
        Cosmological model.

    Returns
    -------
    `astropy.units.quantity.Quantity`
        Critical surface mass density between a lens (i.e. cluster or DM halo)
        and each source redshift in units of solar masses per square parsec.
        For sources at the redshift of the halo and below, Sigma_critical is
        set to np.inf.

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
