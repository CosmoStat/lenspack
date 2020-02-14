# -*- coding: utf-8 -*-

"""SHEAR MODULE

This module contains functions related to weak-lensing shear, for example
measuring the tangential/cross components of shear about a point and computing
the observed ellipticity of a galaxy that has been sheared by weak lensing.

"""

import numpy as np
from lenspack.geometry.measures import spherical_polar_angle


def gamma_tx(ra, dec, gamma1, gamma2, center=(0, 0), coordinates='spherical'):
    """Compute the tangential/cross components of shear about a point.

    Parameters
    ----------
    x, y : array_like, 1D
        Cartesian positions of points/galaxies on the sky. A Euclidean metric
        is assumed for calculating distances between the points.
    gamma1, gamma2 : array_like, 1D
        Two components of shear/ellipticity at the (`x`, `y`) locations.
    center : tuple of floats, optional
        Reference position. Default is (0, 0).
    coordinates : {'spherical', 'cartesian'}
        Whether to treat (ra, dec) positions as points on the sphere
        ('spherical') or the plane ('cartesian').

    Returns
    -------
    gamma_t, gamma_x : tuple of floats or 1D numpy arrays
        Tangential and cross shear/ellipticity component for each input.

    Notes
    -----
    The equations are from Schneider's 2005 weak-lensing review [Eq. (17)].

        gamma_t = -Re[gamma * exp(-2i * phi)],
        gamma_x = -Im[gamma * exp(-2i * phi)],

    where gamma is the complex shear and phi is the polar angle relative
    to the center point.

    """
    # Standardize inputs
    ra = np.atleast_1d(ra).flatten()
    dec = np.atleast_1d(dec).flatten()
    gamma1 = np.atleast_1d(gamma1).flatten()
    gamma2 = np.atleast_1d(gamma2).flatten()

    if not len(ra) == len(dec) == len(gamma1) == len(gamma2):
        raise Exception("Input array lengths must be equal.")

    coordinates = coordinates.lower()
    coord_error = "Coordinates must be either 'spherical' or 'cartesian'."
    assert coordinates in ('spherical', 'cartesian'), coord_error

    # Determine polar angles
    if coordinates == 'spherical':
        phi = spherical_polar_angle(center[0], center[1], ra, dec)  # [0, 2pi]
    else:
        phi = np.arctan2(dec - center[1], ra - center[0])  # [-pi, pi]
    angle = 2 * phi

    # Compute tangential shear/ellipticity components
    gamma_t = -gamma1 * np.cos(angle) - gamma2 * np.sin(angle)
    gamma_x = gamma1 * np.sin(angle) - gamma2 * np.cos(angle)

    # Clean up
    if len(gamma_t) == 1:
        gamma_t = gamma_t[0]
        gamma_x = gamma_x[0]

    return gamma_t, gamma_x


def apply_shear(e1, e2, g1, g2):
    """Shear galaxies with intrinsic ellipticity by a (reduced) shear.

    Parameters
    ----------
    e1, e2 : array_like (1D)
        Intrinsic ellipticity components of galaxies.
    g1, g2 : array_like (1D)
        Weak-lensing reduced shear components to apply to e = e1 + ie2.

    Returns
    -------
    e1_obs, e2_obs : tuple of floats or (1D) numpy arrays
        Two components of observed ellipticity.

    Warnings
    --------
    A warning will be printed if applying a non-weak shear,
    i.e. if abs(g) >= 1.

    """
    # Standardize inputs
    e1 = np.atleast_1d(e1).flatten()
    e2 = np.atleast_1d(e2).flatten()
    g1 = np.atleast_1d(g1).flatten()
    g2 = np.atleast_1d(g2).flatten()

    if not len(e1) == len(e2) == len(g1) == len(g2):
        raise Exception("Input array lengths must be equal.")

    # Complex intrinsic ellipticity
    e_int = e1 + 1j * e2
    # Compute observed ellipticity
    g = g1 + 1j * g2
    e_obs = (e_int + g) / (1. + g.conjugate() * e_int)
    # Idenfity galaxies for which reduced shear exceeds 1
    bad_inds = np.abs(g) >= 1
    # Use the alternative formula for positions with g > 1
    if sum(bad_inds) > 0:
        numer = 1. + g[bad_inds] * e_int.conjugate()[bad_inds]
        denom = e_int.conjugate()[bad_inds] + g.conjugate()[bad_inds]
        e_obs[bad_inds] = numer / denom
        print("Warning : {} galaxies with |g| >= 1.".format(sum(bad_inds)))

    # Clean up
    if len(e_obs) == 1:
        e_obs = e_obs[0]

    return e_obs.real, e_obs.imag


def random_rotation(gamma1, gamma2):
    """Randomly rotate a spin-2 quantity, such as galaxy ellipticity.

    Parameters
    ----------
    gamma1, gamma2 : array_like (1D)
        First and second components of the spin-2 quantity.

    Returns
    -------
    tuple of floats or 1D numpy arrays
        First and second components of the rotated quantity.

    """
    # Standardize inputs
    gamma1 = np.atleast_1d(gamma1).flatten()
    gamma2 = np.atleast_1d(gamma2).flatten()

    if not len(gamma1) == len(gamma2):
        raise Exception("Input array lengths must be equal.")

    # Rotate
    theta = np.pi * np.random.random(len(gamma1))
    new_gamma1 = np.cos(theta) * gamma1 - np.sin(theta) * gamma2
    new_gamma2 = np.sin(theta) * gamma1 + np.cos(theta) * gamma2

    # Clean up
    if len(new_gamma1) == 1:
        new_gamma1 = new_gamma1[0]
        new_gamma2 = new_gamma2[0]

    return new_gamma1, new_gamma2
