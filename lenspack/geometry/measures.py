# -*- coding: utf-8 -*-

import numpy as np


def angular_distance(ra1, dec1, ra2, dec2):
    """Determine the angular distance between points on the unit sphere.

    This function is vectorized, so the inputs can be given as arrays (of
    the same length), and the distances between each pair of corresponding
    points is computed.

    Parameters
    ----------
    ra[i], dec[i] : float or array_like
        Coordinates of point [i] on the sphere, where ra[i] is the
        longitudinal angle, and dec[i] is the latitudinal angle in degrees.

    Returns
    -------
    float or numpy array
        Central angle(s) between points [1] and [2] in degrees.

    Raises
    ------
    Exception
        For inputs of different length.

    Notes
    -----
    The geodesic distance between points on a sphere of radius R can be
    obtained by multiplying the result of `angular_distance` by R.

    Examples
    --------
    TODO

    """
    # Work in radians
    phi1 = np.deg2rad(ra1)
    theta1 = np.deg2rad(dec1)
    phi2 = np.deg2rad(ra2)
    theta2 = np.deg2rad(dec2)

    # Check input lengths
    if not (len(phi1) == len(theta1) == len(phi2) == len(theta2)):
        raise Exception("Input lengths must be the same.")

    numerator = np.sqrt((np.cos(theta2) * np.sin(phi2 - phi1))**2 +
                        (np.cos(theta1) * np.sin(theta2) -
                         np.sin(theta1) * np.cos(theta2) *
                         np.cos(phi2 - phi1))**2)
    denominator = (np.sin(theta1) * np.sin(theta2) +
                   np.cos(theta1) * np.cos(theta2) * np.cos(phi2 - phi1))
    central_angle = np.rad2deg(np.arctan2(numerator, denominator))

    return central_angle


def solid_angle(extent):
    """Compute the solid angle subtended by a rectangle in RA/Dec space.

    Parameters
    ----------
    extent : array_like
        Field extent as [ra_min, ra_max, dec_min, dec_max] in degrees, where
        ra_min <= ra_max and dec_min <= dec_max.

    Returns
    -------
    float
        Solid angle in square degrees.

    Raises
    ------
    Exception
        For inputs not in the correct format.

    Examples
    --------
    TODO

    """
    # Must have four input values
    if len(np.atleast_1d(extent)) != 4:
        raise Exception("Input extent must be of the form " +
                        "[ra_min, ra_max, dec_min, dec_max].")

    # Unpack
    ramin, ramax, decmin, decmax = extent

    # Check validity of bounds
    if ramax <= ramin:
        raise Exception("Must have ra_max >= ra_min.")
    if decmax <= decmin:
        raise Exception("Must have dec_max >= dec_min.")

    # Work in radians
    alpha0 = np.deg2rad(ramin)
    alpha1 = np.deg2rad(ramax)
    delta0 = np.deg2rad(decmin)
    delta1 = np.deg2rad(decmax)

    # Truncate if values are out of bounds
    # alpha0 = max(0, alpha0)
    # alpha1 = min(2 * np.pi, alpha1)
    # delta0 = max(-np.pi / 2, delta0)
    # delta1 = min(np.pi / 2, delta1)

    # Compute solid angle
    sa = (alpha1 - alpha0) * (np.sin(delta1) - np.sin(delta0))

    return sa * np.power(180 / np.pi, 2)
