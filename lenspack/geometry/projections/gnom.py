# -*- coding: utf-8 -*-

import numpy as np


def radec2xy(ra0, dec0, ra, dec):
    """Project spherical sky coordinates to a tangent plane.

    Parameters
    ----------
    ra0 : float
        Right ascension of the projection origin.
    dec0 : float
        Declination of the projection origin.
    ra : float or array_like
        Right ascension of point(s) to project.
    dec : float or array_like
        Declination of point(s) to project.

    Notes
    -----
    All input units are assumed to be degrees.

    Returns
    -------
    x, y : tuple of floats or numpy arrays
        Projected coordinate(s) in the tangent plane relative to (0, 0), i.e.
        the origin in the projected space.

    Raises
    ------
    Exception
        For input arrays of different sizes.

    Examples
    --------
    ...

    """
    # Standardize inputs
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)

    if len(ra) != len(dec):
        raise Exception("Input ra and dec must have the same length.")

    # Convert input coordinates to radians
    alpha0 = np.deg2rad(ra0)
    delta0 = np.deg2rad(dec0)
    alpha = np.deg2rad(ra)
    delta = np.deg2rad(dec)

    # Project points
    denom = (np.cos(delta0) * np.cos(delta) * np.cos(alpha - alpha0) +
             np.sin(delta0) * np.sin(delta))
    x = np.cos(delta) * np.sin(alpha - alpha0) / denom
    y = ((np.cos(delta0) * np.sin(delta) -
          np.sin(delta0) * np.cos(delta) * np.cos(alpha - alpha0)) / denom)

    # Potentially remove unnecessary array layers
    if len(x) == 1:
        x, y = x[0], y[0]

    return x, y


def xy2radec(ra0, dec0, x, y):
    """Project tangent plane coordinates back to the spherical sky.

    Parameters
    ----------
    ra0 : float
        Right ascension of the projection origin.
    dec0 : float
        Declination of the projection origin.
    x : float or array_like
        X coordinate of point(s) in the tangent plane to de-project.
    y : float or array_like
        Y coordinate of point(s) in the tangent plane to de-project.

    Notes
    -----
    Projection origin (ra0, dec0) units are assumed to be degrees.

    Returns
    -------
    ra, dec : tuple of floats or numpy arrays
        De-projected (RA, Dec) value(s) on the sphere in degrees.

    Raises
    ------
    Exception
        For input arrays of different sizes.

    Examples
    --------
    ...

    """
    # Standardize inputs
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if len(x) != len(y):
        raise Exception("Input x and y must have the same length.")

    # Convert projection center to radians
    x0 = np.deg2rad(ra0)
    y0 = np.deg2rad(dec0)

    # Compute de-projected coordinates
    z = np.sqrt(x * x + y * y)
    c = np.arctan(z)

    # Prevent division by zero
    factor = np.ones(len(z))
    inds = (z != 0)
    factor[inds] = y[inds] / z[inds]

    delta = np.arcsin(np.cos(c) * np.sin(y0) + factor * np.cos(y0) * np.sin(c))
    denom = z * np.cos(y0) * np.cos(c) - y * np.sin(y0) * np.sin(c)
    alpha = x0 + np.arctan2(x * np.sin(c), denom)

    # Convert output coordinates to degrees
    ra = np.rad2deg(alpha)
    dec = np.rad2deg(delta)

    # Potentially remove unnecessary array layers
    if len(ra) == 1:
        ra, dec = ra[0], dec[0]

    return ra, dec


class projector(object):
    """A convenient class for many gnomonic projections to a tangent plane."""
    def __init__(self, ra0, dec0):
        """Project spherical coordinates to a tangent plane.

        The fixed center of the (gnomonic) projection is at (ra0, dec0).
        Inputs should be in degrees.

        """
        self.ra0 = float(ra0)
        self.dec0 = float(dec0)

    def radec2xy(self, ra, dec):
        """Projection of spherical coordinates to the tangent plane.

        Parameters
        ----------
        ra, dec : float or array_like
            RA and Dec coordinates to project.

        Returns projected x, y coordinates.

        """
        return radec2xy(self.ra0, self.dec0, ra, dec)

    def xy2radec(self, x, y):
        """Inverse projection of tangent plane points back to the sphere.

        Parameters
        ----------
        x, y : float or array_like
            Coordinates of points in the tangent plane to de-project relative
            to (0, 0) corresponding to (ra0, dec0) on the sphere.

        Returns de-projected ra, dec coordinates.

        """
        return xy2radec(self.ra0, self.dec0, x, y)
