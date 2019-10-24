# -*- coding: utf-8 -*-

import numpy as np


def angular_distance(ra1, dec1, ra2, dec2):
    """Determine the angular (geodesic) distance between points on a sphere.

    This function is vectorized, so the inputs can be given as arrays (of
    the same length), and the distances between each pair of corresponding
    points is computed.

    Parameters
    ----------
    ra[i], dec[i] : float or array_like
        Coordinates of point [i] on the sphere, where ra[i] is the
        longitudinal angle, and dec[i] is the latitudinal angle [degrees].

    Returns
    -------
    float or numpy array
        Central angle(s) between points [1] and [2] in degrees.

    Examples
    --------
    TODO

    """
    phi1 = np.deg2rad(ra1)
    theta1 = np.deg2rad(dec1)
    phi2 = np.deg2rad(ra2)
    theta2 = np.deg2rad(dec2)
    numerator = np.sqrt((np.cos(theta2) * np.sin(phi2 - phi1))**2 +
                        (np.cos(theta1) * np.sin(theta2) -
                         np.sin(theta1) * np.cos(theta2) *
                         np.cos(phi2 - phi1))**2)
    denominator = (np.sin(theta1) * np.sin(theta2) +
                   np.cos(theta1) * np.cos(theta2) * np.cos(phi2 - phi1))
    central_angle = np.rad2deg(np.arctan2(numerator, denominator))

    return central_angle


def solidangle(extent):
    """Compute the solid angle defined by a rectangle in RA/Dec space.

    Parameters
    ----------
    extent : array_like
        Field extent of the form [ra_min, ra_max, dec_min, dec_max].

    Returns
    -------
    float
        Solid angle in square degrees.

    Examples
    --------
    TODO

    """
    if len(np.atleast_1d(extent)) != 4:
        print("ERROR: extent must be of the form " +
              "[ra_min, ra_max, dec_min, dec_max].")
        return

    # Unpack
    ramin, ramax, decmin, decmax = extent

    if ramax <= ramin:
        print("ERROR: ra_max <= ra_min.")
        return
    if decmax <= decmin:
        print("ERROR: dec_max <= dec_min.")
        return

    # TODO could use wltools.conversions.to_rad here?
    def convert_to_rad(x):
        try:
            converted = x.to(u.radian)
        except AttributeError:
            if debug:
                print("No units for {}. Assuming degrees.".format(x))
            return convert_to_rad(x * u.degree)
        except u.UnitConversionError:
            if debug:
                print("Bad units for {}. Assuming degrees instead.".format(x))
            return convert_to_rad(x.value * u.degree)
        return converted.value

    # Convert inputs to radians
    alpha0 = convert_to_rad(ramin)
    alpha1 = convert_to_rad(ramax)
    delta0 = convert_to_rad(decmin)
    delta1 = convert_to_rad(decmax)

    # Truncate if values are out of bounds
    # alpha0 = max(0, alpha0)
    # alpha1 = min(2 * np.pi, alpha1)
    # delta0 = max(-np.pi / 2, delta0)
    # delta1 = min(np.pi / 2, delta1)

    # Compute solid angle
    sa = (alpha1 - alpha0) * (np.sin(delta1) - np.sin(delta0)) * u.radian**2

    return sa.to(u.degree**2)
