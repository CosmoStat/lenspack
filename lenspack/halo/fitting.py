# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
try:
    import emcee
except ImportError:
    print("Warning: could not import emcee package.")
    print("         mcmcfit class cannot be used.")
from multiprocessing import Pool

from lenspack.halo.profiles import nfw_profile
from lenspack.utils import convert_units as conv


def lsq_fit(theta, gamma_t, err, z_halo, z_src, cosmology, model='nfw'):
    """Fit a halo profile to measured tangential shear data by least squares.

    Parameters
    ----------
    theta : array_like
        Angular distances (e.g. bin centers) from the halo/cluster center.
        Units of arcmin are assumed if not given as an astropy quantity.
    gamma_t : array_like
        Mean measured tangential shear at distances `theta`.
    err : array_like
        Error on `gamma_t`. This is typically the standard error of the mean
        in each `theta` bin.
    z_halo : float
        Redshift of the halo/cluster.
    z_src : float or array_like
        Effective redshift of source galaxies per `theta` bin. If only a
        single float is given, this value is used for all `theta` bins.
    cosmology : astropy.cosmology.core.Cosmology
        Assumed cosmological model in which the halo/cluster lives.
    model : {'nfw', 'bmo', 'einasto', 'sis'}, optional
        Halo model type. Currently only 'nfw' is supported, so it is default.
        See lenspack.halos.profiles for available options.

    Returns
    -------
    tuple of numpy arrays
        Best-fit parameters as ((c200, m200), cov), where cov is the 2x2
        covariance matrix.

    """
    # Check inputs
    assert len(theta) == len(gamma_t) == len(err), "Input lengths not equal."
    assert model in ('nfw', 'bmo', 'einasto', 'sis'), "Invalid model."

    # Convert angular theta to proper distance [Mpc]
    arcmin2mpc = conv(cosmology.kpc_proper_per_arcmin(z_halo), "Mpc / arcmin")
    r = conv(theta, 'arcmin') * arcmin2mpc

    def nfw_gamma_t(r, c200, m200):
        """Predicted shear profile of an NFW model."""
        halo = nfw_profile(z_halo, c200, m200=m200, cosmology=cosmology)
        # g_t = halo.gamma_t(r, z_src) / (1 - halo.kappa(r, z_src))
        return halo.gamma_t(r, z_src)

    def bmo_gamma_t(r, c200, m200):
        """Predicted shear profile of a BMO model."""
        # halo = bmo_profile(z_halo, c200, m200=m200, cosmology=cosmology)
        # return halo.gamma_t(r, z_src)
        pass

    # Add options here once the profiles are defined in lenspack.halo.profiles
    if model == 'nfw':
        model_gamma_t = nfw_gamma_t
    else:
        raise ValueError("Only the NFW model is currently supported.")

    # Fit the model
    p0 = (4, 5e14)  # Initial (c200, m200) values
    fit = curve_fit(model_gamma_t, xdata=r, ydata=gamma_t, sigma=err, p0=p0)

    return fit
