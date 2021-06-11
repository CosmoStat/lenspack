# -*- coding: utf-8 -*-

"""STARLET L1-NORM MODULE

This module contains functions for computing the starlet l1norm
as defined in Eq. (1) of https://arxiv.org/pdf/2101.01542.pdf.

"""

import numpy as np
from astropy.stats import mad_std
from lenspack.image.transforms import starlet2d


def noise_coeff(image, nscales):
    
    """Compute the noise coefficients $\sigma_j$ 
       to get the estimate of the noise at the scale j
       following Starck and Murtagh (1998).
       
    Parameters
    ----------
    image : array_like
        Two-dimensional input image.
    nscales : int
        Number of wavelet scales to compute. Should not exceed log2(N), where
        N is the smaller of the two input dimensions.
    
    Returns
    -------
    sigma_j : numpy.ndarray
        Values of the standard deviation of the noise at scale j
    """
         
    noise_sigma=np.random.randn(image.shape[0],image.shape[0])
    noise_wavelet=starlet2d(noise_sigma, nscales)
    sigma_j=np.array([np.std(scale) for scale in noise_wavelet])
    return sigma_j


def get_l1norm_noisy(image, noise, nscales, nbins):
    
    """Compute the starlet $\ell_1$-norm of a noisy image
       following Eq. (1) of https://arxiv.org/abs/2101.01542. 
       
    Parameters
    ----------
    image : array_like
        Two-dimensional input noiseless image.
    noise : array_like
        Two-dimensional input of the noise to be added to image
    nscales : int
        Number of wavelet scales to compute. Should not exceed log2(N), where
        N is the smaller of the two input dimensions.
    nbins : int
        Number of bins in S/N desired for the summary statistic
    
    Returns
    -------
        bins_snr, starlet_l1norm : tuple of 1D numpy arrays
        Bin centers in S/N and Starlet $\ell_1$-norm of the noisy image
    """

    #add noise to noiseless image
    image_noisy=image+noise
    
    #perform starlet decomposition
    image_starlet = starlet2d(image_noisy, nscales)

    # estimate of the noise
    noise_estimate = mad_std(image_noisy)
    
    std_coeff=noise_coeff(image, nscales)

    l1_coll = []
    bins_coll = []
    for image_temp, std_co in zip(image_starlet, std_coeff):

        std_scalej = std_co*noise_estimate

        snr = image_temp/std_scalej
        thresholds_snr = np.linspace(np.min(snr), np.max(snr), nbins+1)
        bins_snr=0.5*(thresholds_snr[:-1] + thresholds_snr[1:])
        digitized = np.digitize(snr, thresholds_snr)
        bin_l1_norm = [np.sum(np.abs(snr[digitized == i])) for i in range(1, len(thresholds_snr))]
        l1_coll.append(bin_l1_norm)
        bins_coll.append(bins_snr)
        
    return np.array(bins_coll), np.array(l1_coll)
