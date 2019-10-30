# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import convolve1d


def starlet2d(image, nscales=5):
    """Compute the multiscale 2D starlet transform of an image.

    Parameters
    ----------
    image : array_like (2D)
        Input image.
    nscales : int
        Number of wavelet scales to compute.

    Returns
    -------
    3D numpy array
        Wavelet coefficients. An input `image` of shape (N, M) and `nscales`
        equal to n will produce an output of shape (n + 1, N, M). The first
        n maps are wavelet coefficients, while the final map is the coarse
        scale map.

    Notes
    -----
    This function produces the same output as the mr_transform binary of the
    iSAP C++ code package (see References) to good accuracy.

    References
    ----------
    * Starck, Murtagh, Fadili, 'Sparse Image and Signal Processing: Wavelets
      and Related Geometric Multiscale Analysis', Cambridge University Press,
      Cambridge (GB), 2016.
    * http://www.cosmostat.org/software/isap

    Examples
    --------
    # Transform a random image of standard deviation 10.
    >>> img = 10 * np.random.randn(64, 64)
    >>> wt = starlet2d(img, 5)
    >>> wt.shape
    (6, 64, 64)

    # Reconstruction
    >>> rec = np.sum(wt, axis=0)
    >>> rec.shape == img.shape
    True
    >>> np.sum(np.abs(rec - img))
    2.4814638191483773e-12

    """
    # Filter banks
    h = np.array([1, 4, 6, 4, 1]) / 16.
    # g = np.array([0, 0, 1, 0, 0]) - h

    # Setting for convolve1d in order to match output of mr_transform
    mode = 'nearest'

    # Initialize output
    result = np.zeros((nscales + 1, image.shape[0], image.shape[1]))
    cj = image

    # Compute multiscale starlet transform
    for j in range(nscales):
        # Create j-level version of h for à trous algorithm
        if j > 0:
            hj = np.array([[x] + [0] * (2**j - 1) for x in h]).flatten()
            hj = hj[:-(2**j - 1)]
        else:
            hj = h
        # Smooth coefficients at scale j+1
        cjplus1 = convolve1d(convolve1d(cj, weights=hj, mode=mode, axis=0),
                             weights=hj, mode=mode, axis=1)
        # Wavelet coefficients at scale j
        result[j] = cj - cjplus1
        cj = cjplus1
    result[-1] = cj

    return result
