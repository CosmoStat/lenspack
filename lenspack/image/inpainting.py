# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import erf

from lenspack.image.transforms import dct2d, idct2d


def inpaint(image, mask, niter=100, thresholding='hard'):
    """Fill in gaps in an image using the Discrete Cosine Transform (DCT).

    Parameters
    ----------
    image : array_like (2D)
        Input image.
    mask : array_like (2D)
        Boolean or binary mask representing the missing pixels of `image`.
        False or zero pixel values are considered to be masked.
    niter : int, optional
        Number of iterations. Default is 100.
    thresholding : {'soft', 'hard'}, optional
        Type of thresholding. Default is 'hard'.

    Returns
    -------
    2D numpy array
        Inpainted image.

    References
    ----------
    * Elad, Starck, Querre, & Donoho, ACHA 19, 340 (2005)
    * Pires, Starck, Amara, et al., MNRAS 395, 1265 (2009)

    """
    # Check inputs
    assert image.shape == mask.shape, "Incompatible mask."
    assert thresholding in ('soft', 'hard'), "Invalid thresholding."

    # Enforce binary mask condition
    mask = mask.astype(bool).astype(float)

    # Set threshold limits
    lmax = np.max(np.abs(dct2d(image, norm='ortho')))
    lmin = 0

    # Do iterative inpainting
    result = np.zeros_like(image)
    for ii in range(niter):
        # Compute residual
        residual = image - result
        # Take a step
        update = result + mask * residual
        # Change basis with DCT
        alpha = dct2d(update, norm='ortho')
        # Threshold coefficients
        # lval = lmax - ii * (lmax - lmin) / (niter - 1)  # linear decay
        lval = lmin + (lmax - lmin) * (1 - erf(2.8 * ii / niter))  # exp decay
        new_alpha = np.copy(alpha)  # Can we do this without copying ?
        if thresholding == 'hard':
            new_alpha[np.abs(new_alpha) <= lval] = 0
        else:
            new_alpha = np.abs(new_alpha) - lval
            new_alpha[new_alpha < 0] = 0
            new_alpha = np.sign(alpha) * new_alpha
        # Go back to direct space
        result = idct2d(new_alpha, norm='ortho')
        # Enforce std. dev. constraint inside the mask
        std_out = result[mask.astype(bool)].std()
        std_in = result[~mask.astype(bool)].std()
        if std_in != 0:
            result[~mask.astype(bool)] *= std_out / std_in

    return result
