# -*- coding: utf-8 -*-

"""PEAKS MODULE

This module contains functions for detecting and counting peaks (local maxima)
in images. Peak counts in weak-lensing maps are a useful statistic for
constraining cosmological models.

"""

import numpy as np


def find_peaks2d(image, threshold=None, ordered=True, mask=None,
                 include_border=False):
    """Identify peaks in an image (2D array) above a given threshold.

    A peak, or local maximum, is defined as a pixel of larger value than its
    eight neighbors. A mask may be provided to exclude certain regions from
    the search. The border is excluded by default.

    Parameters
    ----------
    image : array_like
        Two-dimensional input image.
    threshold : float, optional
        Minimum pixel amplitude to be considered as a peak. If not provided,
        the default value is set to the minimum of `image`.
    ordered : bool, optional
        If True, return peaks in decreasing order according to height.
    mask : array_like (same shape as `image`), optional
        Boolean array identifying which pixels of `image` to consider/exclude
        in finding peaks. A numerical array will be converted to binary, where
        only zero values are considered masked.
    include_border : bool, optional
        If True, include peaks found on the border of the image. Default is
        False.

    Returns
    -------
    X, Y, heights : tuple of 1D numpy arrays
        Pixel indices of peak positions and their associated heights.

    Notes
    -----
    The basic idea for this algorithm was provided by Chieh-An Lin.

    Examples
    --------
    TODO

    """
    image = np.atleast_2d(image)

    # Deal with the mask first
    if mask is not None:
        mask = np.atleast_2d(mask)
        if mask.shape != image.shape:
            print("Warning: mask not compatible with image -> ignoring.")
            mask = np.ones(image.shape)
        else:
            # Make sure mask is binary, i.e. turn nonzero values into ones
            mask = mask.astype(bool).astype(float)
    else:
        mask = np.ones(image.shape)

    # Add 1 pixel padding if including border peaks
    if include_border:
        image = np.pad(image, pad_width=1, mode='constant',
                       constant_values=image.min())
        mask = np.pad(mask, pad_width=1, mode='constant', constant_values=1)

    # Determine threshold level
    if threshold is None:
        # threshold = image[mask.astype('bool')].min()
        threshold = image.min()
    else:
        threshold = max(threshold, image.min())

    # Shift everything to be positive to properly handle negative peaks
    offset = image.min()
    threshold = threshold - offset
    image = image - offset

    # Extract the center map
    map0 = image[1:-1, 1:-1]

    # Extract shifted maps
    map1 = image[0:-2, 0:-2]
    map2 = image[1:-1, 0:-2]
    map3 = image[2:,   0:-2]
    map4 = image[0:-2, 1:-1]
    map5 = image[2:,   1:-1]
    map6 = image[0:-2, 2:  ]
    map7 = image[1:-1, 2:  ]
    map8 = image[2:,   2:  ]

    # Compare center map with shifted maps
    merge = ( (map0 > map1) & (map0 > map2) & (map0 > map3) & (map0 > map4)
            & (map0 > map5) & (map0 > map6) & (map0 > map7) & (map0 > map8) )

    bordered = np.lib.pad(merge, (1, 1), 'constant', constant_values=(0, 0))
    peaksmap = image * bordered * mask
    X, Y = np.nonzero(peaksmap > threshold)

    # Extract peak heights
    heights = image[X, Y] + offset

    # Compensate for border padding
    if include_border:
        X = X - 1
        Y = Y - 1

    # Sort peaks according to height
    if ordered:
        inds = np.argsort(heights)[::-1]
        return X[inds], Y[inds], heights[inds]

    return X, Y, heights


def peaks_histogram(image, bins=None, mask=None):
    """Compute a histogram of peaks in an image.

    Parameters
    ----------
    image : array_like
        Two-dimensional input image.
    bins : int or array_like (1D), optional
        Specification of bin edges or the number of bins to use for the
        histogram. If not provided, a default of 10 bins linearly spaced
        between the image minimum and maximum (inclusive) is used.
    mask : array_like (same shape as `image`), optional
        Boolean array identifying which pixels of `image` to consider/exclude
        in finding peaks. A numerical array will be converted to binary, where
        only zero values are considered masked.

    Returns
    -------
    counts, bin_edges : tuple of 1D numpy arrays
        Histogram and bin boundary values.

    Notes
    -----
    This function calls `find_peaks2d` and then uses `numpy` to compute the
    histogram. If the returned `counts` has N values, `bin_edges` will have
    N + 1 values.

    Examples
    --------
    TODO

    """
    # Define bin edges
    if bins is None:
        bins = np.linspace(image.min(), image.max(), 10)
    elif isinstance(bins, int):
        bins = np.linspace(image.min(), image.max(), bins)
    else:
        bins = np.atleast_1d(bins)

    # Compute peaks and histogram
    x, y, heights = find_peaks2d(image, threshold=None, mask=mask)
    counts, bin_edges = np.histogram(heights, bins)

    return counts, bin_edges
