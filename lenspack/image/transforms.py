# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np
from shutil import which
from subprocess import call
from astropy.io import fits
from scipy.ndimage import convolve1d
from scipy.fftpack import dct, idct


def starlet2d(image, nscales=5):
    """Compute the multiscale 2D starlet transform of an image.

    Parameters
    ----------
    image : array_like (2D)
        Input image.
    nscales : int
        Number of wavelet scales to compute. Should not exceed log2(N), where
        N is the smaller of the two input dimensions.

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
    Sparse2D C++ code package (see References) to high precision.

    References
    ----------
    * Starck, Murtagh, Fadili, 'Sparse Image and Signal Processing: Wavelets
      and Related Geometric Multiscale Analysis', Cambridge University Press,
      Cambridge (GB), 2016.
    * https://github.com/cosmostat/sparse2d

    Examples
    --------
    >>> # Transform a Gaussian random field of standard deviation 10.
    >>> img = 10 * np.random.randn(64, 64)
    >>> wt = starlet2d(img, 5)
    >>> wt.shape
    (6, 64, 64)

    >>> # Reconstruction
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


def dct2d(image, norm='ortho'):
    """Compute the discrete cosine transform (type 2) of an image.

    Parameters
    ----------
    image : array_like, 2D
        Input image.
    norm : {None, 'ortho', 'sparse2d'}, optional
        Normalization option. See scipy.fftpack.dct documentation (Type II)
        for a description of the None and 'ortho' options. The 'sparse2d'
        option is available to match the output from the im_dct Sparse2D
        binary, which involves an additional scaling of the zero-frequency
        elements. Default is 'ortho'.

    Returns
    -------
    2D numpy array
        Type 2 DCT.

    Notes
    -----
    Using no normalization (i.e. norm=None) will not automatically
    recover the original image after performing the inverse transformation.
    Each transform brings an overall scaling factor of 2N.

    See Also
    --------
    idct2d
        Inverse 2D DCT.

    Examples
    --------
    ...

    """
    # Check inputs
    image = np.array(image)
    assert len(image.shape) == 2, "Input image must be 2D."
    assert norm in (None, 'ortho', 'sparse2d'), "Invalid norm."

    # Compute DCT along each axis
    if norm == 'sparse2d':
        result = dct(dct(image, norm='ortho', axis=0), norm='ortho', axis=1)
        result[:, 0] *= np.sqrt(2)
        result[0, :] *= np.sqrt(2)
    else:
        result = dct(dct(image, norm=norm, axis=0), norm=norm, axis=1)

    return result


def idct2d(image, norm='ortho'):
    """Compute the inverse discrete cosine transform (type 2) of an image.

    Parameters
    ----------
    image : array_like (2D)
        Input image.
    norm : {None, 'ortho', 'sparse2d'}, optional
        Normalization option. Default is 'ortho'.

    Returns
    -------
    2D numpy array
        Inverse type 2 DCT.

    See Also
    --------
    dct2d
        Forward 2D DCT.

    Examples
    --------
    ...

    """
    # Check inputs
    image = np.array(image)
    assert len(image.shape) == 2, "Input image must be 2D."
    assert norm in (None, 'ortho', 'sparse2d'), "Invalid norm."

    # Compute inverse DCT along each axis
    if norm == 'sparse2d':
        image[:, 0] /= np.sqrt(2)
        image[0, :] /= np.sqrt(2)
        result = idct(idct(image, norm='ortho', axis=0), norm='ortho', axis=1)
    else:
        result = idct(idct(image, norm=norm, axis=0), norm=norm, axis=1)

    return result


def blockdct2d(image, norm='ortho', blocksize=None, overlap=False):
    """Compute a block (local) discrete cosine transform of an image.

    This is an extension of dct2d to perform the transform on sub-blocks
    of the image.

    Parameters
    ----------
    image : array_like, 2D
        Input image.
    norm : {None, 'ortho', 'sparse2d'}, optional
        Normalization option. See scipy.fftpack.dct documentation (Type II)
        for a description of the None and 'ortho' options. The 'sparse2d'
        option is available to match the output from the im_dct Sparse2D
        binary, which involves an additional scaling of the zero-frequency
        elements. Default is 'ortho'.
    blocksize : int, optional
        Size of sub-blocks for a local DCT.
    overlap : bool, optional
        Whether to overlap sub-blocks.

    Returns
    -------
    2D numpy array
        Local type 2 DCT.

    See Also
    --------
    iblockdct2d
        Inverse local 2D DCT.

    Examples
    --------
    ...

    TODO
    -----
    This needs MORE TESTING before deployment !

    """
    # Check inputs
    image = np.array(image)
    assert len(image.shape) == 2, "Input image must be 2D."
    assert image.shape[0] == image.shape[1], "Input image must be square."
    assert norm in (None, 'ortho', 'sparse2d'), "Invalid norm."

    # Determine output shape based on blocksize
    n = image.shape[0]
    if blocksize is not None:
        if blocksize == n:
            result = np.zeros_like(image)
        elif blocksize not in [n / 2, n / 4, n / 8]:
            print("Warning: invalid blocksize --> using {}".format(n))
            blocksize = n
            result = np.zeros_like(image)
        else:
            if overlap:
                size = 2 * n - blocksize
                result = np.zeros((size, size))
            else:
                result = np.zeros_like(image)
    else:
        blocksize = n
        result = np.zeros_like(image)

    print(blocksize)

    # Compute DCT on sub blocks
    if overlap:
        for ii in range(2 * n / blocksize - 1):
            for jj in range(2 * n / blocksize - 1):
                i1 = ii * blocksize
                i2 = i1 + blocksize
                j1 = jj * blocksize
                j2 = j1 + blocksize
                imsub = image[i1 / 2: i1 / 2 + blocksize,
                              j1 / 2: j1 / 2 + blocksize]
                result[i1:i2, j1:j2] = dct2d(imsub, norm=norm)
    else:
        for ii in range(0, n, blocksize):
            for jj in range(0, n, blocksize):
                i1 = ii
                i2 = ii + blocksize
                j1 = jj
                j2 = jj + blocksize
                imsub = image[i1:i2, j1:j2]
                result[i1:i2, j1:j2] = dct2d(imsub, norm=norm)

    return result


def iblockdct2d(image, norm='ortho', blocksize=None, overlap=False):
    """Compute the inverse block (local) discrete cosine transform of an image.

    This is an extension of idct2d to perform the transform on sub-blocks
    of the image.

    Parameters
    ----------
    image : array_like, 2D
        Input image.
    norm : {None, 'ortho', 'sparse2d'}, optional
        Normalization option. See scipy.fftpack.dct documentation (Type II)
        for a description of the None and 'ortho' options. The 'sparse2d'
        option is available to match the output from the im_dct Sparse2D
        binary, which involves an additional scaling of the zero-frequency
        elements. Default is 'ortho'.
    blocksize : int, optional
        Size of sub-blocks for a local inverse DCT.
    overlap : bool, optional
        Whether to overlap sub-blocks.

    Returns
    -------
    2D numpy array
        Local type 2 inverse DCT.

    Examples
    --------
    ...

    TODO
    -----
        This needs MORE TESTING before deployment !

    """
    if norm not in [None, 'ortho', 'sparse2d']:
        print("Warning: invalid norm --> using sparse2d")
        norm = 'sparse2d'

    # Determine output shape
    n = image.shape[0]
    if blocksize is not None:
        if blocksize == n:
            result = np.zeros_like(image)
        else:
            if overlap:
                size = (n + blocksize) / 2
                result = np.zeros((size, size))
            else:
                result = np.zeros_like(image)
    else:
        blocksize = n
        result = np.zeros_like(image)

    # Compute inverse DCT on sub blocks
    if overlap:
        for ii in range(n / blocksize):
            for jj in range(n / blocksize):
                i1 = ii * blocksize
                i2 = i1 + blocksize
                j1 = jj * blocksize
                j2 = j1 + blocksize
                i1r = i1 / 2
                i2r = i1r + blocksize
                j1r = j1 / 2
                j2r = j1r + blocksize
                imsub = image[i1:i2, j1:j2]
                result[i1r:i2r, j1r:j2r] += idct(imsub, norm=norm)

        # Take averages
        step = blocksize / 2
        counts = np.ones_like(result)
        counts[step:-step, :step] = 2
        counts[step:-step, -step:] = 2
        counts[:step, step:-step] = 2
        counts[-step:, step:-step] = 2
        counts[step:-step, step:-step] = 4
        result /= counts
    else:
        for ii in range(0, n, blocksize):
            for jj in range(0, n, blocksize):
                i1 = ii
                i2 = ii + blocksize
                j1 = jj
                j2 = jj + blocksize
                imsub = image[i1:i2, j1:j2]
                result[i1:i2, j1:j2] = idct(imsub, norm=norm)

    return result


def mr_transform(image, nscales=4, type=2, verbose=False):
    """Compute the multi-resolution wavelet transform of an image.

    Parameters
    ----------
    image : array_like, 2D
        Input image.
    nscales : int, optional
        Number of wavelet scales to compute. Default is 4.
    type : int, optional
        Type of the multiresolution transform. See the original mr_transform
        documentation for details. Default is 2, which corresponds to the
        'bspline wavelet transform: a trous algorithm', i.e. the starlet.
    verbose : bool, optional
        If True, print details of the temporary file I/O process.

    Returns
    -------
    3D numpy array
        Result of the wavelet transform.

    Notes
    -----
    This function is a wrapper for the mr_transform C++ binary of the Sparse2D
    code package (see References). The astropy package is necessary to write
    out `image` as a temporary fits file on which mr_transform can act.

    References
    ----------
    * Starck, Murtagh, Fadili, 'Sparse Image and Signal Processing: Wavelets
      and Related Geometric Multiscale Analysis', Cambridge University Press,
      Cambridge (GB), 2016.
    * https://github.com/cosmostat/sparse2d

    Examples
    --------
    ...

    """
    # Verify that mr_transform is installed
    assert which('mr_transform'), "Cannot find mr_transform. Is it installed?"

    # Create a temporary directory to hold the image and its transform
    tmpdir = tempfile.mkdtemp()
    saved_umask = os.umask(0o077)
    image_path = os.path.join(tmpdir, 'image.fits')
    mr_path = os.path.join(tmpdir, 'image.mr')
    if verbose:
        print("\nCreating {}".format(image_path))
        print("         {}".format(mr_path))

    # Call mr_transform on the saved image
    try:
        if verbose:
            print("Writing image to fits.")
        fits.writeto(image_path, image)
        callstr = ['mr_transform', '-t', str(type), '-n', str(nscales + 1),
                   image_path, mr_path]
        if verbose:
            print("Executing " + " ".join(callstr))
        call(callstr)
        mr = fits.getdata(mr_path)
    except IOError as e:
        print("Something went wrong... trying again.")
        # print("Removing {}".format(image_path))
        os.remove(image_path)
        # print("Unmasking {}".format(tmpdir))
        os.umask(saved_umask)
        # print("Removing {}".format(tmpdir))
        os.rmdir(tmpdir)
        # print("Calling mr_transform.")
        return mr_transform(image, nscales=nscales, type=type, verbose=verbose)
    else:
        # If successful, remove file paths
        os.remove(image_path)
        os.remove(mr_path)
        if verbose:
            print("Success.")
            print("Removed {}".format(image_path))
            print("        {}".format(mr_path))
        # Remove temporary directory
        os.umask(saved_umask)
        os.rmdir(tmpdir)
        if verbose:
            print("        {}".format(tmpdir))

    if (os.path.exists(tmpdir) or os.path.exists(image_path) or
            os.path.exists(mr_path)):
        print("Warning : not all files or directories were removed in")
        print(mr_path)

    return mr
