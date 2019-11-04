# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import convolve
from scipy.signal import fftconvolve

from lenspack.utils import radius2d, round_up_to_odd


FILTER_OPTIONS = {'s98', 'vw98', 'jbj04', 'starlet', 'gauss'}
METHOD_OPTIONS = {'direct', 'fourier', 'brute'}
BORDER_OPTIONS = {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}


def _validate_option(value, options):
    """Check that an option is one of an allowed set."""
    if value not in options:
        raise ValueError("{} is not in the set {}.".format(value, options))
    return value


def aperture_mass(image, theta, filter='s98', method='fourier',
                  border='constant'):
    """Filter an image (convergence map) using an aperture mass kernel.

    Parameters
    ----------
    image : array_like
        2D convergence map.
    theta : float
        Aperture radial size in pixels.
    filter : {'s98', 'vw98', 'jbj04', 'starlet'}
        Radial filter function. Default is 's98'.
    method : str, optional
        Computation technique. Options are
        (1) 'direct' for angular-space convolution using
            `scipy.ndimage.convolve` with an approximate (truncated) kernel
        (2) 'fourier' for fourier-space convolution using
            `scipy.signal.fftconvolve`
        (3) 'brute' for brute force pixel-by-pixel computation in direct
            space. Gives exactly the same result as (1) when 'constant'
            is used in convolve function.
        Default is 'fourier'.
    border : str, optional
        If `method` is 'direct', `border` determines how the border is treated
        via the scipy.ndimage.convolve option. Options are
        {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}. Defaults is
        'constant' with zero-filling.

    Returns
    -------
    aperture_mass : 2D numpy array
        Aperture mass map, i.e. a filtered version of `image`.

    Raises
    ------
    ValueError
        For invalid input options.

    Notes
    -----
    The different methods have vastly different computation times.
    Fourier-space convolution is by far the fastest and the only usable
    option for large images and large apertures. It gives very nearly the same
    results as direct-space convolution when the convolution `border` is
    'constant' with zero-padding. The latter also gives the same result as
    the brute-force approach.

    When using the starlet filter, the result is closest to the mr_transform
    (ISAP binary) output when the border is 'constant'.

    Examples
    --------
    ...

    """
    # Validate input options
    filter = _validate_option(filter, FILTER_OPTIONS)
    method = _validate_option(method, METHOD_OPTIONS)
    border = _validate_option(border, BORDER_OPTIONS)

    # Direct space convolution
    if method == 'direct':
        kernel = gen_apmass_kernel(theta, filter)
        assert len(kernel) > len(image), "Kernel is too big."
        apmass = convolve(image, kernel, mode=border)
    # Fourier space convolution
    elif method == 'fourier':
        # kernel = gen_apmass_kernel(theta, filter, size=len(image))
        # fkernel = np.fft.fft2(kernel)
        # fimage = np.fft.fft2(image[:len(kernel),:len(kernel)])
        # fapmass = fkernel * fimage
        # apmass = np.fft.fftshift(np.fft.ifft2(fapmass).real)
        kernel = gen_apmass_kernel(theta, filter)
        apmass = fftconvolve(image, kernel, mode='same')
    # Brute force convolution
    else:
        apmass = np.zeros_like(image)
        npix = len(image)
        for i in range(npix):
            for j in range(npix):
                x = abs(np.arange(npix) - j)
                y = abs(np.arange(npix) - i)
                X, Y = np.meshgrid(x**2, y**2)
                radii = np.sqrt(X + Y)
                apmass[i, j] = (u_function(radii, theta, filter) * image).sum()
            if i % 10 == 0:
                print("row {}".format(i))

    return apmass


def gen_apmass_kernel(theta, filter, size=None):
    """Generate a convolution kernel for aperture mass filtering.

    Parameters
    ----------
    theta : float
        Aperture radial size in pixels.
    filter : {'s98', 'vw98', 'jbj04', 'starlet'}
        Radial filter function.
    size : int, optional
        Side length N of the output (N, N) kernel matrix. If not given, the
        output shape is determined by the minimum radius at which the filter
        function becomes effectively negligible.

    Returns
    -------
    2D numpy array
        Aperture mass kernel matrix of shape (N, N). N is always odd when
        `size` is not specified.

    """
    filter = _validate_option(filter, FILTER_OPTIONS)

    if size is None:
        # Evaluate out to twice the aperture size, unless using S98 or JBJ04
        opts = {'s98': 1.1, 'vw98': 2, 'jbj04': 4, 'starlet': 2, 'gauss': 4}
        k = opts[filter]
        N = round_up_to_odd(2 * k * theta)
        if N > 2:
            radii = radius2d(N, mode='fft')
            kernel = u_function(radii, xs=theta, filter=filter)
        else:
            # Set as identity kernel if aperture is too small
            kernel = np.zeros((3, 3))
            kernel[1, 1] = 1.
    else:
        radii = radius2d(size, mode='exact')
        kernel = u_function(radii, xs=theta, filter=filter)

    return kernel


def u_function(x, xs, filter, l=1):
    """Aperture mass isotropic filter function U(theta).

    Parameters
    ----------
    x : array_like
        Radial distances.
    xs : float
        Scale radius.
    filter : str
        Functional form - one of {'s98', 'vw98', 'jbj04', 'starlet'}
        from the following references :
        s98 - Schneider, van Waerbeke, Jain, Kruse (1998), MNRAS 296, 873
        vw98 - Van Waerbeke (1998), A&A 334, 1
        jbj04 - Jarvis, Bernstein, Jain (2004), MNRAS 352, 338
        starlet - Leonard, Pires, Starck (2012), MNRAS 423, 3045
        * See also - Zhang, Pen, Zhang et al. (2003), ApJ 598, 818,
                     Schirmer et al. (2007), A&A 462, 875.
    l : int, optional
        Parameter of the 's98' filter. Default is 1.

    Returns
    -------
    numpy array
        Function value at each `x`.

    Notes
    -----
    For the starlet, a factor of 1 / xs**2 was added to the normalization
    in order to give consistent results with mr_transform (ISAP binary).

    """
    filter = _validate_option(filter, FILTER_OPTIONS)
    x = np.atleast_1d(x).astype(float)
    y = x / xs

    # Filter function definitions
    if filter == 's98':
        # Factor to match Giocoli et al. (2015) normalization
        prefactor = np.sqrt(276) / 24
        A = (l + 2) / np.pi / xs**2
        result = A * np.power(1. - y**2, l) * (1. - (l + 2.) * y**2)
        result = prefactor * result * np.heaviside(xs - np.abs(x), 0.5)
    elif filter == 'vw98':
        A = 1. / xs**2
        b = 4.
        result = A * (1. - b * y**2) * np.exp(-b * y**2)
    elif filter == 'jbj04':
        A = 1. / (2. * np.pi) / xs**2
        b = 0.5
        result = A * (1. - b * y**2) * np.exp(-b * y**2)
    elif filter == 'starlet':
        A = 1. / 9 / xs**2
        term1 = 93 * abs(y)**3 - 64 * (abs(0.5 - y)**3 + abs(0.5 + y)**3)
        term2 = 18 * (abs(1 - y)**3 + abs(1 + y)**3)
        term3 = 0.5 * (abs(2 - y)**3 + abs(2 + y)**3)
        result = A * (term1 + term2 - term3)
    elif filter == 'gauss':
        # Secret option, since it's not really an aperture mass filter but is
        # useful in debugging
        result = np.exp(-0.5 * y**2) / (2 * np.pi * xs**2)

    return result
