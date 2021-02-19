# -*- coding: utf-8 -*-

"""UNIT TESTS FOR IMAGE

This module contains unit tests for the image module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from lenspack.image.transforms import starlet2d, dct2d, idct2d


class TransformsTestCase(TestCase):

    def setUp(self):

        self.nscales = 5
        self.npix = 64
        self.image = img = 10 * np.random.normal(size=(self.npix, self.npix))
        spike = np.zeros_like(self.image)
        spike[self.npix // 2, self.npix // 2] = 1
        self.spike = spike

    def tearDown(self):

        self.nscales = None
        self.npix = None
        self.image = None
        self.spike = None

    def test_starlet2d(self):

        # Test output shape of starlet transform
        wt = starlet2d(self.image, self.nscales)
        output_shape = (self.nscales + 1, self.npix, self.npix)
        npt.assert_equal(output_shape, wt.shape,
                         err_msg="Incorrect starlet2d output shape.")

        # Test reconstruction
        rec = np.sum(wt, axis=0)
        npt.assert_allclose(rec, self.image,
                            err_msg="Incorrect starlet reconstruction.")

        # Test wavelet filter norms
        wt_spike = starlet2d(self.spike, self.nscales)
        norms = np.sqrt(np.sum(wt_spike[:-1]**2, axis=(1, 2)))
        expected = [0.890796310279, 0.2006638510244, 0.0855075047534]
        if len(norms > 2):
            npt.assert_allclose(norms[:3], expected,
                                err_msg="Incorrect filter norms.")

    def test_dct2d(self):

        # Test reconstruction
        dct = dct2d(self.image)
        rec = idct2d(dct)
        npt.assert_allclose(rec, self.image,
                            err_msg="Incorrect DCT reconstruction.")

        # Test exceptions
        npt.assert_raises(Exception, dct2d, self.image[0])
        npt.assert_raises(Exception, dct2d, self.image, 'symmetric')
        npt.assert_raises(Exception, idct2d, self.image[0])
        npt.assert_raises(Exception, idct2d, self.image, 'symmetric')
