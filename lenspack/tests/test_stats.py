# -*- coding: utf-8 -*-

"""UNIT TESTS FOR IMAGE

This module contains unit tests for the iamge module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import stats
from lenspack.stats import mad, skew, kurt, mu_n, kappa_n, fdr, hc


class StatsTestCase(TestCase):

    def setUp(self):

        # [-5., -4., -3., ... 3., 4., 5.]
        self.array = np.arange(11.) - 5

    def tearDown(self):

        self.array = None

    def test_mad(self):

        # Test output value
        npt.assert_equal(mad(self.array), 3.0, err_msg="Incorrect MAD value.")

    def test_skew(self):

        # Test output value and agreement with scipy
        npt.assert_equal(skew(self.array), 0, err_msg="Incorrect skew value.")
        npt.assert_equal(skew(self.array**2), 0.5661385170722978,
                         err_msg="Incorrect skew value.")
        npt.assert_almost_equal(skew(self.array**2), stats.skew(self.array**2),
                                decimal=15,
                                err_msg="Does not match scipy.skew.")

    def test_kurt(self):

        # Test output value and agreement with scipy
        npt.assert_almost_equal(kurt(self.array), -1.22,
                                decimal=15,
                                err_msg="Incorrect kurt value.")
        npt.assert_almost_equal(kurt(self.array), stats.kurtosis(self.array),
                                decimal=15,
                                err_msg="Does not match scipy.kurtosis.")

    def test_mu_n(self):

        # Test output value
        npt.assert_equal(mu_n(self.array, order=1), 0,
                         err_msg="Incorrect mu_n for order 1.")
        npt.assert_equal(mu_n(self.array, order=2), 10,
                         err_msg="Incorrect mu_n for order 2.")
        npt.assert_equal(mu_n(self.array, order=3), 0,
                         err_msg="Incorrect mu_n for order 3.")
        npt.assert_equal(mu_n(self.array, order=4), 178,
                         err_msg="Incorrect mu_n for order 4.")
        npt.assert_equal(mu_n(self.array, order=5), 0,
                         err_msg="Incorrect mu_n for order 5.")
        npt.assert_equal(mu_n(self.array, order=6), 3730,
                         err_msg="Incorrect mu_n for order 6.")

        # Test agreement with scipy
        npt.assert_equal(mu_n(self.array, order=1),
                         stats.moment(self.array, moment=1),
                         err_msg="Does not match scipy.moment for order 1.")
        npt.assert_equal(mu_n(self.array, order=2),
                         stats.moment(self.array, moment=2),
                         err_msg="Does not match scipy.moment for order 2.")
        npt.assert_equal(mu_n(self.array, order=3),
                         stats.moment(self.array, moment=3),
                         err_msg="Does not match scipy.moment for order 3.")
        npt.assert_equal(mu_n(self.array, order=4),
                         stats.moment(self.array, moment=4),
                         err_msg="Does not match scipy.moment for order 4.")
        npt.assert_equal(mu_n(self.array, order=5),
                         stats.moment(self.array, moment=5),
                         err_msg="Does not match scipy.moment for order 5.")
        npt.assert_equal(mu_n(self.array, order=6),
                         stats.moment(self.array, moment=6),
                         err_msg="Does not match scipy.moment for order 6.")

        # Test exceptions
        npt.assert_raises(Exception, mu_n, self.array, order=0)

    def test_kappa_n(self):

        # Test output value
        npt.assert_equal(kappa_n(self.array, order=2), 10,
                         err_msg="Incorrect mu_n for order 2.")
        npt.assert_equal(kappa_n(self.array, order=3), 0,
                         err_msg="Incorrect mu_n for order 3.")
        npt.assert_equal(kappa_n(self.array, order=4), -122,
                         err_msg="Incorrect mu_n for order 4.")
        npt.assert_equal(kappa_n(self.array, order=5), 0,
                         err_msg="Incorrect mu_n for order 5.")
        npt.assert_equal(kappa_n(self.array, order=6), 7030,
                         err_msg="Incorrect mu_n for order 6.")

        # Test exceptions
        npt.assert_raises(Exception, kappa_n, self.array, order=1)
