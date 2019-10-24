# -*- coding: utf-8 -*-

"""UNIT TESTS FOR GEOMETRY

This module contains unit tests for the geometry module.

"""

from unittest import TestCase
import numpy.testing as npt
from lenspack.geometry.projections import gnom
from lenspack.geometry.measures import angular_distance, solid_angle


class MeasuresTestCase(TestCase):

    def setUp(self):

        self.ra1, self.dec1, self.ra2, self.dec2 = [167.5, -32.3, 214.9, 0.07]
        self.ra1_list = [0, 14]
        self.dec1_list = [10, 5]
        self.ra2_list = [10, 40]
        self.dec2_list = [10, 27.3]
        self.extent = [135, 225, -45, 45]

    def tearDown(self):

        self.ra1_list = None
        self.dec1_list = None
        self.ra2_list = None
        self.dec2_list = None
        self.extent = None

    def test_angular_distance(self):

        # Test a single pair of points
        npt.assert_equal(angular_distance(self.ra1, self.dec1,
                                          self.ra2, self.dec2),
                         55.146214166755215,
                         err_msg="Incorrect angular distance.")

        # Test multiple pairs of points
        npt.assert_equal(angular_distance(self.ra1_list, self.dec1_list,
                                          self.ra2_list, self.dec2_list),
                         [9.847699509621588, 33.31969521055391],
                         err_msg="Incorrect angular distances.")

        # Test exceptions
        npt.assert_raises(Exception, angular_distance, self.ra1_list,
                          self.dec1, self.ra2_list, self.dec2_list)

    def test_solid_angle(self):

        npt.assert_equal(solid_angle(self.extent), 7292.56216087256,
                         err_msg="Incorrect solid angle.")

        npt.assert_raises(Exception, solid_angle, self.extent[:3])


class ProjectionsTestCase(TestCase):

    def setUp(self):

        self.ra0 = 10.0
        self.dec0 = 35.0
        self.proj = projections.gnom.projector(self.ra0, self.dec0)

    def tearDown(self):

        self.ra0 = None
        self.dec0 = None
        self.proj = None

    def test_projector(self):

        npt.assert_array_equal(self.proj.radec2xy(self.ra0, self.dec0),
                               [0, 0], err_msg="Center projection failed.")

        npt.assert_array_equal(self.proj.xy2radec(0, 0), [self.ra0, self.dec0],
                               err_msg="Center de-projection failed.")
