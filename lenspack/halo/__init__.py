# -*- coding: utf-8 -*-

"""HALO MODULE

This module contains codes for describing the radial mass density profile of
dark matter halos and for predicting their weak-lensing signal. Currently,
the only profile implemented is the Navarro, Frenk, & White (NFW, 1997) model,
but others like the singular isothermal sphere (SIS), Einasto, and Baltz,
Marshall, & Oguri (BMO) models should be added in the future.

"""

__all__ = ['fitting', 'profiles']

from . import *
