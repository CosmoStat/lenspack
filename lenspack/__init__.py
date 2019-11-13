# -*- coding: utf-8 -*-

"""LENSPACK

Lenspack contains the following subpackages and submodules.

Subpackages
-----------
* ``geometry`` : projection of sky coordinates to a tangent plane
* ``halo`` : NFW profiles and their lensing signals
* ``image`` : filters, transforms, shear-to-convergence inversion
* ``tests`` : unit tests of lenspack's modules
* ``theory`` : prediction of lensing observables in a given cosmology

Submodules
----------
* ``peaks`` : identification and counts of local maxima in weak-lensing maps
* ``shear`` : tangential and cross components of shear
* ``stats`` : second- and higher-order statistics
* ``utils`` : general utility functions

"""

# List of submodules
__all__ = ['geometry', 'halo', 'image', 'theory', 'peaks', 'shear', 'stats',
           'utils']

from . import *
from .info import __version__, __about__
