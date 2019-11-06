# -*- coding: utf-8 -*-

"""PACKAGE

Provide some basic description of what your package module contains.

"""

# List of submodules
__all__ = ['geometry', 'halo', 'image', 'theory', 'peaks', 'shear', 'stats',
           'utils']

from . import *
from .info import __version__, __about__
