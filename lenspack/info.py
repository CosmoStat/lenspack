# -*- coding: utf-8 -*-

"""PACKAGE INFO

This module provides some basic information about the package.

"""

# Set the package release version
version_info = (1, 0, 0)
__version__ = '.'.join(str(c) for c in version_info)

# Set the package details
__author__ = 'Austin Peel'
__email__ = 'austin.peel@cea.fr'
__year__ = '2019'
__url__ = 'https://github.com/austinpeel/lenspack'
__description__ = 'A collection of weak lensing tools in Python'
__requires__ = ['numpy', 'scipy', 'astropy']  # Package dependencies

# Default package properties
__license__ = 'MIT'
__about__ = ('{} \n\n Author: {} \n Email: {} \n Year: {} \n {} \n\n'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))
__setup_requires__ = ['pytest-runner', ]
__tests_require__ = ['pytest', 'pytest-cov', 'pytest-pep8']
