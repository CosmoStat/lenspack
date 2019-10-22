# -*- coding: utf-8 -*-

"""PACKAGE INFO

This module provides some basic information about the package.

"""

# Set the package release version
version_info = (0, 0, 0)
__version__ = '.'.join(str(c) for c in version_info)

# Set the package details
__author__ = 'configure_author'
__email__ = 'configure_email'
__year__ = 'configure_year'
__url__ = 'configure_url'
__description__ = 'configure_description'
__requires__ = []  # Your package dependencies

# Default package properties
__license__ = 'MIT'
__about__ = ('{} \n\n Author: {} \n Email: {} \n Year: {} \n {} \n\n'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))
__setup_requires__ = ['pytest-runner', ]
__tests_require__ = ['pytest', 'pytest-cov', 'pytest-pep8']
