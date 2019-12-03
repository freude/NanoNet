from __future__ import absolute_import
from pkg_resources import get_distribution

__version__ = get_distribution('nanonet').version
__author__ = "Mike Klymenko"
__email__ = "mike.klymenko@rmit.edu.au"

from .greens_functions import surface_greens_function