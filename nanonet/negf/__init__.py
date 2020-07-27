from __future__ import absolute_import
from pkg_resources import get_distribution

__version__ = get_distribution('nano-net').version
__author__ = "Mike Klymenko"
__email__ = "mike.klymenko@rmit.edu.au"

from .greens_functions import surface_greens_function
from .recursive_greens_functions import recursive_gf