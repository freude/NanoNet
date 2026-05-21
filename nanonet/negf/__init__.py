from __future__ import absolute_import
from importlib.metadata import version

__version__ = version('nano-net')
__author__ = "Mike Klymenko"
__email__ = "mike.klymenko@rmit.edu.au"

from .greens_functions import surface_greens_function
from .recursive_greens_functions import recursive_gf
