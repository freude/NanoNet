from __future__ import absolute_import
from importlib.metadata import version

__version__ = version('nano-net')
__author__ = "Mike Klymenko"
__email__ = "mike.klymenko@monash.edu"

from .orbitals import Orbitals
from .hamiltonian import Hamiltonian
from .hamiltonian_sparse import HamiltonianSp
from .aux_functions import get_k_coords, yaml_parser
from .hamiltonian_initializer import set_tb_params, initializer

from .reduced_mode_space import reduce_mode_space, bs_vs_e, bs
import logging
from pyfiglet import Figlet

logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.StreamHandler(stream=None)
logging.info(Figlet(font='standard').renderText('NanoNET'))
logging.info("Vesion " + __version__)
