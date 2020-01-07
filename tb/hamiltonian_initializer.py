"""
The module contains functions facilitating setting tight-binding parameters and
initializing Hamiltonian objects from a Python dictionary.
"""

from __future__ import absolute_import
import sys
import numpy as np
from .orbitals import Orbitals
from . import diatomic_matrix_element as dme
from .hamiltonian import Hamiltonian
from .hamiltonian_sparse import HamiltonianSp


def set_tb_params(**kwargs):
    """
    Initialize a set of the user-defined tight-binding parameters.

    Parameters
    ----------
    kwargs : dict of dict
        Dictionary of the tight-binding parameters.
        The dictionary follows a certain name convention - each new entry should conform
        with following format:
        for the entry names,  PARAMS_<el1>_<el2>_<order>, where <el1> and <el2> are
        chemical elements of a pair of atoms, and <order> is a number specifying the order
        of nearest neighbours;
        for the dictionary values,  <orb1><orb2>_<mol>, where <orb1> and <orb2> are
        the orbital quantum numbers (s, p, d etc.) and <mol> is the symmetry of a molecular orbital
        (sigma, pi etc).

    """
    for item in kwargs:
        if item.startswith('PARAMS_'):
            setattr(dme, item, kwargs[item])


def initializer(**kwargs):
    """
    Creates a Hamiltonian object from a set of parameters stored in a Python dictionary.

    This functions is used by CLI scripts to create Hamiltonian objects
    from a configuration file (normally in a yaml format) which is previously
    parsed into a Python dictionary data structure.

    Parameters
    ----------
    kwargs : dict
        Dictionary of parameters needed to make a Hamiltonian object.

    Returns
    -------
    h : tb.Hamiltonian
        instance of the class Hamiltonian

    """
    set_tb_params(**kwargs)
    Orbitals.orbital_sets = kwargs.get('orbital_sets', {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'})
    sys.modules[__name__].VERBOSITY = kwargs.get('VERBOSITY', 1)

    xyz = kwargs.get('xyz', {})
    nn_distance = kwargs.get('nn_distance', 2.7)
    sparse = kwargs.get('sparse', 0)
    sigma = kwargs.get('sigma', 1.1)
    num_eigs = kwargs.get('num_eigs', 14)

    if sparse:
        h = HamiltonianSp(xyz=xyz, nn_distance=nn_distance, sigma=sigma, num_eigs=num_eigs)
    else:
        h = Hamiltonian(xyz=xyz, nn_distance=nn_distance)

    h.initialize()

    primitive_cell = kwargs.get('primitive_cell', [0, 0, 0])

    if np.sum(np.abs(np.array(primitive_cell))) > 0:
        h.set_periodic_bc(primitive_cell=primitive_cell)

    return h
