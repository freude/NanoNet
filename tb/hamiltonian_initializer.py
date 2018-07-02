import sys
import numpy as np
from atoms import Atom
import diatomic_matrix_element as dme
from hamiltonian import Hamiltonian
from hamiltonian_sparse import HamiltonianSp


def set_tb_params(**kwargs):
    for item in kwargs:
        if item.startswith('PARAMS_'):
            setattr(dme, item, kwargs[item])


def initializer(**kwargs):

    set_tb_params(**kwargs)
    Atom.orbital_sets = kwargs.get('orbital_sets', {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'})
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
