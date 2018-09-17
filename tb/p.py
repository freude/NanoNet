"""
This module contains the input parameters for performing calculations of band structure using empirical tight binding theory.
"""
from __future__ import division
from past.utils import old_div
import numpy as np


# Mathematical constants
PI = 3.141592653589793238462643383279502884197169399375105820974


# Lattice constants
a_si = 5.50
a_bi = 4.5332
c_bi = 11.7967
gamma_bi = 0.2303


# Labels for orbitals and bonds
ORBITAL_QN = {0: 's', 1: 'p', 2: 'd'}
M_QN = {0: 'sigma', 1: 'pi', 2: 'delta'}


# Coordinates of points with high symmetry in the first Brillouin zone

SPECIAL_K_POINTS_SI = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * PI / a_si, 0],
    'L': [old_div(PI, a_si),  old_div(PI, a_si),  old_div(PI, a_si)],
    'W': [old_div(PI, a_si),  2 * PI / a_si,  0],
    'U': [old_div(PI, (2 * a_si)), 2 * PI / a_si, old_div(PI, (2 * a_si))],
    'K': [3 * PI / (2 * a_si), 3 * PI / (2 * a_si), 0]
}

reciprocal_lattice_vectors = (2 * PI / a_bi) * np.matrix([[-1, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
                                                          [1, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
                                                          [0, (2 * np.sqrt(3.0) / 3.0), (a_bi / c_bi)]])

SPECIAL_K_POINTS_BI = {
    'GAMMA': np.squeeze(np.asarray([0.00, 0.00, 0.00] * reciprocal_lattice_vectors)).tolist(),
    'T': np.squeeze(np.asarray([0.50, 0.50, 0.50] * reciprocal_lattice_vectors)).tolist(),
    'L': np.squeeze(np.asarray([0.00, 0.50, 0.00] * reciprocal_lattice_vectors)).tolist(),
    'X': np.squeeze(np.asarray([0.50, 0.50, 0.00] * reciprocal_lattice_vectors)).tolist(),
    'LAMBDA': np.squeeze(np.asarray([0.25, 0.25, 0.25] * reciprocal_lattice_vectors)).tolist(),
    'K': np.squeeze(np.asarray([((0.50 * gamma_bi) + 0.25), (0.75 - (0.50 * gamma_bi)), 0.00] *
                               reciprocal_lattice_vectors)).tolist(),
    'W': np.squeeze(np.asarray([0.50, (1.00 - gamma_bi), gamma_bi] * reciprocal_lattice_vectors)).tolist(),
    'U': np.squeeze(np.asarray([((0.50 * gamma_bi) + 0.25), (1.00 - gamma_bi), (0.50 * gamma_bi + 0.25)] *
                               reciprocal_lattice_vectors)).tolist()
}
