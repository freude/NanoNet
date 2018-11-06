"""
This module contains the input parameters for performing calculations of band structure using empirical tight binding theory.
"""
from __future__ import division
import numpy as np


# Mathematical constants
PI = np.pi


# Lattice constants
a_si = 5.50
a_bi = 4.5332
c_bi = 11.7967
g_bi = 1.3861
gamma_bi = 0.240385652727133

# Coordinates of points with high symmetry in the first Brillouin zone

SPECIAL_K_POINTS_SI = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * PI / a_si, 0],
    'L': [PI / a_si,  PI / a_si,  PI/ a_si],
    'W': [PI / a_si,  2 * PI / a_si,  0],
    'U': [PI / (2 * a_si), 2 * PI / a_si, PI/ (2 * a_si)],
    'K': [3 * PI / (2 * a_si), 3 * PI / (2 * a_si), 0]
}


def get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, coordinate_of_high_symmetry_point):
    cartesian_coordinate_in_reciprocal_space = np.squeeze(np.asarray(np.matrix(coordinate_of_high_symmetry_point) *
                                                                     reciprocal_lattice_vectors)).tolist()
    return cartesian_coordinate_in_reciprocal_space


reciprocal_lattice_vectors_bi = g_bi * np.matrix([[-1.0, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
                                                  [1.0, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
                                                  [0.0, (2.0 * np.sqrt(3.0) / 3.0), (a_bi / c_bi)]])

SPECIAL_K_POINTS_BI = {
    'LAMBDA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.25, 0.25, 0.25]),
    'GAMMA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.00, 0.00, 0.00]),
    'T': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, 0.50, 0.50]),
    'L': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.00, 0.50, 0.00]),
    'X': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, 0.50, 0.00]),
    'K': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [((0.50 * gamma_bi) + 0.25), (0.75 -
                                                                                    (0.50 * gamma_bi)), 0.00]),
    'W': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, (1.00 - gamma_bi), gamma_bi]),
    'U': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [((0.50 * gamma_bi) + 0.25), (1.00 -
                                                                                    gamma_bi), ((0.50 * gamma_bi) +
                                                                                    0.25)])
}


cell = 4.32903 * np.array([[ 1.000000000000000, 0.000000000000000, 0.000000000000000],
                           [-0.500000000000000, 0.866025403784439, 0.000000000000000],
                           [ 0.000000000000000, 0.000000000000000, 6.929958905343691]])


# 1NN - Bi-Bi
PARAMS_BI_BI1 = {'ss_sigma': -0.608,
                 'sp_sigma': 1.320,
                 'pp_sigma': 1.854,
                 'pp_pi': -0.600}

# 2NN - Bi-Bi
PARAMS_BI_BI2 = {'ss_sigma': -0.384,
                 'sp_sigma': 0.433,
                 'pp_sigma': 1.396,
                 'pp_pi': -0.344}

# 3NN - Bi-Bi
PARAMS_BI_BI3 = {'ss_sigma': 0,
                 'sp_sigma': 0,
                 'pp_sigma': 0.156,
                 'pp_pi': 0}