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


def get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, coordinate_of_high_symmetry_point):
    cartesian_coordinate_in_reciprocal_space = np.squeeze(np.asarray(np.matrix(coordinate_of_high_symmetry_point) *
                                                                     reciprocal_lattice_vectors)).tolist()
    return cartesian_coordinate_in_reciprocal_space


cell = 4.32903 * np.array([[ 1.000000000000000, 0.000000000000000, 0.000000000000000],
                           [-0.500000000000000, 0.866025403784439, 0.000000000000000],
                           [ 0.000000000000000, 0.000000000000000, 6.929958905343691]])

reciprocal_lattice_vectors_bi_bilayer = 2 * np.pi * np.asmatrix(np.concatenate(([np.cross(cell[1], cell[2]) / np.dot(cell[0], np.cross(cell[1], cell[2]))],
                                                                                [np.cross(cell[2], cell[0]) / np.dot(cell[1], np.cross(cell[2], cell[0]))],
                                                                                [np.cross(cell[0], cell[1]) / np.dot(cell[2], np.cross(cell[0], cell[1]))]), axis=0))

cell = 4.32903 * np.array([[ 1.000000000000000, 0.000000000000000, 0.000000000000000],
                           [-0.500000000000000, 0.866025403784439, 0.000000000000000]])

SPECIAL_K_POINTS_BI = {
    'GAMMA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi_bilayer, [0.00000, 0.00000, 0.00000]),
    'K':     get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi_bilayer, [0.33333, 0.33333, 0.00000]),
    'M':     get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi_bilayer, [0.50000, 0.00000, 0.00000])
}


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