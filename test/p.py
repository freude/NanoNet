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


# Labels for orbitals and bonds
ORBITAL_QN = {0: 's', 1: 'p', 2: 'd'}
M_QN = {0: 'sigma', 1: 'pi', 2: 'delta'}

# Spin-orbit coupling term for bismuth
LAMBDA = 0.3
# LAMBDA = 0.25
# LAMBDA = 0.2167
# LAMBDA = 0.1333
# LAMBDA = 0

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


# reciprocal_lattice_vectors_bi = g_bi * np.matrix([[-1.0, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
#                                                   [1.0, (-np.sqrt(3.0) / 3.0), (a_bi / c_bi)],
#                                                   [0.0, (2.0 * np.sqrt(3.0) / 3.0), (a_bi / c_bi)]])
#
# SPECIAL_K_POINTS_BI = {
#     'LAMBDA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.25, 0.25, 0.25]),
#     'GAMMA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.00, 0.00, 0.00]),
#     'T': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, 0.50, 0.50]),
#     'L': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.00, 0.50, 0.00]),
#     'X': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, 0.50, 0.00]),
#     'K': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [((0.50 * gamma_bi) + 0.25), (0.75 -
#                                                                                     (0.50 * gamma_bi)), 0.00]),
#     'W': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.50, (1.00 - gamma_bi), gamma_bi]),
#     'U': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [((0.50 * gamma_bi) + 0.25), (1.00 -
#                                                                                     gamma_bi), ((0.50 * gamma_bi) +
#                                                                                     0.25)])
# }


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