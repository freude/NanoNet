from __future__ import division
import numpy as np


def get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, coordinate_of_high_symmetry_point):
    cartesian_coordinate_in_reciprocal_space = np.squeeze(np.asarray(np.matrix(coordinate_of_high_symmetry_point) *
                                                                     reciprocal_lattice_vectors)).tolist()
    return cartesian_coordinate_in_reciprocal_space


# Primitive cell

primitive_cell = 4.5332 * np.array([[ 1.000000000000000, 0.000000000000000, 0.000000000000000],
                                    [-0.500000000000000, 0.866025403784439, 0.000000000000000],
                                    [ 0.000000000000000, 0.000000000000000, 6.929958905343691]])

reciprocal_lattice_vectors = 2 * np.pi * np.asmatrix(np.concatenate(([np.cross(primitive_cell[1], primitive_cell[2]) / np.dot(primitive_cell[0], np.cross(primitive_cell[1], primitive_cell[2]))],
                                                                     [np.cross(primitive_cell[2], primitive_cell[0]) / np.dot(primitive_cell[1], np.cross(primitive_cell[2], primitive_cell[0]))],
                                                                     [np.cross(primitive_cell[0], primitive_cell[1]) / np.dot(primitive_cell[2], np.cross(primitive_cell[0], primitive_cell[1]))]), axis=0))


# High symmetry points

SPECIAL_K_POINTS_BI = {
    'GAMMA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, [0.000000000000000, 0.000000000000000, 0.000000000000000]),
    'K':     get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, [0.333333333333333, 0.333333333333333, 0.000000000000000]),
    'M':     get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, [0.500000000000000, 0.000000000000000, 0.000000000000000])
}


# Off-site parameters for first, second, and third nearest neighbours

PARAMS_BI_BI1 = {'ss_sigma': -0.608,
                 'sp_sigma':  1.320,
                 'pp_sigma':  1.854,
                 'pp_pi':    -0.600}

PARAMS_BI_BI2 = {'ss_sigma': -0.384,
                 'sp_sigma':  0.433,
                 'pp_sigma':  1.396,
                 'pp_pi':    -0.344}

PARAMS_BI_BI3 = {'ss_sigma': 0.000,
                 'sp_sigma': 0.000,
                 'pp_sigma': 0.156,
                 'pp_pi':    0.000}