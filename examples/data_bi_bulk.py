from __future__ import division
import numpy as np


def get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, coordinate_of_high_symmetry_point):
    cartesian_coordinate_in_reciprocal_space = np.squeeze(np.asarray(np.matrix(coordinate_of_high_symmetry_point) *
                                                                     reciprocal_lattice_vectors)).tolist()
    return cartesian_coordinate_in_reciprocal_space


# Crystal structure parameters

a = 4.5332
c = 11.7967
g = 1.3861
gamma = 0.240385652727133


# Primitive cell

primitive_cell_a = a * np.array([[(-1.0 / 2.0), (-np.sqrt(3.0) / 6.0), 0.0],
                                 [(1.0 / 2.0), (-np.sqrt(3.0) / 6.0), 0.0],
                                 [0.0, (np.sqrt(3.0) / 3.0), 0.0]])

primitive_cell_c = c * np.array([[0.0, 0.0, (1.0 / 3.0)],
                                 [0.0, 0.0, (1.0 / 3.0)],
                                 [0.0, 0.0, (1.0 / 3.0)]])

primitive_cell = primitive_cell_a + primitive_cell_c

reciprocal_lattice_vectors_bi = g * np.matrix([[-1.0, (-np.sqrt(3.0) / 3.0), (a / c)],
                                                  [1.0, (-np.sqrt(3.0) / 3.0), (a / c)],
                                                  [0.0, (2.0 * np.sqrt(3.0) / 3.0), (a / c)]])

# High symmetry points

SPECIAL_K_POINTS_BI = {
    'LAMBDA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi, [0.25, 0.25, 0.25]),
    'GAMMA': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,  [0.00, 0.00, 0.00]),
    'T': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.50, 0.50, 0.50]),
    'L': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.00, 0.50, 0.00]),
    'X': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.50, 0.50, 0.00]),
    'K': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [((0.50 * gamma) + 0.25), (0.75 -
                                                                                          (0.50 * gamma)), 0.00]),
    'W': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [0.50, (1.00 - gamma), gamma]),
    'U': get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors_bi,      [((0.50 * gamma) + 0.25), (1.00 -
                                                                                           gamma), ((0.50 * gamma) + 0.25)])
}


# Tight binding parameters for first, second, and third nearest neighbours

PARAMS_BI_BI1 = {'ss_sigma': -0.608,
                 'sp_sigma':  1.320,
                 'pp_sigma':  1.854,
                 'pp_np.pi': -0.600}

PARAMS_BI_BI2 = {'ss_sigma': -0.384,
                 'sp_sigma':  0.433,
                 'pp_sigma':  1.396,
                 'pp_np.pi': -0.344}

PARAMS_BI_BI3 = {'ss_sigma': 0.000,
                 'sp_sigma': 0.000,
                 'pp_sigma': 0.156,
                 'pp_np.pi': 0.000}