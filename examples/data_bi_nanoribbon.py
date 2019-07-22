from __future__ import division
import numpy as np


def get_high_symmetry_point_in_cartesian_space(reciprocal_lattice_vectors, coordinate_of_high_symmetry_point):
    cartesian_coordinate_in_reciprocal_space = np.squeeze(np.asarray(np.matrix(coordinate_of_high_symmetry_point) *
                                                                     reciprocal_lattice_vectors)).tolist()
    return cartesian_coordinate_in_reciprocal_space


# Primitive cell

lattice_constant = 4.5332


# High symmetry points

SPECIAL_K_POINTS_BI_NANORIBBON = {
    'GAMMA': (2 * np.pi / lattice_constant) * np.array([0.0, 0.0, 0.0]),
    'X':     (2 * np.pi / lattice_constant) * np.array([1.0, 0.0, 0.0])
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