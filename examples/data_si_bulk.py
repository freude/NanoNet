import numpy as np


# Crystal structure parameters

a = 5.50


# Primitive cell

primitive_cell = a * np.array([[0.0, 0.5, 0.5],
                               [0.5, 0.0, 0.5],
                               [0.5, 0.5, 0.0]])


# High symmetry points

SPECIAL_K_POINTS_SI = {
    'GAMMA': [0, 0, 0],
    'X': [0, 2 * np.pi / a, 0],
    'L': [np.pi / a,  np.pi / a,  np.pi / a],
    'W': [np.pi / a,  2 * np.pi / a,  0],
    'U': [np.pi / (2 * a), 2 * np.pi / a, np.pi/ (2 * a)],
    'K': [3 * np.pi / (2 * a), 3 * np.pi / (2 * a), 0]
}