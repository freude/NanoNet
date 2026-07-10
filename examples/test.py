import matplotlib.pyplot as plt
import numpy as np
from nanonet.tb import get_k_coords
import nanonet.tb as tb
from nanonet.verbosity import set_verbosity
from nanonet.config import rank
from pprint import pprint


set_verbosity(0)


def make_hamiltonian(pot, field):

    lat_const = 1.42

    # --------------------- vectors rotated (in Angstroms) --------------------

    a1 = np.sqrt(3) * lat_const
    a3 = 0.5 * np.sqrt(3) * a1

    period = np.array([[a1, 0, 0],
                       [0, 2 * a3, 0]])

    coords = """8
    Graphene
    A   0.0   0.0   0.0
    A   0.0   1.42  0.0
    A   1.23  2.13  0.0
    A   1.23  3.55  0.0
    B   0.0   0.0   3.5
    B   0.0   2.84  3.5
    B   1.23  0.71  3.5
    B   1.23  2.13  3.5
    """

    # --------------------------- Basis set --------------------------

    # field = 0.5

    s_orb = tb.Orbitals('A')
    s_orb.add_orbital("pz", energy=-0.28 + field + pot, orbital=1, magnetic=0, spin=0)

    s_orb = tb.Orbitals('B')
    s_orb.add_orbital("pz", energy=-0.28 - field + pot, orbital=1, magnetic=0, spin=0)

    s_orb = tb.Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    # ------------------------ set TB parameters----------------------
    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    tb.set_tb_params(PARAMS_A_A1={'pp_pi': -3.4013},
                     PARAMS_A_A2={'pp_pi': 0.3292},
                     PARAMS_A_A3={'pp_pi': -0.2411},
                     PARAMS_A_A5={'pp_pi': 0 * 0.1226},
                     PARAMS_A_A7={'pp_pi': 0 * 0.0898},
                     )

    tb.set_tb_params(PARAMS_B_B1={'pp_pi': -3.4013},
                     PARAMS_B_B2={'pp_pi': 0.3292},
                     PARAMS_B_B3={'pp_pi': -0.2411},
                     PARAMS_B_B5={'pp_pi': 0 * 0.1226},
                     PARAMS_B_B7={'pp_pi': 0 * 0.0898}
                     )

    tb.set_tb_params(PARAMS_A_B4={'pp_sigma': 0.3963},
                     PARAMS_A_B6={'pp_sigma': 0.1671},
                     PARAMS_A_B8={'pp_sigma': 0}
                     )

    # --------------------------- Hamiltonian -------------------------

    h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.0, 3.6, 3.76, 4.0, 4.27, 4.3])
    h.initialize()
    h.set_periodic_bc(period)

    return h

def main():

    h = make_hamiltonian(0.0, 0.0)
    a, _ = h.diagonalize_periodic_bc(np.array([1.2773235, 0.30100533, 0.0]))

    print(a)
    print('---------------')
    pprint(h.h_matrix_bc_factor * h.h_matrix + h.h_matrix_bc_add)

if __name__ == "__main__":
    main()



