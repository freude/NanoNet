import matplotlib.pyplot as plt
import numpy as np
from nanonet.tb import get_k_coords
import nanonet.tb as tb

# --------------------- vectors (in Angstroms) --------------------

lat_const = 1.42
lat_const_rec = 2 * np.pi / (3 * np.sqrt(3) * lat_const)

# a1 = 0.5 * lat_const * 3
# a2 = 0.5 * lat_const * np.sqrt(3)

# period = np.array([[a1, a2, 0.0],
#                    [a1, -a2, 0.0]])
#
# coords = """2
# Graphene
# C1   0.00   0.00   0.00
# C2   {}   0.00   0.00
# """.format(lat_const)
#
# special_k_points = {
#     'GAMMA': [0, 0, 0],
#     'K': [lat_const_rec * np.sqrt(3), lat_const_rec, 0],
#     'K_prime': [lat_const_rec * np.sqrt(3), -lat_const_rec, 0],
#     'M': [lat_const_rec * np.sqrt(3), 0, 0]
# }

# --------------------- vectors rotated (in Angstroms) --------------------

lat_const = 1.42
a1 = np.sqrt(3) * lat_const
a2 = 0.5 * a1
a3 = 0.5 * np.sqrt(3) * a1

period = np.array([[a1, 0.0, 0.0],
                   [a2,  a3, 0.0]])

vec = (1/3) * period[0,:] + (1/3) * period[1,:]

coords = """4
Graphene
A   0.00   0.00   0.00
A     {}     {}   0.00
B     {}     {}   3.5
B   0.00     {}   3.5
""".format(vec[0], vec[1], vec[0], vec[1], lat_const)

special_k_points = {
    'GAMMA': np.array([0, 0, 0]) * lat_const_rec ,
    'K': np.array([2, 0, 0]) * lat_const_rec ,
    'K_prime': np.array([1, np.sqrt(3), 0]) * lat_const_rec ,
    'M': np.array([3/2, np.sqrt(3)/2, 0]) * lat_const_rec
}

sym_points = ['GAMMA', 'M', 'K', 'GAMMA']
num_points = [100, 150, 150]
k_points = get_k_coords(sym_points, num_points, special_k_points)

# --------------------------- Basis set --------------------------

s_orb = tb.Orbitals('A')
s_orb.add_orbital("pz", energy=-0.28+0.5, orbital=1, magnetic=0, spin=0)

s_orb = tb.Orbitals('B')
s_orb.add_orbital("pz", energy=-0.28-0.5, orbital=1, magnetic=0, spin=0)

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
                 PARAMS_A_A5={'pp_pi': 0.1226},
                 PARAMS_A_A7={'pp_pi': 0.0898},
                 )

tb.set_tb_params(PARAMS_B_B1={'pp_pi': -3.4013},
                 PARAMS_B_B2={'pp_pi': 0.3292},
                 PARAMS_B_B3={'pp_pi': -0.2411},
                 PARAMS_B_B5={'pp_pi': 0.1226},
                 PARAMS_B_B7={'pp_pi': 0.0898}
                 )

tb.set_tb_params(PARAMS_A_B4={'pp_sigma': 0.3963},
                 PARAMS_A_B6={'pp_sigma': 0.1671},
                 PARAMS_A_B8={'pp_sigma': 0}
                 )


# --------------------------- Hamiltonian -------------------------

h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.0, 3.6, 3.76, 4.0, 4.27, 4.3])
h.initialize()
h.set_periodic_bc(period)

band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

for jj, item in enumerate(k_points):
    band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

plt.figure(1)
ax = plt.axes()
ax.set_ylabel('Energy (eV)')
ax.plot(np.sort(band_structure), 'k')
ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
ax.xaxis.grid()
plt.show()
