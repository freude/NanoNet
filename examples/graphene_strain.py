""" This script is to show how to work with the argument `radial_dep` of
the member-function Hamiltonian.initialize().

In the example below, we shown how this feature can be used to model
the effect of the strain in graphene on its band structure.
"""
import matplotlib.pyplot as plt
import numpy as np
from nanonet.tb import get_k_coords
import nanonet.tb as tb


def make_coordinates(strain=0, angle=0, poisson=0.165):
    rot_mat90 = np.array([[np.cos(0.5 * np.pi), -np.sin(0.5 * np.pi)],
                          [np.sin(0.5 * np.pi), np.cos(0.5 * np.pi)]])

    theta = angle               # force angle
    sigma = poisson             # Poisson ratio
    eps0 = strain               # strain

    # strain_tensor
    eps_11 = np.cos(theta) ** 2 - sigma * np.sin(theta) ** 2
    eps_12 = (1.0 + sigma) * np.sin(theta) * np.cos(theta)
    eps_21 = (1.0 + sigma) * np.sin(theta) * np.cos(theta)
    eps_22 = np.sin(theta) ** 2 - sigma * np.cos(theta) ** 2
    eps_11 *= eps0
    eps_12 *= eps0
    eps_21 *= eps0
    eps_22 *= eps0

    eps = np.array([[eps_11, eps_12],
                    [eps_21, eps_22]]) + np.eye(2)

    # recompute lattice parameters and inter-atomic spacing after applying the strain
    lat_const = 1.42
    a1 = 0.5 * lat_const * 3
    a2 = 0.5 * lat_const * np.sqrt(3)

    vec1 = np.array([a1, a2])
    vec2 = np.array([a1, -a2])
    vec1 = eps.dot(vec1)
    vec2 = eps.dot(vec2)

    period = np.array([[vec1[0], vec1[1], 0.0],
                       [vec2[0], vec2[1], 0.0]])

    atomic_coords = np.array([lat_const, 0.0])
    atomic_coords = eps.dot(atomic_coords)

    coords = """2
    Graphene
    C1   0.00   0.00   0.00
    C2   {}     {}     0.00
    """.format(*atomic_coords)

    # reciprocal space
    b1 = 2 * np.pi * rot_mat90.dot(vec2) / (vec1.dot(rot_mat90.dot(vec2)))
    b2 = 2 * np.pi * rot_mat90.dot(vec1) / (vec2.dot(rot_mat90.dot(vec1)))

    M1 = np.linalg.pinv(np.array([[b1[0], b1[1]],
                                  [b1[0] + b2[0], b1[1] + b2[1]]]))

    M2 = np.linalg.pinv(np.array([[b2[0], b2[1]],
                                  [b1[0] + b2[0], b1[1] + b2[1]]]))

    C1 = 0.5 * np.array([np.linalg.norm(b1)**2, np.linalg.norm(b1 + b2)**2])
    C2 = 0.5 * np.array([np.linalg.norm(b2)**2, np.linalg.norm(b1 + b2)**2])

    k_point = M1.dot(C1)
    k_point1 = M2.dot(C2)
    m_point = 0.5 * (b1 + b2)

    special_k_points = {
        'GAMMA': [0, 0, 0],
        'K': [k_point[0], k_point[1], 0],
        'K_prime': [k_point1[0], k_point1[1], 0],
        'M': [m_point[0], m_point[1], 0]
    }

    sym_points = ['GAMMA', 'M', 'K', 'GAMMA']
    num_points = [25, 25, 25]
    k_points = get_k_coords(sym_points, num_points, special_k_points)

    return coords, period, k_points, num_points, sym_points


def main():

    # this function redefines position of atoms and unit cell geomentry
    # taking into account the strain
    coords, period, k_points, num_points, sym_points = make_coordinates(strain=0.5)

    # below we define our tight-binding model for graphene
    # --------------------------- Basis set --------------------------
    s_orb = tb.Orbitals('C')
    s_orb.add_orbital("pz", energy=0, orbital=1, magnetic=0, spin=0)

    # ------------------------ set TB parameters----------------------
    t = 2.8
    tb.set_tb_params(PARAMS_C_C={'pp_pi': t})

    # --------------------------- Hamiltonian -------------------------
    h = tb.Hamiltonian(xyz=coords, nn_distance=2.5)

    # this is a function that determines the distance dependence of the tight-binding parameters.
    # In the current version it takes just one argument being the inter-atomic distance and returns the factor
    # that will be multiplied to the tight-binding parameters.

    # This function is accepted as an argument in the member-function Hamiltonian.initialize() and
    # it is evaluated inside the function Hamiltonian()._get_me() (line 435).
    # Since this feature is in a developing stage and the distance is only one acceptable argument,
    # you may be interested in modifying the line 435 in _get_me() in order to make it work with more arguments.
    # This is the place where you can change the signature of the supplied function adding more arguments;
    # for instance if you want it to process not only the distance
    # but the orbital symmetry type as well.

    def my_func(dist):
        print(dist)
        a0 = 1.42
        return np.exp(-3.37*(dist/a0-1))

    # The argument `radial_dep` is the name of the function defined above.
    h.initialize(radial_dep=my_func)
    h.set_periodic_bc(period)

    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

    # visualize
    plt.figure()
    ax = plt.axes()
    ax.set_title(r'Band structure of graphene, 1st NN')
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(band_structure), 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


if __name__ == '__main__':

    main()

