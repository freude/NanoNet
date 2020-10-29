import matplotlib.pyplot as plt
import numpy as np
from nanonet.tb import get_k_coords
import nanonet.tb as tb

# --------------------- vectors (in Angstroms) --------------------

lat_const = 1.42
a1 = 0.5 * lat_const * 3
a2 = 0.5 * lat_const * np.sqrt(3)

period = np.array([[a1, a2, 0.0],
                   [a1, -a2, 0.0]])

# --------------------------- wave vectors -------------------------

lat_const_rec = 2 * np.pi / (3 * np.sqrt(3) * lat_const)

special_k_points = {
    'GAMMA': [0, 0, 0],
    'K': [lat_const_rec * np.sqrt(3), lat_const_rec, 0],
    'K_prime': [lat_const_rec * np.sqrt(3), -lat_const_rec, 0],
    'M': [lat_const_rec * np.sqrt(3), 0, 0]
}

sym_points = ['GAMMA', 'M', 'K', 'GAMMA']
num_points = [25, 25, 25]
k_points = get_k_coords(sym_points, num_points, special_k_points)

# ------------------------------------------------------------------

fig_counter = 1


def graphene_first_nearest_neighbour():

    coords = """2
    Graphene
    C1   0.00   0.00   0.00
    C2   {}   0.00   0.00
    """.format(lat_const)

    # --------------------------- Basis set --------------------------

    s_orb = tb.Orbitals('C')
    s_orb.add_orbital("pz", energy=0, orbital=1, magnetic=0, spin=0)

    # ------------------------ set TB parameters----------------------

    t = 2.8
    tb.set_tb_params(PARAMS_C_C={'pp_pi': t})

    # --------------------------- Hamiltonian -------------------------

    h = tb.Hamiltonian(xyz=coords, nn_distance=1.5)
    h.initialize()
    h.set_periodic_bc(period)

    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

    # visualize
    global fig_counter
    plt.figure(fig_counter)
    fig_counter += 1
    ax = plt.axes()
    ax.set_title(r'Band structure of graphene, 1st NN')
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(band_structure), 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


def radial_dep(coords):
    """
        Step-wise radial dependence function
    """
    norm_of_coords = np.linalg.norm(coords)
    if norm_of_coords < 1.5:
        return 1
    elif 2.5 > norm_of_coords > 1.5:
        return 2
    elif 3.0 > norm_of_coords > 2.5:
        return 3
    else:
        return 100


def graphene_third_nearest_neighbour_with_overlaps():
    """
    All parameters are taken from Reich et al, Phys. Rev. B 66, 035412 (2002)
    Returns
    -------

    """

    coords = """2
    Graphene
    C1   0.00   0.00   0.00
    C2   {}   0.00   0.00
    """.format(lat_const)

    # --------------------------- Basis set --------------------------

    s_orb = tb.Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    # ------------------------ set TB parameters----------------------

    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    tb.set_tb_params(PARAMS_C_C1={'pp_pi': gamma0},
                     PARAMS_C_C2={'pp_pi': gamma1},
                     PARAMS_C_C3={'pp_pi': gamma2},
                     OV_C_C1={'pp_pi': s0},
                     OV_C_C2={'pp_pi': s1},
                     OV_C_C3={'pp_pi': s2})

    # --------------------------- Hamiltonian -------------------------

    h = tb.Hamiltonian(xyz=coords, nn_distance=3.1, comp_overlap=True)
    h.initialize(radial_dep)
    h.set_periodic_bc(period)

    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

    # visualize
    global fig_counter
    plt.figure(fig_counter)
    fig_counter += 1
    ax = plt.axes()
    ax.set_title('Band structure of graphene, 3d NN \n after Reich et al, Phys. Rev. B 66, 035412 (2002)')
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(band_structure), 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


def graphene_nanoribbons_zigzag():

    from ase.build.ribbon import graphene_nanoribbon
    from ase.visualize.plot import plot_atoms

    atoms = graphene_nanoribbon(11, 1, type='zigzag')

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()
    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Nanoribbon')

    for j, item in enumerate(coord):
        coords.append('C' + str(j+1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)

    s_orb = tb.Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    # ------------------------ set TB parameters----------------------

    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    tb.set_tb_params(PARAMS_C_C1={'pp_pi': gamma0},
                     PARAMS_C_C2={'pp_pi': gamma1},
                     PARAMS_C_C3={'pp_pi': gamma2},
                     OV_C_C1={'pp_pi': s0},
                     OV_C_C2={'pp_pi': s1},
                     OV_C_C3={'pp_pi': s2})

    # --------------------------- Hamiltonian -------------------------

    h = tb.Hamiltonian(xyz=coords, nn_distance=3.1, comp_overlap=True)
    h.initialize(radial_dep)
    h.set_periodic_bc(period)

    k_points = np.linspace(0.0, np.pi/period[0][1], 20)
    band_structure = np.zeros((len(k_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc([0.0, item, 0.0])

    # visualize
    ax = plt.axes()
    ax.set_title('Graphene nanoribbon, zigzag 11')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax.plot(k_points, np.sort(band_structure), 'k')
    ax.xaxis.grid()
    plt.show()

    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,00z')
    ax1.axis('off')
    plt.show()


def graphene_nanoribbons_armchair():

    from ase.build.ribbon import graphene_nanoribbon
    from ase.visualize.plot import plot_atoms

    atoms = graphene_nanoribbon(11, 1, type='armchair')

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()
    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Nanoribbon')

    for j, item in enumerate(coord):
        coords.append('C' + str(j+1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)

    s_orb = tb.Orbitals('C')
    s_orb.add_orbital("pz", energy=-0.28, orbital=1, magnetic=0, spin=0)

    # ------------------------ set TB parameters----------------------

    gamma0 = -2.97
    gamma1 = -0.073
    gamma2 = -0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    tb.set_tb_params(PARAMS_C_C1={'pp_pi': gamma0},
                     PARAMS_C_C2={'pp_pi': gamma1},
                     PARAMS_C_C3={'pp_pi': gamma2},
                     OV_C_C1={'pp_pi': s0},
                     OV_C_C2={'pp_pi': s1},
                     OV_C_C3={'pp_pi': s2})

    # --------------------------- Hamiltonian -------------------------

    h = tb.Hamiltonian(xyz=coords, nn_distance=3.1, comp_overlap=True)
    h.initialize(radial_dep)
    h.set_periodic_bc(period)

    k_points = np.linspace(0.0, np.pi/period[0][1], 20)
    band_structure = np.zeros((len(k_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc([0.0, item, 0.0])

    # visualize
    ax = plt.axes()
    ax.set_title('Graphene nanoribbon, armchair 11')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax.plot(k_points, np.sort(band_structure), 'k')
    ax.xaxis.grid()
    plt.show()

    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='90x,0y,00z')
    ax1.axis('off')
    plt.show()


def graphene_nanotube():

    from ase.build.tube import nanotube
    from ase.visualize.plot import plot_atoms

    n = 10
    m = 10

    atoms = nanotube(n, m)
    atoms.wrap()

    period = np.array([list(atoms.get_cell()[2])])
    period[:, [1, 2]] = period[:, [2, 1]]
    coord = atoms.get_positions()
    coord[:, [1, 2]] = coord[:, [2, 1]]
    coords = []
    coords.append(str(len(coord)))
    coords.append('Nanoribbon')

    for j, item in enumerate(coord):
        coords.append('C' + str(j + 1) + ' ' + str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]))

    coords = '\n'.join(coords)

    # --------------------------- Basis set --------------------------

    s_orb = tb.Orbitals('C')
    s_orb.add_orbital("pz", energy=0, orbital=1, magnetic=0, spin=0)
    # s_orb.add_orbital("py", energy=0, orbital=1, magnetic=1, spin=0)
    # s_orb.add_orbital("px", energy=0, orbital=1, magnetic=-1, spin=0)

    # ------------------------ set TB parameters----------------------

    t = 2.8
    tb.set_tb_params(PARAMS_C_C={'pp_pi': t})

    # --------------------------- Hamiltonian -------------------------

    h = tb.Hamiltonian(xyz=coords, nn_distance=1.7, comp_angular_dep=False)
    h.initialize()
    h.set_periodic_bc(period)

    k_points = np.linspace(0.0, np.pi/period[0][1], 20)
    band_structure = np.zeros((len(k_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc([0.0, item, 0.0])

    # visualize
    ax = plt.axes()
    ax.set_title('Band structure of carbon nanotube, ({0}, {1}) \n 1st nearest neighbour approximation'.format(n, m))
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax.plot(k_points, np.sort(band_structure), 'k')
    ax.xaxis.grid()
    plt.show()

    ax1 = plot_atoms(atoms, show_unit_cell=2, rotation='10x,50y,30z')
    ax1.axis('off')
    plt.show()


if __name__ == '__main__':

    graphene_first_nearest_neighbour()
    graphene_third_nearest_neighbour_with_overlaps()
    graphene_nanoribbons_zigzag()
    graphene_nanoribbons_armchair()
    graphene_nanotube()

