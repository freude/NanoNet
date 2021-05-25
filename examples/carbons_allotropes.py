import matplotlib.pyplot as plt
import numpy as np
from nanonet.tb import get_k_coords
import nanonet.tb as tb

# --------------------- vectors (in Angstroms) --------------------

lat_const = 1.42
a1 = lat_const * 3
a2 = lat_const * (np.sqrt(3) + 1.0)

period = np.array([[a1, 0.0, 0.0],
                   [0.0, a2, 0.0]])

# --------------------------- wave vectors -------------------------

lat_const_rec = 2 * np.pi / (3 * np.sqrt(3) * lat_const)

special_k_points = {
    'GAMMA': [0, 0, 0],
    'X': [np.pi / a1, 0, 0],
    'Z': [np.pi / a1, np.pi / a2, 0],
    'Y': [0, np.pi / a2, 0]
}

sym_points = ['GAMMA', 'X', 'Z', 'Y', 'GAMMA', 'Z']
num_points = [25, 25, 25, 25, 25]
k_points = get_k_coords(sym_points, num_points, special_k_points)

# ------------------------------------------------------------------

fig_counter = 1


def _plot_bs_1D(band_structure, title, num_points):

    global fig_counter
    fig_counter += 1

    plt.figure(fig_counter)
    ax = plt.axes()
    ax.set_title(title)
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(band_structure), 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.ylim([-4, 4])
    plt.show()


def _plot_bs_2D(X, Y, band_structure, title):

    global fig_counter
    fig_counter += 1

    plt.figure(fig_counter)
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    ax.plot_surface(X, Y, band_structure[:, :, 2], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, band_structure[:, :, 3], rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_zlabel('Energy (eV)')
    plt.show()


def graphene_first_nearest_neighbour():

    coords = """6
    Graphene
    C1   0.00   0.00   0.00
    C2    {0}   0.00   0.00
    C3   0.00    {0}   0.00
    C4    {0}    {0}   0.00
    C5    {1}    {3}   0.00
    C6    {2}    {3}   0.00    
    """.format(lat_const, 1.5*lat_const, 2.5*lat_const, lat_const * (1+0.5*np.sqrt(3)))

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
    _plot_bs_1D(band_structure, 'Band structure of biphenylene, 1st NN', num_points)


def graphene_first_nearest_neighbour_2D():

    coords = """6
    Graphene
    C1   0.00   0.00   0.00
    C2    {0}   0.00   0.00
    C3   0.00    {0}   0.00
    C4    {0}    {0}   0.00
    C5    {1}    {3}   0.00
    C6    {2}    {3}   0.00    
    """.format(lat_const, 1.5*lat_const, 2.5*lat_const, lat_const * (1+0.5*np.sqrt(3)))

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

    num_k = 50
    k_x = np.linspace(-np.pi/a1, np.pi/a1, num_k)
    k_y = np.linspace(-np.pi/a2, np.pi/a2, num_k)

    band_structure = np.zeros((num_k, num_k, h.h_matrix.shape[0]))

    for jx, item_x in enumerate(k_x):
        for jy, item_y in enumerate(k_y):
            band_structure[jx, jy, :], _ = h.diagonalize_periodic_bc(np.array([item_x, item_y, 0.0]))

    # visualize
    print('hi')
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


def graphene_third_nearest_neighbour_with_overlaps():
    """
    All parameters are taken from Reich et al, Phys. Rev. B 66, 035412 (2002)
    Returns
    -------

    """

    coords = """6
    Graphene
    C1   0.00   0.00   0.00
    C2    {0}   0.00   0.00
    C3   0.00    {0}   0.00
    C4    {0}    {0}   0.00
    C5    {1}    {3}   0.00
    C6    {2}    {3}   0.00    
    """.format(lat_const, 1.5*lat_const, 2.5*lat_const, lat_const * (1+0.5*np.sqrt(3)))

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

    h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True)
    h.initialize()
    h.set_periodic_bc(period)

    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

    # visualize
    _plot_bs_1D(band_structure,
                'Band structure of biphenylene, 3d NN \n after Reich et al, Phys. Rev. B 66, 035412 (2002)',
                num_points)


def graphene_third_nearest_neighbour_with_overlaps_2D():
    """
    All parameters are taken from Reich et al, Phys. Rev. B 66, 035412 (2002)
    Returns
    -------

    """

    coords = """6
    Graphene
    C1   0.00   0.00   0.00
    C2    {0}   0.00   0.00
    C3   0.00    {0}   0.00
    C4    {0}    {0}   0.00
    C5    {1}    {3}   0.00
    C6    {2}    {3}   0.00    
    """.format(lat_const, 1.5*lat_const, 2.5*lat_const, lat_const * (1+0.5*np.sqrt(3)))

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

    h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True)
    h.initialize()
    h.set_periodic_bc(period)

    num_k = 50
    k_x = np.linspace(-np.pi/a1, np.pi/a1, num_k)
    k_y = np.linspace(-np.pi/a2, np.pi/a2, num_k)

    X, Y = np.meshgrid(k_x, k_y)

    band_structure = np.zeros((num_k, num_k, h.h_matrix.shape[0]))

    for jx, item_x in enumerate(k_x):
        for jy, item_y in enumerate(k_y):
            band_structure[jx, jy, :], _ = h.diagonalize_periodic_bc(np.array([item_x, item_y, 0.0]))

    # visualize
    _plot_bs_2D(X, Y, band_structure,
                'Band structure of biphenylene, 3d NN \n after Reich et al, Phys. Rev. B 66, 035412 (2002)')


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

    h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True)
    h.initialize()
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

    h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True)
    h.initialize()
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

    # graphene_first_nearest_neighbour()
    # graphene_third_nearest_neighbour_with_overlaps()
    graphene_third_nearest_neighbour_with_overlaps_2D()
    # graphene_nanoribbons_zigzag()
    # graphene_nanoribbons_armchair()
    # graphene_nanotube()

