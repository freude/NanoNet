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
        band_structure[jj, :], _ = h.diagonalize(item)

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

    h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True)
    h.initialize()
    h.set_periodic_bc(period)

    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))
    eigenvects = np.zeros((sum(num_points), h.h_matrix.shape[0], h.h_matrix.shape[1]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize(item)

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


def graphene_third_nearest_neighbour_with_overlaps_eigenvectors():
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

    kx = np.linspace(-0.9*np.pi / a2, 0.9*np.pi / a2, 50)
    ky = np.linspace(-0.9*np.pi / a2, 0.9*np.pi / a2, 50)

    # compute band structure
    band_structure = np.zeros((len(kx), len(ky), h.h_matrix.shape[0]), dtype=np.complex)
    vects = np.zeros((len(kx), len(ky), h.h_matrix.shape[0], h.h_matrix.shape[1]), dtype=np.complex)

    for jj1, item1 in enumerate(kx):
        for jj2, item2 in enumerate(ky):
            band_structure[jj1, jj2, :], vects1 = h.diagonalize([item1, item2, 0.0])
            for j in range(h.h_matrix.shape[0]):
                phase = np.angle(vects1[0, j])
                vects[jj1, jj2, :, j] = vects1[:, j] * np.exp(-1j*phase)

    # visualize
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Contour-plot of the band structure of graphene')
    ax[0].set_title('Valence band')
    ax[0].contourf(band_structure[:, :, 0], 10)
    ax[0].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[0].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    ax[1].set_title('Conduction band')
    ax[1].contourf(band_structure[:, :, 1], 10)
    # ax[0, 1].set_ylabel(r'k$_x$ ($\frac{\pi}{b}$)')
    ax[1].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(kx, ky)
    ax.set_title('Surface plot of the band structure of graphene')
    ax.plot_surface(X, Y, band_structure[:, :, 0])
    ax.plot_surface(X, Y, band_structure[:, :, 1])
    ax.set_zlabel('Energy (eV)')
    ax.set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    ax.set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(3, 2)
    fig.suptitle('Eigenvector components for the valence band')
    ax[0,0].contourf(np.abs(vects[:, :, 0, 0]), 10)
    ax[0,0].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[0,0].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    ax[0,1].contourf(np.abs(vects[:, :, 1, 0]), 10)
    # ax[0, 1].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[0,1].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')

    ax[1,0].contourf(np.real(vects[:, :, 0, 0]), 10)
    ax[1,0].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[1,0].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    ax[1,1].contourf(np.real(vects[:, :, 1, 0]), 10)
    # ax[0, 1].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[1,1].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')

    ax[2,0].contourf(np.imag(vects[:, :, 0, 0]), 10)
    ax[2,0].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[2,0].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    ax[2,1].contourf(np.imag(vects[:, :, 1, 0]), 10)
    # ax[0, 1].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[2,1].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    plt.show()

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Absolute values of the eigenvector components for the conduction band')
    ax[0].contourf(np.abs(vects[:, :, 0, 1]), 10)
    ax[0].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[0].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    ax[1].contourf(np.abs(vects[:, :, 1, 1]), 10)
    # ax[0, 1].set_ylabel(r'k$_y$ ($\frac{\pi}{b}$)')
    ax[1].set_xlabel(r'k$_x$ ($\frac{\pi}{a}$)')
    plt.show()

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].contourf(np.angle(vects[:, :, 0, 0]), 10)
    ax[0, 1].contourf(np.angle(vects[:, :, 1, 0]), 10)
    ax[1, 0].contourf(np.angle(vects[:, :, 0, 1]), 10)
    ax[1, 1].contourf(np.angle(vects[:, :, 1, 1]), 10)
    plt.show()

    ans = np.zeros((h.h_matrix.shape[0], h.h_matrix.shape[1], len(kx), len(ky)), dtype=np.complex)

    for j1 in range(h.h_matrix.shape[0]):
        for j2 in range(h.h_matrix.shape[1]):
            aa1 = vects[:, :, :, j1]
            aa2 = vects[:, :, :, j2]
            grd = np.gradient(aa2, axis=(0, 1))
            for jj1, item1 in enumerate(kx):
                for jj2, item2 in enumerate(ky):
                    vec = grd[0][jj1, jj2, :] * item1 + grd[1][jj1, jj2, :] * item2
                    ans[j1, j2, jj1, jj2] = 1j*np.dot(np.conj(aa1[jj1, jj2, :]), vec)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Eigenvector components for band #2 (absolute values)')
    ax[0, 0].contourf(np.abs(ans[0, 0, :, :]), 10)
    ax[0, 1].contourf(np.abs(ans[0, 1, :, :]), 10)
    ax[1, 0].contourf(np.abs(ans[1, 0, :, :]), 10)
    ax[1, 1].contourf(np.abs(ans[1, 1, :, :]), 10)
    plt.show()

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Eigenvector components for band #2 (absolute values)')
    ax[0, 0].contourf(np.real(ans[0, 0, :, :]), 10)
    ax[0, 1].contourf(np.real(ans[0, 1, :, :]), 10)
    ax[1, 0].contourf(np.real(ans[1, 0, :, :]), 10)
    ax[1, 1].contourf(np.real(ans[1, 1, :, :]), 10)
    plt.show()

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Eigenvector components for band #2 (absolute values)')
    ax[0, 0].contourf(np.imag(ans[0, 0, :, :]), 10)
    ax[0, 1].contourf(np.imag(ans[0, 1, :, :]), 10)
    ax[1, 0].contourf(np.imag(ans[1, 0, :, :]), 10)
    ax[1, 1].contourf(np.imag(ans[1, 1, :, :]), 10)
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

    h = tb.Hamiltonian(xyz=coords, nn_distance=[1.5, 2.5, 3.1], comp_overlap=True)
    h.initialize()
    h.set_periodic_bc(period)

    k_points = np.linspace(0.0, np.pi/period[0][1], 20)
    band_structure = np.zeros((len(k_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize([0.0, item, 0.0])

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
        band_structure[jj, :], _ = h.diagonalize([0.0, item, 0.0])

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
        band_structure[jj, :], _ = h.diagonalize([0.0, item, 0.0])

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
    graphene_third_nearest_neighbour_with_overlaps_eigenvectors()
    graphene_nanoribbons_zigzag()
    graphene_nanoribbons_armchair()
    graphene_nanotube()

