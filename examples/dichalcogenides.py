"""
This example script computes DOS and transmission function for the four-atom-width nanostrip
using the recursive Green's function algorithm.
"""
import numpy as np
import nanonet.tb as tb
from nanonet.tb import get_k_coords, set_tb_params
import matplotlib.pyplot as plt
import qsymm
from nanonet.tb.hamiltonian_qsymm import HamiltonianQsymm
from nanonet.tb.hamiltonian_sym import HamiltonianSym


def main():

    # parameters
    t0 = -0.184
    t1 = 0.401
    t2 = 0.507
    t11 = 0.218
    t12 = 0.338
    t22 = 0.057

    params = [t0,
              -3 / np.sqrt(3) / 2 * t2 - 1 / 2 * t1,
              np.sqrt(3) / 2 * t1 - 1 / 2 * t2,
              1 / 4 * t11 + 3 / 4 * t22,
              3 / 4 / np.sqrt(3) * (t22 - t11) + t12,
              3 / 4 / np.sqrt(3) * (t22 - t11) - t12]

    # Time reversal
    TR = qsymm.time_reversal(2, np.eye(3))

    # Mirror symmetry
    Mx = qsymm.mirror([1, 0], np.diag([1, -1, 1]))

    # Threefold rotation on d_z^2, d_xy, d_x^2-y^2 states.
    C3U = np.array([
        [1, 0, 0],
        [0, -0.5, -np.sqrt(3) / 2],
        [0, np.sqrt(3) / 2, -0.5]
    ])

    # Could also use the predefined representation of rotations on d-orbitals
    Ld = qsymm.groups.L_matrices(3, 2)
    C3U2 = qsymm.groups.spin_rotation(2 * np.pi * np.array([0, 0, 1 / 3]), Ld)

    # Restrict to d_z^2, d_xy, d_x^2-y^2 states
    mask = np.array([1, 2, 0])
    C3U2 = C3U2[mask][:, mask]

    assert np.allclose(C3U, C3U2)

    C3 = qsymm.rotation(1 / 3, U=C3U)

    symmetries = [TR, Mx, C3]

    # One site per unit cell (M atom), with three orbitals
    norbs = [('a', 3)]

    # Hopping to a neighbouring atom one primitive lattice vector away
    hopping_vectors = [('a', 'a', [1, 0])]

    set_tb_params(hopping_vectors=hopping_vectors,
                  symm=symmetries,
                  norbs=norbs,
                  params=params)

    lat_const = 3.1903159618
    # define the basis set - one s-type orbital
    mo_orb = tb.Orbitals('Mo')
    mo_orb.add_orbital('wannier', energy=1.046, tag=0)
    mo_orb.add_orbital('wannier', energy=2.104, tag=1)
    mo_orb.add_orbital('wannier', energy=2.104, tag=2)

    input_file = """1
                    MoS2
                    Mo    0.0  0.0   0.0
                 """

    # compute Hamiltonian matrices
    h = HamiltonianQsymm(xyz=input_file, nn_distance=3.2)
    h.initialize()
    period = np.array([[3.1903159618, 0.0, 0.0],
                       [1.5951579809, 2.7628946691, 0.0]])

    h.set_periodic_bc(period)

    special_k_points = {
        'GAMMA': [0, 0, 0],
        'K': [4 * np.pi / 3 / lat_const, 0, 0],
        'M': [np.pi / lat_const, np.pi / np.sqrt(3) / lat_const, 0]
    }

    # define wave vector coordinates
    sym_points = ['GAMMA', 'K', 'M', 'GAMMA']
    num_points = [25, 15, 25]
    k_points = get_k_coords(sym_points, num_points, special_k_points)

    # compute band structure
    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

    # visualize
    ax = plt.axes()
    ax.set_title(r'Band structure of MoS$_2$ layer')
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(np.real(band_structure))[:, :8], 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


def main1():

    # lattice constants
    a = 3.477
    b = 6.249

    # Define 4 sites with one orbital each
    sites = ['Ad', 'Ap', 'Bd', 'Bp']
    norbs = [(site, 1) for site in sites]

    Ad = np.array([0.86925, 2.18715, 0.0])
    Ap = np.array([0.86925, -0.68739, 0.0])
    Bp = np.array([-0.86925, 0.68739, 0.0])
    Bd = np.array([-0.86925, -2.18715, 0.0])

    rAp_rBd = (Bd - Ap)
    rBp_rAp = -(Bp - Ap)
    rBd_rAd = -(Bd - Ad) - np.array([0.0, b, 0.0])
    rAp_rBd /= np.linalg.norm(rAp_rBd)
    rBp_rAp /= np.linalg.norm(rBp_rAp)
    rBd_rAd /= np.linalg.norm(rBd_rAd)

    rAp_rBd = rAp_rBd[:2]
    rBp_rAp = rBp_rAp[:2]
    rBd_rAd = rBd_rAd[:2]
    # Define hoppings to include

    hopping_vectors = [
        ('Bd', 'Bd', np.array([1, 0])),
        ('Ap', 'Ap', np.array([1, 0])),
        ('Bd', 'Ap', rAp_rBd),
        ('Ap', 'Bp', rBp_rAp),
        ('Ad', 'Bd', rBd_rAd),
    ]

    # Inversion
    perm_inv = {'Ad': 'Bd', 'Ap': 'Bp', 'Bd': 'Ad', 'Bp': 'Ap'}
    onsite_inv = {site: (1 if site in ['Ad', 'Bd'] else -1) * np.eye(1) for site in sites}
    inversion = qsymm.groups.symmetry_from_permutation(-np.eye(2), perm_inv, norbs, onsite_inv)

    # Glide
    perm_glide = {site: site for site in sites}
    onsite_glide = {site: (1 if site in ['Ad', 'Bd'] else -1) * np.eye(1) for site in sites}
    glide = qsymm.groups.symmetry_from_permutation(np.array([[-1, 0], [0, 1]]), perm_glide, norbs, onsite_glide)

    # TR
    time_reversal = qsymm.time_reversal(2, np.eye(4))

    symmetries = {glide, inversion, time_reversal}

    tpx = 1.34
    tdx = -0.4
    tpAB = 0.40
    tdAB = 0.52
    t0AB = 1.02

    params = [t0AB, tdx, tpx, tdAB, tpAB]

    set_tb_params(hopping_vectors=hopping_vectors,
                  symm=symmetries,
                  norbs=norbs,
                  params=params)

    # define the basis set - one s-type orbital
    mo_orb = tb.Orbitals('WA')
    mo_orb.add_orbital('wannier', energy=1.44, tag=0)
    mo_orb = tb.Orbitals('TeA')
    mo_orb.add_orbital('wannier', energy=-0.38, tag=1)
    mo_orb = tb.Orbitals('WB')
    mo_orb.add_orbital('wannier', energy=1.44, tag=2)
    mo_orb = tb.Orbitals('TeB')
    mo_orb.add_orbital('wannier', energy=-0.38, tag=3)

    input_file = """4
                    WTe2
                    WA   -{0}   {1}   0.0
                    TeA   -{0}  -{2}   0.0
                    WB    {0}  -{1}   0.0
                    TeB    {0}   {2}   0.0
                 """.format(0.25*a, 0.35*b, 0.11*b)

    # compute Hamiltonian matrices
    h = HamiltonianQsymm(xyz=input_file, nn_distance=1.1*b)
    h.initialize()
    period = np.array([[a, 0.0, 0.0],
                       [0.0, b, 0.0]])

    h.set_periodic_bc(period)

    special_k_points = {
        'GAMMA': [0, 0, 0],
        'Y': [0, np.pi / b, 0],
        'X': [np.pi / a, 0, 0],
        'M': [np.pi / a, np.pi / b, 0]
    }

    # aaa = h.get_tb_matrix(special_k_points['K'])

    # define wave vector coordinates
    sym_points = ['X', 'GAMMA', 'Y', 'M', 'GAMMA']
    num_points = [25, 15, 15, 15]
    k_points = get_k_coords(sym_points, num_points, special_k_points)

    # compute band structure
    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]), dtype=np.complex)

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize(item)

    # visualize
    ax = plt.axes()
    ax.set_title(r'Band structure of MoS$_2$ layer')
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(np.real(band_structure))[:, :8], 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


def main2():

    # Define 4 sites with one orbital each
    sites = ['Ad', 'Ap', 'Bd', 'Bp']
    norbs = [(site, 1) for site in sites]

    # Define symbolic coordinates for orbitals
    rAp = qsymm.sympify('[x_Ap, y_Ap]')
    rAd = qsymm.sympify('[x_Ad, y_Ad]')
    rBp = qsymm.sympify('[x_Bp, y_Bp]')
    rBd = qsymm.sympify('[x_Bd, y_Bd]')

    a = 3.477
    b = 6.249

    Ad = np.array([-0.86925, 1.99968, 0.0])
    Ap = np.array([-0.86925, -0.43743, 0.0])
    Bp = np.array([0.86925, 0.43743, 0.0])
    Bd = np.array([0.86925, -1.99968, 0.0])

    rAp_rBd = (Bd - Ap)                              # t0AB
    rBp_rAp = -(Bp - Ap)                             # pAB
    rBd_rAd = -(Bd - Ad)                             # dAB
    # rAd_rAp = Ad - Ap
    # rAd_rAp1 = np.array([0.0, b, 0.0]) - (Ad - Ap)
    rAd_rAp2 = -np.array([a, 0, 0.0]) - (Ad - Ap)     # t0x
    rAp_rBd1 = -(Bd - Ap) - np.array([a, 0, 0.0])       # t0ABx

    hopping_vectors = [
        ('Bd', 'Bd', np.array([a, 0, 0])),   # tdx
        ('Ap', 'Ap', np.array([a, 0, 0])),   # tpx
        ('Ap', 'Ap', np.array([0, b, 0])),   # tpy
        ('Bd', 'Ap', rAp_rBd),               # t0AB
        ('Ap', 'Bp', rBp_rAp),               # pAB
        ('Ad', 'Bd', rBd_rAd),               # dAB
        ('Bd', 'Ap', rAp_rBd1),              # t0ABx
        ('Ad', 'Ap', rAd_rAp2)               # t0x
    ]

    hopping_vectors = [(item[0], item[1], item[2] / np.linalg.norm(item[2])) for item in hopping_vectors]
    hopping_vectors = [(item[0], item[1], item[2][:2]) for item in hopping_vectors]

    # Inversion
    perm_inv = {'Ad': 'Bd', 'Ap': 'Bp', 'Bd': 'Ad', 'Bp': 'Ap'}
    onsite_inv = {site: (1 if site in ['Ad', 'Bd'] else 1) * np.eye(1) for site in sites}
    inversion = qsymm.groups.symmetry_from_permutation(-np.eye(2), perm_inv, norbs, onsite_inv)

    onsite_inv1 = {site: (1 if site in ['Ad', 'Bd'] else 1) * np.eye(1) for site in sites}
    inversion1 = qsymm.groups.symmetry_from_permutation(-np.eye(2), perm_inv, norbs, onsite_inv1)

    # Glide
    perm_glide = {site: site for site in sites}
    onsite_glide = {site: (1 if site in ['Ad', 'Bd'] else -1) * np.eye(1) for site in sites}
    glide = qsymm.groups.symmetry_from_permutation(np.array([[-1, 0], [0, 1]]), perm_glide, norbs, onsite_glide)
    glide1 = qsymm.groups.symmetry_from_permutation(np.array([[1, 0], [0, -1]]), perm_glide, norbs, onsite_glide)

    # TR
    time_reversal = qsymm.time_reversal(2, np.eye(4))

    symmetries = {glide, inversion, time_reversal}

    tpx = 1.13
    tdx = -0.41
    tpAB = 0.40
    tdAB = 0.51
    t0ABx = 0.39
    t0AB = 0.29
    t0x = 0.14
    tpy = 0.13

    params = [tdx, tpx, t0ABx, t0AB, tpy, tdAB, tpAB, t0x]
    # params = [tdx, tpx,tdAB, tpAB, tpy, t0ABx, t0AB, t0x]

    set_tb_params(hopping_vectors=hopping_vectors,
                  symm=symmetries,
                  norbs=norbs,
                  params=params)

    # define the basis set - one s-type orbital
    mo_orb = tb.Orbitals('WA')
    mo_orb.add_orbital('wannier', energy=0.74, tag=0)
    mo_orb = tb.Orbitals('TeA')
    mo_orb.add_orbital('wannier', energy=-1.75, tag=1)
    mo_orb = tb.Orbitals('WB')
    mo_orb.add_orbital('wannier', energy=0.74, tag=2)
    mo_orb = tb.Orbitals('TeB')
    mo_orb.add_orbital('wannier', energy=-1.75, tag=3)

    input_file = """4
                    WTe2
                    WA   -{0}   {1}   0.0
                    TeA   -{0}  -{2}   0.0
                    WB    {0}  -{1}   0.0
                    TeB    {0}  {2}   0.0
                 """.format(0.25*a, 0.32*b, 0.07*b)

    # compute Hamiltonian matrices
    h = HamiltonianQsymm(xyz=input_file, nn_distance=1.1 * b)
    # h = HamiltonianSym(xyz=input_file, nn_distance=1.01 * b)
    h.initialize()
    period = np.array([[a, 0.0, 0.0],
                       [0.0, b, 0.0]])

    h.set_periodic_bc(period)

    special_k_points = {
        'GAMMA': [0, 0, 0],
        'Y': [0, np.pi / b, 0],
        'X': [np.pi / a, 0, 0],
        'M': [np.pi / a, np.pi / b, 0]
    }

    # aaa = h.get_tb_matrix(special_k_points['K'])

    # define wave vector coordinates
    sym_points = ['X', 'GAMMA', 'Y', 'M', 'GAMMA']
    num_points = [25, 15, 15, 15]
    k_points = get_k_coords(sym_points, num_points, special_k_points)

    # compute band structure
    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]), dtype=np.complex)

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize(item)

    # visualize
    ax = plt.axes()
    ax.set_title(r'Band structure of MoS$_2$ layer')
    ax.set_ylabel('Energy (eV)')
    ax.plot(np.sort(np.real(band_structure))[:, :8], 'k')
    ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
    plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
    ax.xaxis.grid()
    plt.show()


# def main3():
#     # Define 4 sites with one orbital each
#     sites = ['Ad', 'Ap', 'Bd', 'Bp']
#     norbs = [(site, 1) for site in sites]
#
#     # Define symbolic coordinates for orbitals
#     rAp = qsymm.sympify('[x_Ap, y_Ap]')
#     rAd = qsymm.sympify('[x_Ad, y_Ad]')
#     rBp = qsymm.sympify('[x_Bp, y_Bp]')
#     rBd = qsymm.sympify('[x_Bd, y_Bd]')
#
#     a = 3.477
#     b = 6.249
#
#     # Ad = np.array([-0.86925, 1.99968, 0.0])
#     # Ap = np.array([-0.86925, -0.43743, 0.0])
#     # Bp = np.array([0.86925, 0.43743, 0.0])
#     # Bd = np.array([0.86925, -1.99968, 0.0])
#
#     # Ad = np.array([0.86925, 1.99968, 0.0])
#     # Ap = np.array([0.86925, -0.43743, 0.0])
#     # Bp = np.array([-0.86925, 0.43743, 0.0])
#     # Bd = np.array([-0.86925, -1.99968, 0.0])
#     rAp_rBd = (rBd - rAp)
#     rBp_rAp = -(rBp - rAp)
#     rBd_rAd = -(rBd - rAd) - np.array([0.0, b, 0.0])
#     # rAd_rAp = Ad - Ap
#     # rAd_rAp1 = np.array([0.0, b, 0.0]) - (Ad - Ap)
#     rAd_rAp2 = np.array([a, 0, 0.0]) - (Ad - Ap)
#     rAp_rBd1 = (Ap - Bd) - np.array([a, 0, 0.0])
#
#     hopping_vectors = [
#         ('Bd', 'Bd', np.array([1, 0, 0])),
#         ('Ap', 'Ap', np.array([1, 0, 0])),
#         ('Ap', 'Ap', np.array([0, 1, 0])),
#         ('Bd', 'Ap', rAp_rBd),
#         ('Ap', 'Bp', rBp_rAp),
#         ('Ad', 'Bd', rBd_rAd),
#         ('Bd', 'Ap', rAp_rBd1),
#         ('Ad', 'Ap', rAd_rAp2)
#     ]
#
#     hopping_vectors = [(item[0], item[1], item[2] / np.linalg.norm(item[2])) for item in hopping_vectors]
#     hopping_vectors = [(item[0], item[1], item[2][:2]) for item in hopping_vectors]
#
#     # Inversion
#     perm_inv = {'Ad': 'Bd', 'Ap': 'Bp', 'Bd': 'Ad', 'Bp': 'Ap'}
#     onsite_inv = {site: (1 if site in ['Ad', 'Bd'] else -1) * np.eye(1) for site in sites}
#     inversion = qsymm.groups.symmetry_from_permutation(-np.eye(2), perm_inv, norbs, onsite_inv)
#
#     onsite_inv1 = {site: (1 if site in ['Ad', 'Bd'] else 1) * np.eye(1) for site in sites}
#     inversion1 = qsymm.groups.symmetry_from_permutation(-np.eye(2), perm_inv, norbs, onsite_inv1)
#
#     # Glide
#     perm_glide = {site: site for site in sites}
#     onsite_glide = {site: (1 if site in ['Ad', 'Bd'] else -1) * np.eye(1) for site in sites}
#     glide = qsymm.groups.symmetry_from_permutation(np.array([[-1, 0], [0, 1]]), perm_glide, norbs, onsite_glide)
#     glide1 = qsymm.groups.symmetry_from_permutation(np.array([[1, 0], [0, -1]]), perm_glide, norbs, onsite_glide)
#
#     # TR
#     time_reversal = qsymm.time_reversal(2, np.eye(4))
#
#     symmetries = {glide, inversion, time_reversal}
#
#     tpx = 1.13
#     tdx = -0.41
#     tpAB = 0.40
#     tdAB = 0.51
#     t0AB = 0.39
#     t0ABx = 0.29
#     t0x = 0.14
#     tpy = 0.13
#
#     params = [tdx, tpx, t0AB, t0ABx, tpy, tdAB, tpAB, t0x]
#     params = [tpx, tdx, tpAB, tdAB, t0AB, t0ABx, t0x, tpy]
#     params = [tdx, tpx, t0ABx, t0AB, tpy, tdAB, tpAB, t0x]
#
#     set_tb_params(hopping_vectors=hopping_vectors,
#                   symm=symmetries,
#                   norbs=norbs,
#                   params=params)
#
#     # define the basis set - one s-type orbital
#     mo_orb = tb.Orbitals('WA')
#     mo_orb.add_orbital('wannier', energy=0.74, tag=0)
#     mo_orb = tb.Orbitals('TeA')
#     mo_orb.add_orbital('wannier', energy=-1.75, tag=1)
#     mo_orb = tb.Orbitals('WB')
#     mo_orb.add_orbital('wannier', energy=0.74, tag=2)
#     mo_orb = tb.Orbitals('TeB')
#     mo_orb.add_orbital('wannier', energy=-1.75, tag=3)
#
#     input_file = """4
#                     WTe2
#                     WA   -{0}   {1}   0.0
#                     TeA   -{0}  -{2}   0.0
#                     WB    {0}  -{1}   0.0
#                     TeB    {0}  {2}   0.0
#                  """.format(0.25 * a, 0.32 * b, 0.07 * b)
#
#     # compute Hamiltonian matrices
#     h = HamiltonianQsymm(xyz=input_file, nn_distance=1.1 * b)
#     h.initialize()
#     period = np.array([[a, 0.0, 0.0],
#                        [0.0, b, 0.0]])
#
#     h.set_periodic_bc(period)
#
#     special_k_points = {
#         'GAMMA': [0, 0, 0],
#         'Y': [0, np.pi / b, 0],
#         'X': [np.pi / a, 0, 0],
#         'M': [np.pi / a, np.pi / b, 0]
#     }
#
#     # aaa = h.get_tb_matrix(special_k_points['K'])
#
#     # define wave vector coordinates
#     sym_points = ['X', 'GAMMA', 'Y', 'M', 'GAMMA']
#     num_points = [25, 15, 15, 15]
#     k_points = get_k_coords(sym_points, num_points, special_k_points)
#
#     # compute band structure
#     band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]), dtype=np.complex)
#
#     for jj, item in enumerate(k_points):
#         band_structure[jj, :], _ = h.diagonalize(item)
#
#     # visualize
#     ax = plt.axes()
#     ax.set_title(r'Band structure of MoS$_2$ layer')
#     ax.set_ylabel('Energy (eV)')
#     ax.plot(np.sort(np.real(band_structure))[:, :8], 'k')
#     ax.plot([0, band_structure.shape[0]], [0, 0], '--', color='k', linewidth=0.5)
#     plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
#     ax.xaxis.grid()
#     plt.show()


if __name__ == '__main__':

    main()
