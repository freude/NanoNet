import matplotlib.pyplot as plt
import numpy as np
from nanonet.tb import get_k_coords
import nanonet.tb as tb
from nanonet.verbosity import set_verbosity
from nanonet.config import rank


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

def make_band_structure(h, visualize=True):

    lat_const = 1.42

    lat_const_rec = np.pi / (np.sqrt(3) * lat_const)

    special_k_points = {
        'GAMMA': np.array([0, 0, 0]) * lat_const_rec,
        'X': np.array([1, 0, 0]) * lat_const_rec,
        'W': np.array([1, 1 / np.sqrt(3), 0]) * lat_const_rec,
        'Y': np.array([0, 1 / np.sqrt(3), 0]) * lat_const_rec,
        'GAMMA1': np.array([0, 0, 0]) * lat_const_rec,
    }

    sym_points = ['GAMMA', 'X', 'W', 'Y', 'GAMMA', 'W']
    num_points = [50, 50, 50, 50, 50]
    k_points = get_k_coords(sym_points, num_points, special_k_points)

    band_structure = np.zeros((sum(num_points), h.h_matrix.shape[0]))

    for jj, item in enumerate(k_points):
        band_structure[jj, :], _ = h.diagonalize_periodic_bc(item)

    band_structure = np.sort(band_structure)
    occupied = 4
    ec = np.min(band_structure[:, occupied:])
    ev = np.max(band_structure[:, :occupied])
    # band_gap = ec - ev

    # dos = np.load('/Users/mykhailoklymenko/Monash_work/NanoNet/nanonet/tb/dos.npy')
    band_structure1 = np.load('/Users/mykhailoklymenko/Monash_work/data/bs1.npy')
    # energy = np.linspace(-12, 10, len(dos))

    if visualize and rank==0:

        plt.figure(1)
        ax = plt.axes()
        ax.set_ylabel('Energy (eV)')
        ax.plot(band_structure, 'k')
        ax.plot(band_structure1, '--')
        plt.xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
        ax.xaxis.grid()

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [2, 1], "wspace": 0})
        ax1.set_ylabel('Energy (eV)')
        ax1.plot(band_structure[:,:occupied], 'k')
        ax1.plot(band_structure[:,occupied:], 'r')
        ax1.set_xticks(np.insert(np.cumsum(num_points) - 1, 0, 0), labels=sym_points)
        ax1.xaxis.grid()
        # ax2.plot(dos, energy, 'b')
        plt.show()

    return ec, ev

def main():

    # compute eg vs field

    fields = np.linspace(-0.3, 0.3, 100)
    eg = np.zeros_like(fields)
    e_v = np.zeros_like(fields)
    e_c = np.zeros_like(fields)

    for j, field in enumerate(fields):
        print(j)
        h = make_hamiltonian(0.0, field)
        ec, ev = make_band_structure(h, visualize=False)
        eg[j] = ec - ev
        e_c[j] = ec
        e_v[j] = ev

    plt.figure(1)
    plt.plot(fields, eg)
    plt.xlabel('Potential [V]')
    plt.ylabel('Band gap [eV]')
    plt.show()

    plt.figure(1)
    plt.plot(fields, e_c)
    plt.plot(fields, e_v)
    plt.xlabel('Potential [V]')
    plt.ylabel('Band edges [eV]')
    plt.show()

    #
    # np.save('eg_vs_filed.npy', eg)

    eg = np.load('eg_vs_filed.npy')
    plt.figure(1)
    plt.plot(fields, eg)
    plt.show()

    from scipy import interpolate
    f = interpolate.interp1d(fields, eg)
    fields_new = np.linspace(0, 0.5, 50)
    eg_new = f(fields_new)
    plt.figure(1)
    plt.plot(fields_new, eg_new, '-o')
    plt.xlabel('Field [V/m]')
    plt.ylabel('Band gap [eV]')
    plt.show()


def main1():

    h = make_hamiltonian(0.0, 0.25)

    lat_const = 1.42
    lat_const_rec = np.pi / (np.sqrt(3) * lat_const)

    kx = np.linspace(lat_const_rec - 0.5, lat_const_rec - 0.35, 100)
    ky = np.linspace(-0.1, 0.1, 100)

    # kx = np.linspace(-lat_const_rec, lat_const_rec, 50)
    # ky = np.linspace(-1.0 / np.sqrt(3) * lat_const_rec,
    #                  1.0 / np.sqrt(3) * lat_const_rec, 50)

    kkx, kky = np.meshgrid(kx, ky)

    vb = np.zeros_like(kkx)
    cb = np.zeros_like(kkx)

    for j1 in range(len(kx)):
        for j2 in range(len(ky)):
            bands, _ = h.diagonalize_periodic_bc(np.array([kkx[j1, j2], kky[j1, j2], 0.0]))
            vb[j1, j2] = bands[3]
            cb[j1, j2] = bands[4]


    plt.contourf(kkx, kky, vb, 100, cmap="viridis")
    plt.show()

    plt.contourf(kkx, kky, cb, 100, cmap="viridis")
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(kkx, kky, cb, cmap="viridis")
    ax.plot_surface(kkx, kky, vb, cmap="viridis")
    plt.show()

if __name__ == "__main__":
    main()



