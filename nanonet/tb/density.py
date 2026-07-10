import matplotlib.pyplot as plt
import numpy as np
from nanonet.transport.aux_functions import fd
from ase.dft.kpoints import monkhorst_pack
from sisl.physics import MonkhorstPack
from nanonet.config import comm, rank, size, mpi_available, MPI, set_mpi
# set_mpi(False)


def delta(energy):

    broadening = 0.05
    dos = (1.0 / (broadening * np.sqrt(np.pi))) * np.exp(-(energy / broadening)**2)
    return dos


def compute_density1(hamiltonian):

    atoms = hamiltonian.to_ase_atoms()

    mp_kpoints = monkhorst_pack([10, 10, 1])
    cell = atoms.cell.reciprocal() * np.pi * 2
    kpoints = mp_kpoints @ cell.T

    return 0


def compute_density(hamiltonian, pot, ef, tempr, print_all=False):

    atoms = hamiltonian.to_sisl_geom()
    nk = atoms.lattice.pbc.astype(int) * 32
    nk[nk == 0] = 1
    kpts = MonkhorstPack(atoms.cell, nk, trs=False)

    weights = kpts.weight
    kpts = kpts.k
    cell = atoms.lattice.rcell
    kpts = kpts @ cell.T

    local_dens1 = np.array(0, dtype=np.float64)
    local_dens2 = np.array(0, dtype=np.float64)
    dens1 = np.array(0, dtype=np.float64)
    dens2 = np.array(0, dtype=np.float64)
    local_ec = np.array(100, dtype=np.float64)
    local_ev = np.array(-100, dtype=np.float64)
    ec = np.array(0, dtype=np.float64)
    ev = np.array(0, dtype=np.float64)

    indices = list(range(rank, kpts.shape[0], size))

    for jj in indices:
        energy, vects = hamiltonian.diagonalize_periodic_bc(kpts[jj])

        local_ec = min(energy[4], local_ec)
        local_ev = max(energy[3], local_ev)

        a = np.abs(vects) ** 2

        local_dens1 += np.sum(np.sum(a[:4,:], axis=0) * weights[jj] * fd(energy, ef + pot, tempr))
        local_dens2 += np.sum(np.sum(a[4:,:], axis=0) * weights[jj] * fd(energy, ef + pot, tempr))

    if mpi_available:
        comm.Allreduce(local_dens1, dens1, op=MPI.SUM)
        comm.Allreduce(local_dens2, dens2, op=MPI.SUM)
        comm.Allreduce(local_ec, ec, op=MPI.MIN)
        comm.Allreduce(local_ev, ev, op=MPI.MAX)
    else:
        dens1 = local_dens1
        dens2 = local_dens2
        ec = local_ec
        ev = local_ev

    dens1 -= 2.0   # positive potential of the crystal lattice
    dens2 -= 2.0   # positive potential of the crystal lattice

    # from number of electrons in unit cell to charge density

    dens1 *= (1.0 / (hamiltonian.ct.pcv[0, 0] * hamiltonian.ct.pcv[1, 1] * 1e-10 * 1e-10))
    dens2 *= (1.0 / (hamiltonian.ct.pcv[0, 0] * hamiltonian.ct.pcv[1, 1] * 1e-10 * 1e-10))

    if print_all:
        return dens1, dens2, ev, ec
    else:
        return dens1, dens2


def compute_dos(en, hamiltonian):
    atoms = hamiltonian.to_sisl_geom()
    nk = atoms.lattice.pbc.astype(int) * 190
    nk[nk == 0] = 1
    kpts = MonkhorstPack(atoms.cell, nk, trs=True)
    weights = kpts.weight
    kpts = kpts.k
    cell = atoms.lattice.rcell
    kpts = kpts @ cell.T

    dos = np.zeros_like(en)

    for jj1, kpt in enumerate(kpts):
        energy, _ = h.diagonalize_periodic_bc(kpt)
        for e in energy:
            dos += delta(en - e)

    return dos


if __name__=="__main__":

    from pathlib import Path
    import sys
    from ase.visualize import view

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from examples.graphene_bilayer_rect import make_hamiltonian, make_band_structure

    h = make_hamiltonian(0.0, 0.0)
    # ham = np.load("/Users/mykhailoklymenko/Monash_work/data/h.npy")
    # print(ham)
    # print(h.h_matrix_bc_factor - ham)
    ec, ev = make_band_structure(h, visualize=False)
    print(ec - ev)
    dens1, dens2, ev, ec = compute_density(h, 0.0, 0.0, 300, print_all=True)
    print(dens1)
    print(dens2)
    print(ev)
    print(ec)

    # energy = np.linspace(-12.0, 10.0, 2000)
    # dens = compute_dos(energy, h)
    #
    # plt.plot(energy, dens)
    # plt.show()

    # np.save("dos.npy", dens)


