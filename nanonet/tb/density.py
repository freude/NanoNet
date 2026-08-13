import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from examples.graphene_bilayer_rect import make_hamiltonian
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from nanonet.transport.aux_functions import fd
from sisl.physics import MonkhorstPack
from nanonet.config import comm, rank, size, mpi_available, MPI, set_mpi
# set_mpi(False)


def delta(energy):
    """Gaussian approximation to the Dirac delta function with 0.03 eV broadening."""

    broadening = 0.03
    dos = (1.0 / (broadening * np.sqrt(np.pi))) * np.exp(-(energy / broadening)**2)
    return dos


def gen_kpts(hamiltonian, num_kpts, time_reversal_symemtry=True):

    atoms = hamiltonian.to_sisl_geom()
    nk = atoms.lattice.pbc.astype(int) * num_kpts
    nk[nk == 0] = 1
    kpts = MonkhorstPack(atoms.cell, nk, trs=time_reversal_symemtry)

    x_displ = set(np.abs(np.diff(kpts.k[:, 0])))
    x_displ.discard(0.0)
    x_displ = np.min(list(x_displ))

    y_displ = set(np.abs(np.diff(kpts.k[:, 1])))
    y_displ.discard(0.0)
    y_displ = np.min(list(y_displ))

    kpts = MonkhorstPack(atoms.cell, nk, [0.5 * x_displ, 0.5 * y_displ, 0.0], trs=time_reversal_symemtry)

    weights = kpts.weight
    kpts = kpts.k
    cell = atoms.lattice.rcell
    kpts = kpts @ cell.T

    return kpts, weights


def get_reciprocal_lattice(hamiltonian):

    atoms = hamiltonian.to_sisl_geom()
    cell = atoms.lattice.rcell

    return cell


def compute_density(hamiltonian, num_kpts, pot, ef, tempr,
                    print_all=False, use_mpi=True):
    """
    Compute the electron density by integrating over the Brillouin zone.

    The electron density is evaluated from the Hamiltonian for each k-point
    using the specified electrostatic potential, Fermi level, and temperature.
    The calculation can optionally be parallelized using MPI.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object providing the electronic structure and methods
        required to evaluate the density.
    num_kpts : int or tuple of int
        Number of k-points used for Brillouin-zone sampling. If a tuple is
        provided, it specifies the sampling along each reciprocal-space
        direction.
    pot : array_like
        Electrostatic potential applied to the Hamiltonian.
    ef : float
        Fermi level (chemical potential) in eV.
    tempr : float
        Electronic temperature in K.
    print_all : bool, optional
        If ``True``, print the energies of the band edges along with the electron density.
        Default is ``False``.
    use_mpi : bool, optional
        If ``True``, distribute the k-point calculations across MPI
        processes. Default is ``True``.

    Returns
    -------
    density : ndarray
        Electron density corresponding to the supplied potential.
    """

    kpts, weights = gen_kpts(hamiltonian, num_kpts)

    local_dens1 = np.array(0, dtype=np.float64)
    local_dens2 = np.array(0, dtype=np.float64)
    local_ec = np.array(100, dtype=np.float64)
    local_ev = np.array(-100, dtype=np.float64)

    dens1 = np.array(0, dtype=np.float64)
    dens2 = np.array(0, dtype=np.float64)
    ec = np.array(0, dtype=np.float64)
    ev = np.array(0, dtype=np.float64)

    if use_mpi:
        indices = list(range(rank, kpts.shape[0], size))
    else:
        indices = list(range(kpts.shape[0]))

    for jj in indices:
        energy, vects = hamiltonian.diagonalize_periodic_bc(kpts[jj])

        local_ec = min(energy[4], local_ec)
        local_ev = max(energy[3], local_ev)

        a = np.abs(vects) ** 2

        local_dens1 += np.sum(np.sum(a[:4,:], axis=0) * weights[jj] * fd(energy, ef + pot, tempr))
        local_dens2 += np.sum(np.sum(a[4:,:], axis=0) * weights[jj] * fd(energy, ef + pot, tempr))

    if mpi_available and use_mpi:
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


def compute_dos(en, hamiltonian, num_kpts):
    """
    Compute the electronic density of states (DOS).

    The DOS is obtained by diagonalizing the Hamiltonian at each k-point
    in the Brillouin-zone sampling and summing the broadened contributions
    from all eigenvalues. The computation is parallelized over k-points
    when MPI is available.

    Parameters
    ----------
    en : ndarray
        One-dimensional array of energy values (in eV) at which the DOS
        is evaluated.
    hamiltonian : Hamiltonian
        Hamiltonian object providing the electronic structure and methods
        for diagonalization.
    num_kpts : int or tuple of int
        Number of k-points used for Brillouin-zone sampling. If a tuple is
        provided, it specifies the sampling along each reciprocal-space
        direction.

    Returns
    -------
    dos : ndarray
        Density of states evaluated at the energies specified by `en`.
        The returned array has the same shape as `en`.
    """

    kpts, weights = gen_kpts(hamiltonian, num_kpts)
    local_dos = np.zeros_like(en)

    indices = list(range(rank, kpts.shape[0], size))

    for jj in indices:
        energy, _ = hamiltonian.diagonalize_periodic_bc(kpts[jj])
        for e in energy:
            local_dos += delta(en - e)

    dos = np.zeros_like(en)
    if mpi_available:
        comm.Allreduce(local_dos, dos, op=MPI.SUM)
    else:
        dos = local_dos

    return dos


def greens_function_from_tb(energy, h):
    """
    Compute the k-averaged retarded Green's function from a tight-binding
    Hamiltonian.

    The Green's function is evaluated by sampling the Brillouin zone,
    constructing the Hamiltonian at each k-point, and averaging the
    corresponding retarded Green's functions. The computation is
    parallelized over k-points when MPI is available.

    Parameters
    ----------
    energy : array_like
        One-dimensional array of energy values (in eV) at which the
        Green's function is evaluated.
    h : Hamiltonian
        Tight-binding Hamiltonian object providing the periodic
        Hamiltonian matrix for arbitrary k-points.

    Returns
    -------
    gf : ndarray
        Complex-valued Green's function evaluated at each energy. The
        returned array has shape ``(len(energy), N, N)``, where ``N`` is
        the number of orbitals (or basis functions) in the Hamiltonian.
    """

    num_kpts = 64
    kpts, weights = gen_kpts(h, num_kpts, time_reversal_symemtry=True)

    #######################

    num_kx = 64
    num_ky = 64

    cell = get_reciprocal_lattice(h)
    kx = np.linspace(-0.0 * cell[0, 0], 0.5 * cell[0, 0], num_kx)
    ky = np.linspace(-0.0 * cell[1, 1], 0.5 * cell[1, 1], num_ky)
    kkx, kky = np.meshgrid(kx, ky)
    kpts = np.column_stack((kkx.ravel(), kky.ravel(), np.zeros(kkx.size)))
    weights = (kx[2] - kx[1]) * (ky[2] - ky[1]) * (kpts[:, 0] - kpts[:, 0] + 1.0)

    #######################

    indices = np.array(list(range(rank, kpts.shape[0], size)))
    eta = 0.01
    IE = np.multiply.outer(energy + 1j * eta, np.identity(len(np.diag(h.h_matrix))), dtype=np.complex128)
    gf = np.zeros_like(IE)

    for jj in indices:
        mat = h.get_hamiltonian_periodic_bc(np.array([kpts[jj, 0], kpts[jj, 1], 0.0]))
        gf +=  weights[jj] * np.linalg.pinv(IE - mat)

    return gf


def greens_function_from_tb_with_mask(energy, h, mask, num_k:int=300, eta:float|int=0.01):
    """
    Compute the k-averaged retarded Green's function, transmission function and
    density of states from a tight-binding
    Hamiltonian using a k-space mask.

    The Green's function is evaluated by sampling the Brillouin zone and
    averaging the retarded Green's functions only over the k-points
    selected by `mask`. The mask may depend on both k and energy, allowing
    energy-dependent regions of reciprocal space to be included in the
    integration. Hamiltonian matrices are cached to avoid repeated
    evaluations at the same k-points.

    Parameters
    ----------
    energy : array_like
        One-dimensional array of energy values (in eV) at which the
        Green's function is evaluated.
    h : Hamiltonian
        Tight-binding Hamiltonian object providing the periodic
        Hamiltonian matrix for arbitrary k-points.
    mask : callable
        Callable object, an interpolant of a Boolean mask. The mask
        may depend on k only or on both k and energy.
    num_k : int
        Number of k-points in one dimension.
    eta : float
        Green's function infinitesimal values (eV).

    Returns
    -------
    gf : ndarray
        Complex-valued Brillouin-zone-averaged Green's function evaluated
        at each energy. The returned array has shape
        ``(len(energy), N, N)``, where ``N`` is the number of orbitals
        (or basis functions) in the Hamiltonian.
    tr: ndarray
        Transmission function
    dos: ndarray
        Density of states
    """

    num_kx = num_k
    num_ky = num_k
    cell = get_reciprocal_lattice(h)
    kx = np.linspace(-0.0 * cell[0, 0], 0.5 * cell[0, 0], num_kx)
    ky = np.linspace(-0.0 * cell[1, 1], 0.5 * cell[1, 1], num_ky)
    weight = (kx[2] - kx[1]) * (ky[2] - ky[1])
    kkx, kky = np.meshgrid(kx, ky)
    kpts = np.column_stack((kkx.ravel(), kky.ravel(), np.zeros(kkx.size)))
    indices = np.array(list(range(rank, kpts.shape[0], size)))
    print("Num. of indices: ", len(indices))

    IE = np.multiply.outer(energy + 1j * eta, np.identity(len(np.diag(h.h_matrix))), dtype=np.complex128)
    gf = np.zeros_like(IE)
    II = np.identity(len(np.diag(h.h_matrix)))
    mat = {}
    tr = np.zeros_like(energy)

    for j, en in enumerate(energy):
        print(j)
        kpts[:, 2] = en
        mask_bin = mask(kpts)
        indices_new = indices[mask_bin > 0.2]
        print("Num. of indices:", len(indices_new), "out of", len(indices), "indicies.")
        kspecdens = np.zeros_like(kpts[:, 0])

        for jj in indices_new:
            if (kpts[jj, 0], kpts[jj, 1], 0.0) not in mat:
                mat[(kpts[jj, 0], kpts[jj, 1], 0.0)] = h.get_hamiltonian_periodic_bc(np.array([kpts[jj, 0], kpts[jj, 1], 0.0]))
            kernel = weight * np.linalg.pinv(II * (en + 1j * eta) - mat[(kpts[jj, 0], kpts[jj, 1], 0.0)])
            gf[j, :, :] += kernel
            kspecdens[jj] = np.abs(np.imag(np.trace(kernel)))

        for j2 in range(len(ky)):
            p, _ = find_peaks(kspecdens.reshape((len(kx), len(ky)))[j2, :],
                              height=(0.00005, 5.1),
                              distance=5)

            tr[j] += len(p)

    tr *= (ky[2] - ky[1]) / (2 * np.pi)
    dos = -np.imag(np.trace(gf, axis1=1, axis2=2))

    return gf, tr, dos


def greens_function_k_dep(energy,  h):
    """
    Compute the k-resolved retarded Green's function on a uniform
    Brillouin-zone grid.

    The retarded Green's function is evaluated independently at each
    k-point of a uniform reciprocal-space grid. The trace of the Green's
    function is then used to construct a k-resolved intensity map and a
    corresponding Boolean mask.

    Parameters
    ----------
    energy : array_like
        One-dimensional array of energy values (in eV) at which the
        Green's function is evaluated.
    h : Hamiltonian
        Tight-binding Hamiltonian object providing the periodic
        Hamiltonian matrix for arbitrary k-points.

    Returns
    -------
    kx : ndarray
        One-dimensional array of sampled kx coordinates.
    ky : ndarray
        One-dimensional array of sampled ky coordinates.
    intensity : ndarray
        Absolute value of the trace of the Green's function at each
        k-point and energy. The returned array has shape
        ``(len(kx), len(ky), len(energy))``.
    """

    num_kx = 25
    num_ky = 25
    # num_kx = 120
    # num_ky = 120


    cell = get_reciprocal_lattice(h)
    kx = np.linspace(-0.5 * cell[0, 0], 0.5 * cell[0, 0], num_kx)
    ky = np.linspace(-0.5 * cell[1, 1], 0.5 * cell[1, 1], num_ky)
    weight = (kx[2] - kx[1]) * (ky[2] - ky[1])
    # kkx, kky = np.meshgrid(kx, ky)
    # kpts = np.column_stack((kkx.ravel(), kky.ravel(), np.zeros(kkx.size)))

    s = h.h_matrix.shape
    eta = 0.1
    # IE = (energy + 1j * eta) * np.identity(s[0])
    gf = np.zeros((num_kx, num_ky, len(energy), s[0], s[1]), dtype=np.complex128)
    IE = np.multiply.outer(energy + 1j * eta, np.identity(s[0]), dtype=np.complex128)


    for j1, k_x in enumerate(kx):
        for j2, k_y in enumerate(ky):
            mat = h.get_hamiltonian_periodic_bc(np.array([k_x, k_y, 0.0]))
            gf[j1, j2, :, :, :] +=  weight * np.linalg.pinv(IE - mat)

    return kx, ky, gf


def make_masks(energy, h, vis=True):
    """
    Construct an interpolant for the k-space mask from the k-resolved Green's
    function.

    The mask is generated by thresholding the k-resolved Green's function
    and creating a `RegularGridInterpolator` that can be evaluated at
    arbitrary k-points. If multiple energy values are provided, the mask
    is interpolated in both k-space and energy.

    Parameters
    ----------
    energy : array_like
        One-dimensional array of energy values (in eV) at which the mask
        is constructed.
    h : Hamiltonian
        Tight-binding Hamiltonian object providing the periodic
        Hamiltonian matrix.

    Returns
    -------
    masks : scipy.interpolate.RegularGridInterpolator
        Interpolator for the Boolean mask. It accepts either `(kx, ky)`
        coordinates for a single energy or `(kx, ky, energy)` coordinates
        when multiple energy values are provided.
    """

    kx, ky, ans = greens_function_k_dep(energy, h)
    intensity = np.abs(np.trace(ans, axis1=3, axis2=4))
    mask = intensity > 0.02

    if len(energy) == 1:
        masks = RegularGridInterpolator((kx, ky), mask)
    else:
        masks = RegularGridInterpolator((kx, ky, energy), mask)

    print("Masks are ready")

    if vis:

        num_kx = 300
        num_ky = 300
        cell = get_reciprocal_lattice(h)
        kx = np.linspace(-0.0 * cell[0, 0], 0.5 * cell[0, 0], num_kx)
        ky = np.linspace(-0.0 * cell[1, 1], 0.5 * cell[1, 1], num_ky)
        kkx, kky = np.meshgrid(kx, ky)

        if len(masks.grid) == 2:
            kpts = np.column_stack((kkx.ravel(), kky.ravel(), np.zeros(kkx.size)))
            mask_bin = masks(kpts[:, :2]) > 0.2
            plt.pcolormesh(mask_bin.reshape((len(kx), len(ky))), cmap='berlin')
            plt.show()
        elif len(masks.grid) == 3:
            kpts = np.column_stack((kkx.ravel(), kky.ravel(), np.zeros(kkx.size) - 1.4))
            mask_bin = masks(kpts) > 0.2
            plt.pcolormesh(mask_bin.reshape((len(kx), len(ky))), cmap='berlin')
            plt.show()
        else:
            pass

    return masks


def main():

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    fields = np.linspace(0.0, 0.1, 20)
    energy_sparce = np.linspace(-2.0, -0.2, 100)
    energy = np.linspace(-2.0, -0.2, 300)

    trs = np.zeros((len(energy), len(fields)))
    doss = np.zeros((len(energy), len(fields)))

    for j, field in enumerate(fields):
        print("==================")
        print(j, "out of", len(fields))
        print("==================")

        h = make_hamiltonian(0.0, field)
        masks = make_masks(energy_sparce, h, vis=False)
        gf, tr, dos = greens_function_from_tb_with_mask(energy, h, masks)

        trs[:, j] = tr
        doss[:, j] = dos

    np.save("trs.npy", trs)
    np.save("doss.npy", doss)

    plt.plot(energy, trs[:, -1])
    plt.savefig("trs.pdf")

    plt.plot(energy, doss[:, -1])
    plt.savefig("dos.pdf")


if __name__=="__main__":

    main()



