import numpy as np
import nanonet.tb as tb
from test.test_hamiltonian_module import expected_bulk_silicon_band_structure


def test_simple_atomic_chain():
    """ """
    site_energy = -1.0
    coupling = -1.0
    l_const = 1.0

    a = tb.Orbitals('A')
    a.add_orbital(title='s', energy=-1, )

    xyz_file = """1
    H cell
    A       0.0000000000    0.0000000000    0.0000000000
    """
    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -1.0})
    h = tb.HamiltonianSp(xyz=xyz_file, nn_distance=1.1)
    h.initialize()

    PRIMITIVE_CELL = [[0, 0, l_const]]
    h.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 10
    kk = np.linspace(0, 3.14 / l_const, num_points, endpoint=True)

    band_structure = []

    for jj in range(num_points):
        vals, _ = h.diagonalize_periodic_bc([0.0, 0.0, kk[jj]])
        band_structure.append(vals)

    band_structure = np.array(band_structure)

    desired_value = site_energy + 2 * coupling * np.cos(l_const * kk)
    np.testing.assert_allclose(band_structure, desired_value[:, np.newaxis], atol=1e-9)


def test_atomic_chain_two_kinds_of_atoms():
    """ """
    site_energy1 = -1.0
    site_energy2 = -2.0
    coupling = -1.0
    l_const = 2.0

    a = tb.Orbitals('A')
    a.add_orbital(title='s', energy=site_energy1, )
    b = tb.Orbitals('B')
    b.add_orbital(title='s', energy=site_energy2, )

    xyz_file = """2
    H cell
    A       0.0000000000    0.0000000000    0.0000000000
    B       0.0000000000    0.0000000000    1.0000000000
    """
    tb.set_tb_params(PARAMS_A_B={'ss_sigma': coupling})
    h = tb.HamiltonianSp(xyz=xyz_file, nn_distance=1.1)
    h.initialize()

    PRIMITIVE_CELL = [[0, 0, l_const]]
    h.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 10
    kk = np.linspace(0, 3.14 / 2, num_points, endpoint=True)

    band_structure = []

    for jj in range(num_points):
        vals, _ = h.diagonalize_periodic_bc([0.0, 0.0, kk[jj]])
        band_structure.append(vals)

    band_structure = np.array(band_structure)
    desired_value = np.zeros(band_structure.shape)

    b = site_energy1 + site_energy2
    c = site_energy1 * site_energy2 - (2.0 * coupling * np.cos(0.5 * kk * l_const)) ** 2
    desired_value[:, 0] = 0.5 * (b - np.sqrt(b ** 2 - 4.0 * c))
    desired_value[:, 1] = 0.5 * (b + np.sqrt(b ** 2 - 4.0 * c))

    np.testing.assert_allclose(band_structure, desired_value, atol=1e-9)


def test_bulk_silicon():
    """ """
    a_si = 5.50
    PRIMITIVE_CELL = [[0, 0.5 * a_si, 0.5 * a_si],
                      [0.5 * a_si, 0, 0.5 * a_si],
                      [0.5 * a_si, 0.5 * a_si, 0]]

    tb.Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S'}

    xyz_file = """2
    Si2 cell
    Si1       0.0000000000    0.0000000000    0.0000000000
    Si2       1.3750000000    1.3750000000    1.3750000000
    """

    h = tb.HamiltonianSp(xyz=xyz_file, nn_distance=2.5)
    h.initialize()
    h.set_periodic_bc(PRIMITIVE_CELL)

    sym_points = ['L', 'GAMMA', 'X']
    num_points = [10, 25]
    k = tb.get_k_coords(sym_points, num_points, 'Si')
    band_sructure = []

    vals = np.zeros((sum(num_points), h.num_eigs), dtype=np.complex)

    for jj, item in enumerate(k):
        vals[jj, :], _ = h.diagonalize_periodic_bc(item)

    band_structure = np.real(np.array(vals))
    np.testing.assert_allclose(band_structure, expected_bulk_silicon_band_structure()[:,:h.num_eigs], atol=1e-4)


if __name__ == '__main__':
    # test_simple_atomic_chain()
    # test_atomic_chain_two_kinds_of_atoms()
    test_bulk_silicon()
