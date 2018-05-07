import tb
import numpy as np


def test_simple_atomic_chain():

    site_energy = -1.0
    coupling = -1.0
    l_const = 1.0

    a = tb.Atom('A')
    a.add_orbital(title='s', energy=-1, )
    tb.Atom.orbital_sets = {'A': a}

    xyz_file = """1
    H cell
    A       0.0000000000    0.0000000000    0.0000000000
    """
    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -1.0})
    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
    h.initialize()

    PRIMITIVE_CELL = [[0, 0, l_const]]
    h.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 10
    kk = np.linspace(0, 3.14 / l_const, num_points, endpoint=True)

    band_structure = []

    for jj in xrange(num_points):
        vals, _ = h.diagonalize_periodic_bc([0.0, 0.0, kk[jj]])
        band_structure.append(vals)

    band_structure = np.array(band_structure)

    desired_value = site_energy + 2 * coupling * np.cos(l_const * kk)
    np.testing.assert_allclose(band_structure, desired_value[:, np.newaxis], atol=1e-9)


def test_atomic_chain_two_kinds_of_atoms():

    site_energy1 = -1.0
    site_energy2 = -2.0
    coupling = -1.0
    l_const = 2.0

    a = tb.Atom('A')
    a.add_orbital(title='s', energy=site_energy1, )
    b = tb.Atom('B')
    b.add_orbital(title='s', energy=site_energy2, )

    tb.Atom.orbital_sets = {'A': a, 'B': b}

    xyz_file = """2
    H cell
    A       0.0000000000    0.0000000000    0.0000000000
    B       0.0000000000    0.0000000000    1.0000000000
    """
    tb.set_tb_params(PARAMS_A_B={'ss_sigma': coupling})
    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
    h.initialize()

    PRIMITIVE_CELL = [[0, 0, l_const]]
    h.set_periodic_bc(PRIMITIVE_CELL)

    num_points = 10
    kk = np.linspace(0, 3.14 / 2, num_points, endpoint=True)

    band_structure = []

    for jj in xrange(num_points):
        vals, _ = h.diagonalize_periodic_bc([0.0, 0.0, kk[jj]])
        band_structure.append(vals)

    band_structure = np.array(band_structure)
    desired_value = np.zeros(band_structure.shape)

    b = site_energy1 + site_energy2
    c = site_energy1 * site_energy2 - (2.0 * coupling * np.cos(0.5 * kk * l_const))**2
    desired_value[:, 0] = 0.5 * (b - np.sqrt(b ** 2 - 4.0 * c))
    desired_value[:, 1] = 0.5 * (b + np.sqrt(b ** 2 - 4.0 * c))

    np.testing.assert_allclose(band_structure, desired_value, atol=1e-9)


if __name__ == '__main__':

    test_simple_atomic_chain()
    test_atomic_chain_two_kinds_of_atoms()
