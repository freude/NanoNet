import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg


def sort_eigs(alpha, betha, Z, h_l, h_r, flag):
    """The function provides classification and sorting of the eigenvalues
    and eigenvectors. The sorting procedure is described in
    [M. Wimmer, Quantum transport in nanostructures: From computational concepts to
    spintronics in graphene and magnetic tunnel junctions, 2009, ISBN-9783868450255]
    The algorithm puts left-propagating solutions into upper-left block and right-propagating
    solutions - into the lower-right block.
    The sort is performed by absolute value of the eigenvalues. For those whose
    absolute value equals one, classification is performed computing their group velocity
    using eigenvectors. The function also orthogonalizes the eigenvectors contained in the matrix Z
    that correspond to degenerate eigenvalues.

    Parameters
    ----------
    alpha :
        param betha:
    Z :
        param h_l:
    h_r :
        return:
    betha :
        
    h_l :
        
    flag :
        

    Returns
    -------

    """

    margin = 1e-3
    angle_margin = 1e-4
    ans = []

    eigenvals = alpha / betha

    inds = np.where((np.abs(eigenvals) < 1.0 + margin) * (np.abs(eigenvals) > 1.0 - margin))[0]
    inds_deg = []

    for j, j1 in enumerate(inds[:-1]):
        inds_deg.append([j1])
        for j2 in inds[j + 1:]:
            if np.abs(np.real(eigenvals[j1] - eigenvals[j2])) < angle_margin and \
                    np.abs(np.imag(eigenvals[j1] - eigenvals[j2])) < angle_margin:
                inds_deg[j].append(j2)

    inds_deg = [i for i in inds_deg if len(i) > 1]

    for item in inds_deg:
        phi = Z[h_r.shape[0]:, item]
        oper = 1j * np.dot(h_r, eigenvals[item[0]]) - np.dot(h_l, eigenvals[item[0]].conj())
        _, vects = np.linalg.eig(np.dot(np.dot(phi.conj().T, oper), phi))
        print(vects)
        Z[h_r.shape[0]:, item] = np.dot(phi, vects)

    phi = Z

    for j, item in enumerate(zip(alpha, betha)):

        if np.abs(item[1]) != 0.0:
            eigenval = item[0] / item[1]
        else:
            eigenval = 1e21

        if np.abs(eigenval) > 1.0 + margin:
            ans.append(False)
        elif np.abs(eigenval) < 1.0 - margin:
            ans.append(True)
        else:
            gv = np.imag(2 * np.dot(np.dot(phi[h_r.shape[0]:, j].conj().T, h_r), phi[h_r.shape[0]:, j]))
            # gv = group_velocity(phi[h_r.shape[0]:, j], eigenval, h_r)

            if flag:
                gv = group_velocity(phi[h_r.shape[0]:, j], np.conj(eigenval), h_r)

            print("Group velocity is ", gv, np.angle(eigenval))
            if gv > 0:
                ans.append(False)
            else:
                ans.append(True)

    return ans


def iterate_gf(E, h_0, h_l, h_r, gf, num_iter):
    """

    Parameters
    ----------
    E :
        
    h_0 :
        
    h_l :
        
    h_r :
        
    gf :
        
    num_iter :
        

    Returns
    -------

    """
    for j in range(num_iter):
        gf = h_r * np.linalg.pinv(E * np.identity(h_0.shape[0]) - h_0 - gf) * h_l

    return gf


def surface_greens_function_poles(E, h_l, h_0, h_r):
    """Computes eigenvalues and eigenvectors for the complex band structure problem.
    Here, the energy E is a parameter, and the eigenvalues correspond to wave vectors as `exp(ik)`.

    Parameters
    ----------
    E : float
        energy
    h_l :
        left block of three-block-diagonal Hamiltonian
    h_0 :
        central block of three-block-diagonal Hamiltonian
    h_r :
        right block of three-block-diagonal Hamiltonian

    Returns
    -------
    numpy.matrix, numpy.matrix
        eigenvalues, k, and eigenvectors, U,

    """

    main_matrix = np.block([[np.zeros(h_0.shape), np.identity(h_0.shape[0])],
                            [-h_l, E * np.identity(h_0.shape[0]) - h_0]])

    overlap_matrix = np.block([[np.identity(h_0.shape[0]), np.zeros(h_0.shape)],
                               [np.zeros(h_0.shape), h_r]])

    # alpha, betha, _, eigenvects, _, _ = linalg.lapack.cggev(main_matrix, overlap_matrix)
    AA, BB, eigenvects, _ = linalg.qz(main_matrix, overlap_matrix)
    alpha = np.diag(AA)
    betha = np.diag(BB)

    eigenvals = np.zeros(alpha.shape, dtype=np.complex)

    for j, item in enumerate(zip(alpha, betha)):
        if np.abs(item[1]) != 0.0:
            eigenvals[j] = item[0] / item[1]
        else:
            eigenvals[j] = 1e10

    # sort absolute values
    ind = np.argsort(np.abs(eigenvals))
    eigenvals = eigenvals[ind]
    eigenvects = eigenvects[:, ind]

    vals = np.copy(eigenvals)
    roll = np.where((0.999 < np.abs(vals)) * (np.abs(vals) < 1.001))

    if np.size(roll) != 0:
        roll = h_0.shape[0] - int(round(np.mean(roll)))
        if roll != 0:
            print(roll)
            eigenvals = np.roll(eigenvals, roll)
            eigenvects = np.roll(eigenvects, roll, axis=1)
            eigenvals[:abs(roll)] = 0.0
            eigenvals[-abs(roll):] = 1e10
            vals = np.copy(eigenvals)

    mask1 = np.abs(vals) < 0.999
    mask2 = np.abs(vals) > 1.001
    vals = np.angle(vals)

    vals[mask1] = -5
    vals[mask2] = 5
    ind = np.argsort(vals, kind='mergesort')

    eigenvals = eigenvals[ind]
    eigenvects = eigenvects[:, ind]

    eigenvects = eigenvects[h_0.shape[0]:, :]
    eigenvals = np.diag(eigenvals)

    norms = linalg.norm(eigenvects, axis=0)
    norms = np.array([1e30 if np.abs(norm) < 0.000001 else norm for norm in norms])
    eigenvects = eigenvects / norms[np.newaxis, :]

    return eigenvals, eigenvects


def surface_greens_function_poles_Shur(E, h_l, h_0, h_r):
    """Computes eigenvalues and eigenvectors for the complex band structure problem.
    Here, the energy E is a parameter, and the eigenvalues correspond to wave vectors as `exp(ik)`.

    Parameters
    ----------
    E : float
        energy
    h_l :
        left block of three-block-diagonal Hamiltonian
    h_0 :
        central block of three-block-diagonal Hamiltonian
    h_r :
        right block of three-block-diagonal Hamiltonian

    Returns
    -------
    numpy.matrix, numpy.matrix
        eigenvalues, k, and eigenvectors, U,

    """

    main_matrix = np.block([[np.zeros(h_0.shape), np.identity(h_0.shape[0])],
                            [-h_l, E * np.identity(h_0.shape[0]) - h_0]]).astype(np.complex64)

    overlap_matrix = np.block([[np.identity(h_0.shape[0]), np.zeros(h_0.shape)],
                               [np.zeros(h_0.shape), h_r]]).astype(np.complex64)

    sort = lambda a, b, z: sort_eigs(a, b, z, h_l, h_r, False)

    AA, BB, alpha, betha, eigv_right, eigv_left = linalg.ordqz(main_matrix, overlap_matrix,
                                                               output='complex', sort='iuc')

    main_matrix1 = np.block([[np.zeros(h_0.shape), np.identity(h_0.shape[0])],
                             [-h_r, E * np.identity(h_0.shape[0]) - h_0]]).astype(np.complex64)

    overlap_matrix1 = np.block([[np.identity(h_0.shape[0]), np.zeros(h_0.shape)],
                                [np.zeros(h_0.shape), h_l]]).astype(np.complex64)

    sort1 = lambda a, b, z: sort_eigs(a, b, z, h_r, h_l, True)

    AA1, BB1, alpha1, betha1, eigv_right1, eigv_left1 = linalg.ordqz(main_matrix1, overlap_matrix1,
                                                                     output='complex',
                                                                     sort='iuc')

    return h_r.dot(eigv_left[h_0.shape[0]:, :h_0.shape[0]]).dot(np.linalg.pinv(eigv_left[:h_0.shape[0], :h_0.shape[0]])), \
           h_l.dot(eigv_right[h_0.shape[0]:, :h_0.shape[0]]).dot(np.linalg.pinv(eigv_right[:h_0.shape[0], :h_0.shape[0]]))


def group_velocity(eigenvector, eigenvalue, h_r):
    """Computes the group velocity of wave packets from their wave vectors

    Parameters
    ----------
    eigenvector :
        param eigenvalue:
    h_r :
        return:
    eigenvalue :
        

    Returns
    -------

    """

    return np.imag(np.dot(np.dot(np.dot(eigenvector.conj().T, h_r), eigenvalue), eigenvector))


def surface_greens_function(E, h_l, h_0, h_r, method='Shur'):
    """Computes eigenvalues and eigenvectors for the complex band structure problem.
    Here energy E is a parameter, adn the eigenvalues are wave vectors.

    Parameters
    ----------
    E : float
        energy
    h_l :
        left block of three-block-diagonal Hamiltonian
    h_0 :
        central block of three-block-diagonal Hamiltonian
    h_r :
        right block of three-block-diagonal Hamiltonian
    method :
         (Default value = 'Shur')

    Returns
    -------
    type
        surface Green's function, for left and right sides

    """

    if method == 'Shur':
        sgf_l, sgf_r = surface_greens_function_poles_Shur(E, h_l, h_0, h_r)
    else:
        vals, vects = surface_greens_function_poles(E, h_l, h_0, h_r)
        vals = np.diag(vals)

        u_right = np.zeros(h_0.shape, dtype=np.complex)
        u_left = np.zeros(h_0.shape, dtype=np.complex)
        lambda_right = np.zeros(h_0.shape, dtype=np.complex)
        lambda_left = np.zeros(h_0.shape, dtype=np.complex)

        alpha = 0.001

        for j in range(h_0.shape[0]):
            if np.abs(vals[j]) > 1.0 + alpha:

                lambda_left[j, j] = vals[j]
                u_left[:, j] = vects[:, j]

                lambda_right[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
                u_right[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

            elif np.abs(vals[j]) < 1.0 - alpha:
                lambda_right[j, j] = vals[j]
                u_right[:, j] = vects[:, j]

                lambda_left[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
                u_left[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

            else:

                gv = group_velocity(vects[:, j], vals[j], h_r)
                # ind = np.argmin(np.abs(np.angle(vals[h_0.shape[0]:]) + np.angle(vals[j])))
                print("Group velocity is ", gv, np.angle(vals[j]))

                if gv > 0:

                    lambda_left[j, j] = vals[j]
                    u_left[:, j] = vects[:, j]

                    lambda_right[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
                    u_right[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

                else:
                    lambda_right[j, j] = vals[j]
                    u_right[:, j] = vects[:, j]

                    lambda_left[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
                    u_left[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

                # lambda_right[j, j] = vals[j]
                # u_right[:, j] = vects[:, j]
                #
                # lambda_left[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
                # u_left[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

        sgf_l = h_r.dot(u_right).dot(lambda_right).dot(np.linalg.pinv(u_right))
        sgf_r = h_l.dot(u_left).dot(lambda_right).dot(np.linalg.pinv(u_left))

        # sgf_l = u_right[h_0.shape[0]:, :] * np.linalg.pinv(u_right[:h_0.shape[0], :])
        # sgf_r = h_l * u_left * lambda_right * np.linalg.pinv(u_left)

    return iterate_gf(E, h_0, h_l, h_r, sgf_l, 2), iterate_gf(E, h_0, h_r, h_l, sgf_r, 2)#, \
           #lambda_right, lambda_left, vals

    # return h_r * u_right * lambda_right * np.linalg.pinv(u_right), \
    #        h_l * u_left * lambda_right * np.linalg.pinv(u_left), \
    #        lambda_right, lambda_left, vals


# functions below are essentially functional tests

def main():
    """ """
    import sys
    sys.path.insert(0, '/home/mk/TB_project/tb')
    import nanonet.tb as tb

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)
    b = tb.Orbitals('B')
    b.add_orbital('s', -0.5)
    c = tb.Orbitals('C')
    c.add_orbital('s', -0.3)

    tb.Orbitals.orbital_sets = {'A': a, 'B': b, 'C': c}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5},
                     PARAMS_B_B={'ss_sigma': -0.5},
                     PARAMS_A_B={'ss_sigma': -0.5},
                     PARAMS_B_C={'ss_sigma': -0.5},
                     PARAMS_A_C={'ss_sigma': -0.5})

    xyz_file = """4
    H cell
    A1       0.0000000000    0.0000000000    0.0000000000
    B2       0.0000000000    0.0000000000    1.0000000000
    A2       0.0000000000    1.0000000000    0.0000000000
    B3       0.0000000000    1.0000000000    1.0000000000
    """

    # xyz_file = """1
    # H cell
    # A1       0.0000000000    0.0000000000    0.0000000000
    # """

    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
    h.initialize()
    h.set_periodic_bc([[0, 0, 2.0]])
    h_l, h_0, h_r = h.get_hamiltonians()

    energy = np.linspace(-3.0, 1.5, 700)

    sgf_l = []
    sgf_r = []
    sgf_l1 = []
    sgf_r1 = []

    for E in energy:
        print("============Eigenval decompositions============")
        L, R = surface_greens_function(E, h_l, h_0, h_r)
        print("==============Schur decomposition==============")
        L1, R1 = surface_greens_function_poles_Shur(E, h_l, h_0, h_r)
        sgf_l.append(L)
        sgf_r.append(R)
        sgf_l1.append(L1)
        sgf_r1.append(R1)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    sgf_l1 = np.array(sgf_l1)
    sgf_r1 = np.array(sgf_r1)

    num_sites = h_0.shape[0]
    gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l1 - sgf_r1)

    dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

    tr = np.zeros((energy.shape[0]), dtype=np.complex)

    for j, E in enumerate(energy):
        gf0 = gf[j, :, :]
        gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
        gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
        tr[j] = np.real(np.trace(gamma_l.dot(gf0).dot(gamma_r).dot(gf0.conj().T)))
        dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))
    print(sgf_l.shape)


def main1():
    """ """
    import sys
    sys.path.insert(0, '/home/mk/TB_project/tb')
    import nanonet.tb as tb

    tb.Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    h = tb.Hamiltonian(xyz='/home/mk/NEGF_project/SiNW.xyz', nn_distance=2.4)
    h.initialize()
    h.set_periodic_bc([[0, 0, 5.50]])
    h_l, h_0, h_r = h.get_hamiltonians()

    # energy = np.linspace(2.07, 2.3, 50)
    # energy = np.linspace(2.07, 2.3, 50) + 0.2
    # energy = np.linspace(-1.0, -0.3, 200)
    energy = np.concatenate((np.linspace(-0.8, -0.3, 100), np.linspace(2.05, 2.5, 100)))
    # energy = np.linspace(2.3950, 2.4050, 20)

    # energy = energy[20:35]

    sgf_l = []
    sgf_r = []
    factor = []
    factor1 = []
    factor2 = []

    num_sites = h_0.shape[0]

    for j, E in enumerate(energy):
        # L, R = surface_greens_function_poles_Shur(j, E, h_l, h_0, h_r)
        L, R = surface_greens_function(E, h_l, h_0, h_r)

        test_gf = E * np.identity(num_sites) - h_0 - L - R

        metrics = np.linalg.cond(test_gf)
        print("{} of {}: energy is {}, metrics is {}".format(j + 1, energy.shape[0], E, metrics))

        # if metrics > 15000:
        #     R = iterate_gf(E, h_0, h_l, h_r, R, 1)
        #     L = iterate_gf(E, h_0, h_l, h_r, L, 1)

        sgf_l.append(L)
        sgf_r.append(R)
        # factor.append(phase)
        # factor1.append(phase1)
        # factor2.append(phase2)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)
    factor = np.array(factor)
    factor1 = np.array(factor1)
    factor2 = np.array(factor2)

    gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

    # gf = regularize_gf(gf)

    dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

    tr = np.zeros((energy.shape[0]), dtype=np.complex)
    dos = np.zeros((energy.shape[0]), dtype=np.complex)

    for j, E in enumerate(energy):
        gf0 = gf[j, :, :]
        gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
        gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
        dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))
        tr[j] = np.real(np.trace(gamma_l.dot(gf0).dot(gamma_r).dot(gf0.conj().T)))

    ax = plt.axes()
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('DOS')
    ax.plot(energy, dos)
    plt.show()

    ax = plt.axes()
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Transmission coefficient (a.u.)')
    ax.plot(energy, tr)
    plt.show()

    plt.plot(dos)
    plt.show()

    print(sgf_l.shape)


def regularize_gf(gf):
    """

    Parameters
    ----------
    gf :
        

    Returns
    -------

    """
    cutoff = 1e3

    if np.abs(np.sum(gf[0, :, :])) > cutoff:
        j = 0
        while np.abs(np.sum(gf[j, :, :])) > cutoff:
            j += 1
        gf[0, :, :] = gf[j, :, :]

    for j in range(gf.shape[0]):
        if np.abs(np.sum(gf[j, :, :])) > cutoff:
            gf[j, :, :] = gf[j - 1, :, :]

    return gf


def inverse_bs_problem():
    """ """
    import sys
    sys.path.insert(0, '/home/mk/TB_project/tb')
    import nanonet.tb as tb

    # a = tb.Atom('A')
    # a.add_orbital('s', -0.7)
    # b = tb.Atom('B')
    # b.add_orbital('s', -0.5)
    # c = tb.Atom('C')
    # c.add_orbital('s', -0.3)
    #
    # tb.Atom.orbital_sets = {'A': a, 'B': b, 'C': c}
    #
    # tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5},
    #                  PARAMS_B_B={'ss_sigma': -0.5},
    #                  PARAMS_C_C={'ss_sigma': -0.5},
    #                  PARAMS_A_B={'ss_sigma': -0.5},
    #                  PARAMS_B_C={'ss_sigma': -0.5},
    #                  PARAMS_A_C={'ss_sigma': -0.5})
    #
    # xyz_file = """6
    # H cell
    # A1       0.0000000000    0.0000000000    0.0000000000
    # B2       0.0000000000    0.0000000000    1.0000000000
    # C3       0.0000000000    0.0000000000    2.0000000000
    # A4       0.0000000000    1.0000000000    0.0000000000
    # B5       0.0000000000    1.0000000000    1.0000000000
    # C6       0.0000000000    1.0000000000    2.0000000000
    # """
    #
    # h = tb.Hamiltonian(xyz=xyz_file, nn_distance=1.1)
    # h.initialize()
    # h.set_periodic_bc([[0, 0, 3.0]])

    tb.Orbitals.orbital_sets = {'Si': 'SiliconSP3D5S', 'H': 'HydrogenS'}
    h = tb.Hamiltonian(xyz='/home/mk/TB_project/input_samples/SiNW.xyz', nn_distance=2.4)
    h.initialize()
    h.set_periodic_bc([[0, 0, 5.50]])

    h_l, h_0, h_r = h.get_hamiltonians()

    # energy = np.linspace(2.13, 2.15, 20)
    # energy = np.linspace(-2.1, 1.0, 50)
    energy = np.linspace(2.05, 2.5, 200)

    # energy = np.linspace(2.07, 2.3, 50) + 0.2
    # energy = np.linspace(2.3950, 2.4050, 20)

    # energy = energy[20:]

    eigs = []

    for E in energy:

        print(E)

        vals, vects = surface_greens_function_poles(E, h_l, h_0, h_r)

        vals = np.diag(vals)
        vals.setflags(write=1)

        for j, v in enumerate(vals):

            if np.abs(np.absolute(v) - 1.0) > 0.01:
                vals[j] = float('nan')
            else:
                vals[j] = np.angle(v)
                print("The element number is", j, vals[j])

        eigs.append(vals)

    plt.plot(energy, np.array(eigs), 'o')
    plt.show()


def main2():
    """ """
    import sys
    sys.path.insert(0, '/home/mk/TB_project/tb')
    import nanonet.tb as tb

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)

    tb.Orbitals.orbital_sets = {'A': a}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5},
                     PARAMS_B_B={'ss_sigma': -0.5},
                     PARAMS_A_B={'ss_sigma': -0.5},
                     PARAMS_B_C={'ss_sigma': -0.5},
                     PARAMS_A_C={'ss_sigma': -0.5})

    xyz_file = """1
    H cell
    A1       0.0000000000    0.0000000000    0.0000000000                                                                                                      
    """

    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=2.1)
    h.initialize()
    h.set_periodic_bc([[0, 0, 1.0]])
    h_l, h_0, h_r = h.get_hamiltonians()

    energy = np.concatenate(np.linspace(-3.0, 1.5, 150), np.linspace(2.05, 2.5, 150))

    sgf_l = []
    sgf_r = []

    for E in energy:
        L, R = surface_greens_function(E, h_l, h_0, h_r)
        # L, R = surface_greens_function_poles_Shur(E, h_l, h_0, h_r)
        sgf_l.append(L)
        sgf_r.append(R)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    num_sites = h_0.shape[0]
    gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

    dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

    tr = np.zeros((energy.shape[0]), dtype=np.complex)

    for j, E in enumerate(energy):
        gf0 = gf[j, :, :]
        gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
        gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
        tr[j] = np.real(np.trace(gamma_l.dot(gf0).dot(gamma_r).dot(gf0.conj().T)))
        dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))
    print(sgf_l.shape)


def main3():
    """ """
    import sys
    sys.path.insert(0, '/home/mk/TB_project/tb')
    import nanonet.tb as tb

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)

    tb.Orbitals.orbital_sets = {'A': a}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5},
                     PARAMS_B_B={'ss_sigma': -0.5},
                     PARAMS_A_B={'ss_sigma': -0.5},
                     PARAMS_B_C={'ss_sigma': -0.5},
                     PARAMS_A_C={'ss_sigma': -0.5})

    xyz_file = """1
    H cell
    A1       0.0000000000    0.0000000000    0.0000000000                                                                                                      
    """

    h = tb.Hamiltonian(xyz=xyz_file, nn_distance=2.1)
    h.initialize()
    h.set_periodic_bc([[0, 0, 1.0]])
    h_l, h_0, h_r = h.get_hamiltonians()

    energy = np.linspace(-3.0, 1.5, 700)

    sgf_l = []
    sgf_r = []

    for E in energy:
        L, R = surface_greens_function(E, h_l, h_0, h_r)
        # L, R = surface_greens_function_poles_Shur(E, h_l, h_0, h_r)
        sgf_l.append(L)
        sgf_r.append(R)

    sgf_l = np.array(sgf_l)
    sgf_r = np.array(sgf_r)

    num_sites = h_0.shape[0]
    gf = np.linalg.pinv(np.multiply.outer(energy, np.identity(num_sites)) - h_0 - sgf_l - sgf_r)

    dos = -np.trace(np.imag(gf), axis1=1, axis2=2)

    tr = np.zeros((energy.shape[0]), dtype=np.complex)

    for j, E in enumerate(energy):
        gf0 = gf[j, :, :]
        gamma_l = 1j * (sgf_l[j, :, :] - sgf_l[j, :, :].conj().T)
        gamma_r = 1j * (sgf_r[j, :, :] - sgf_r[j, :, :].conj().T)
        tr[j] = np.real(np.trace(gamma_l.dot(gf0).dot(gamma_r).dot(gf0.conj().T)))
        dos[j] = np.real(np.trace(1j * (gf0 - gf0.conj().T)))

    print(sgf_l.shape)


if __name__ == "__main__":
    # inverse_bs_problem()
    # main()
    main1()
    # main2()
