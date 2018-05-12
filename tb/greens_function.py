import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg


def surface_greens_function_poles(E, h_l, h_0, h_r):
    """
    Computes eigenvalues and eigenvectors for the complex band structure problem.
    Here, the energy E is a parameter, and the eigenvalues correspond to wave vectors as `exp(ik)`.

    :param E:     energy
    :type E:      float
    :param h_l:   left block of three-block-diagonal Hamiltonian
    :param h_0:   central block of three-block-diagonal Hamiltonian
    :param h_r:   right block of three-block-diagonal Hamiltonian
    :return:      eigenvalues, k, and eigenvectors, U,
    :rtype:       numpy.matrix, numpy.matrix
    """

    main_matrix = np.block([[np.zeros(h_0.shape), np.identity(h_0.shape[0])],
                            [-h_l, E * np.identity(h_0.shape[0]) - h_0]])

    overlap_matrix = np.block([[np.identity(h_0.shape[0]), np.zeros(h_0.shape)],
                               [np.zeros(h_0.shape), h_r]])

    alpha, betha, _, eigenvects, _, _ = linalg.lapack.cggev(main_matrix, overlap_matrix)

    eigenvals = np.zeros(alpha.shape, dtype=np.complex128)

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
    mask1 = np.abs(vals) < 0.999
    mask2 = np.abs(vals) > 1.001
    vals = np.angle(vals)

    vals[mask1] = -5
    vals[mask2] = 5
    ind = np.argsort(vals, kind='mergesort')

    eigenvals = eigenvals[ind]
    eigenvects = eigenvects[:, ind]

    eigenvects = eigenvects[h_0.shape[0]:, :]
    eigenvals = np.matrix(np.diag(eigenvals))
    eigenvects = np.matrix(eigenvects)

    norms = linalg.norm(eigenvects, axis=0)
    norms = np.array([1e30 if np.abs(norm) < 0.000001 else norm for norm in norms])
    eigenvects = eigenvects / norms[np.newaxis, :]

    return eigenvals, eigenvects


def group_velocity(eigenvector, eigenvalue, h_r):
    """
    Computes the group velocity of wave packets from their wave vectors

    :param eigenvector:       eigenvector
    :type eigenvector:        numpy.matrix(dtype=numpy.complex)
    :param eigenvalue:        eigenvalue
    :type eigenvector:        numpy.complex
    :param h_r:               coupling Hamiltonian
    :type h_r:                numpy.matrix
    :return:                  group velocity for a pair consisting of an eigenvector and an eigenvalue
    """

    return np.imag(eigenvector.H * h_r * eigenvalue * eigenvector)


def iterate_gf(E, h_0, h_l, h_r, gf, num_iter):
    """
    Iterate a self-energy to achieve self-consistency

    :param E:
    :param h_0:
    :param h_l:
    :param h_r:
    :param gf:
    :param num_iter:
    :return:
    """

    for j in xrange(num_iter):
        gf = h_r * np.linalg.pinv(E * np.identity(h_0.shape[0]) - h_0 - gf) * h_l

    return gf


def surface_greens_function(E, h_l, h_0, h_r):
    """
    The function surface self-energies. To do that, it provides classification and
    sorting of the eigenvalues and eigenvectors after the eigenvalue decomposition.
    The sorting procedure is described in [M. Wimmer, Quantum transport in nanostructures:
    From computational concepts to spintronics in graphene and magnetic tunnel junctions,
    2009, ISBN-9783868450255]. The algorithm puts left-propagating solutions into upper-left
    block and right-propagating solutions - into the lower-right block.
    The sort is performed by absolute value of the eigenvalues. For those whose
    absolute value equals one, classification is performed computing their group velocity
    using eigenvectors.

    :param E:
    :param h_l:
    :param h_0:
    :param h_r:

    :return:
    """

    vals, vects = surface_greens_function_poles(E, h_l, h_0, h_r)
    vals = np.diag(vals)

    u_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    u_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))

    alpha = 0.001

    for j in range(h_0.shape[0]):
        if np.abs(vals[j]) > 1.0 + alpha:

            lambda_left[j, j] = vals[j]
            u_left[:, j] = vects[:, j]

            lambda_right[j, j] = vals[-j + 2*h_0.shape[0]-1]
            u_right[:, j] = vects[:, -j + 2*h_0.shape[0]-1]

        elif np.abs(vals[j]) < 1.0 - alpha:
            lambda_right[j, j] = vals[j]
            u_right[:, j] = vects[:, j]

            lambda_left[j, j] = vals[-j + 2*h_0.shape[0]-1]
            u_left[:, j] = vects[:, -j + 2*h_0.shape[0]-1]

        else:

            gv = group_velocity(vects[:, j], vals[j], h_r)
            # print("Group velocity is ", gv, np.angle(vals[j]))
            if gv > 0:

                lambda_left[j, j] = vals[j]
                u_left[:, j] = vects[:, j]

                lambda_right[j, j] = vals[-j + 2*h_0.shape[0]-1]
                u_right[:, j] = vects[:, -j + 2*h_0.shape[0]-1]

            else:
                lambda_right[j, j] = vals[j]
                u_right[:, j] = vects[:, j]

                lambda_left[j, j] = vals[-j + 2*h_0.shape[0]-1]
                u_left[:, j] = vects[:, -j + 2*h_0.shape[0]-1]

    sgf_l = h_r * u_right * lambda_right * np.linalg.pinv(u_right)
    sgf_r = h_l * u_left * lambda_right * np.linalg.pinv(u_left)

    return iterate_gf(E, h_0, h_l, h_r, sgf_l, 0), iterate_gf(E, h_0, h_r, h_l, sgf_r, 0), \
           lambda_right, lambda_left, vals
