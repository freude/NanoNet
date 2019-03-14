"""
The module contains functions that computes Green's functions and their poles
"""
from __future__ import print_function, division
import pickle
from scipy.optimize import minimize
import os.path
import numpy as np
import scipy.linalg as linalg


# def object_function(vec, energy, initial_basis, extended_basis, h_0, h_0_reduced):
#
#     for eee in energy:
#
#         aaa = 1 + \
#               extended_basis.H *\
#               h_0 *\
#               initial_basis *\
#               np.pinv((z - h_0_reduced)**2) *\
#               initial_basis.H *\
#               h_0 *\
#               extended_basis
#
#         bbb = 1* z -\
#               extended_basis.H *\
#               h_0 *\
#               extended_basis - \
#               extended_basis.H * \
#               h_0 * \
#               initial_basis * \
#               np.pinv(z - h_0_reduced) * \
#               initial_basis.H * \
#               h_0 * \
#               extended_basis
#
#     return fff
def object_function1(vec, energy, init_basis, extended_basis, h_l, h_0, h_r, num_of_states):

    vec = np.matrix(vec)
    extended_basis1 = np.array(1.0 / np.sqrt(vec * vec.H)) * np.array(extended_basis * vec.T)
    extended_basis1 = np.hstack((init_basis, extended_basis1))

    h_l_reduced = extended_basis1.H * h_l * extended_basis1
    h_0_reduced = extended_basis1.H * h_0 * extended_basis1
    h_r_reduced = extended_basis1.H * h_r * extended_basis1

    _, _, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    print(num_of_states1 - num_of_states, ' : ', vec)

    return num_of_states1 - num_of_states + (vec * vec.H - 1.0) ** 2


def object_function(vec, energy, init_basis, extended_basis, h_l, h_0, h_r, num_of_states):

    vec = np.matrix(vec)
    extended_basis1 = np.array(1.0 / np.sqrt(vec * vec.H)) * np.array(extended_basis * vec.T)
    extended_basis1 = np.hstack((init_basis, extended_basis1))

    h_l_reduced = extended_basis1.H * h_l * extended_basis1
    h_0_reduced = extended_basis1.H * h_0 * extended_basis1
    h_r_reduced = extended_basis1.H * h_r * extended_basis1

    _, _, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    print(num_of_states1 - num_of_states, ' : ', vec)

    return num_of_states1 - num_of_states + (vec * vec.H - 1.0) ** 2


def bs(E, h_l, h_0, h_r):

    h_list = [h_l, h_0 - E * np.identity(h_0.shape[0]), h_r]
    vals, vects = surface_greens_function_poles(h_list)
    vals = np.diag(vals)

    vals_for_plot = vals.copy()

    acc = 0.001

    vals_for_plot = np.angle(vals_for_plot)
    vals_for_plot[np.abs(np.abs(vals) - 1.0) > acc] = np.nan
    inds = np.where(np.abs(np.abs(vals) - 1.0) <= acc)[0]
    vects = vects[:, inds]
    vals = np.angle(vals[inds])

    return vals, vects, vals_for_plot


def bs_vs_e(energy, h_l, h_0, h_r):

    init_basis = []
    vals_for_plot = []

    for E in energy:
        # print(E)
        _, vec, val_for_plot = bs(E, h_l, h_0, h_r)
        vals_for_plot.append(val_for_plot)
        if vec.size > 0:
            init_basis.append(vec)

    vals_for_plot = np.array(vals_for_plot)

    num_of_states = vals_for_plot[::4, :].size - np.count_nonzero(np.isnan(vals_for_plot[::4, :]))

    init_basis = np.matrix(np.hstack(tuple(init_basis)))

    return init_basis, vals_for_plot, num_of_states


def surface_greens_function_poles(h_list):
    """
    Computes eigenvalues and eigenvectors for the complex band structure problem.
    The eigenvalues correspond to the wave vectors as `exp(ik)`.

    :param h_list:   list of the Hamiltonian blocks - blocks describes coupling
                     with left-side neighbours, Hamiltonian of the side and
                     coupling with right-side neighbours
    :return:         eigenvalues, k, and eigenvectors, U,
    :rtype:          numpy.matrix, numpy.matrix
    """

    # linearize polynomial eigenvalue problem
    pr_order = len(h_list) - 1
    matix_size = h_list[0].shape[0]
    full_matrix_size = pr_order * matix_size
    identity = np.identity(matix_size)

    main_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex)
    overlap_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex)

    for j in range(pr_order):

        main_matrix[(pr_order - 1) * matix_size:pr_order * matix_size,
                    j * matix_size:(j + 1) * matix_size] = -h_list[j]

        if j == pr_order - 1:
            overlap_matrix[j * matix_size:(j + 1) * matix_size,
                           j * matix_size:(j + 1) * matix_size] = h_list[pr_order]
        else:
            overlap_matrix[j * matix_size:(j + 1) * matix_size,
                           j * matix_size:(j + 1) * matix_size] = identity
            main_matrix[j * matix_size:(j + 1) * matix_size,
                        (j + 1) * matix_size:(j + 2) * matix_size] = identity

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

    eigenvects = eigenvects[matix_size:, :]
    eigenvals = np.matrix(np.diag(eigenvals))
    eigenvects = np.matrix(eigenvects)

    norms = linalg.norm(eigenvects, axis=0)
    norms = np.array([1e30 if np.abs(norm) < 0.000001 else norm for norm in norms])
    eigenvects = eigenvects / norms[np.newaxis, :]

    return eigenvals, eigenvects


def group_velocity(eigenvector, eigenvalue, h_r):
    """
    Computes the group velocity of wave packets

    :param eigenvector:       eigenvector
    :type eigenvector:        numpy.matrix(dtype=numpy.complex)
    :param eigenvalue:        eigenvalue
    :type eigenvector:        numpy.complex
    :param h_r:               coupling Hamiltonian
    :type h_r:                numpy.matrix
    :return:                  group velocity for a pair consisting of
                              an eigenvector and an eigenvalue
    """

    return np.imag(eigenvector.H * h_r * eigenvalue * eigenvector)


def iterate_gf(E, h_0, h_l, h_r, gf, num_iter):
    """
    Iterates a self-energy to achieve self-consistency

    :param E:
    :param h_0:
    :param h_l:
    :param h_r:
    :param gf:
    :param num_iter:
    :return:
    """

    for _ in range(num_iter):
        gf = h_r * np.linalg.pinv(E * np.identity(h_0.shape[0]) - h_0 - gf) * h_l

    return gf


def surface_greens_function(E, h_l, h_0, h_r, iterate=False):
    """
    Computes surface self-energies using the eigenvalue decomposition.
    The procedure is described in
    [M. Wimmer, Quantum transport in nanostructures: From computational concepts
    to spintronics in graphene and magnetic tunnel junctions, 2009, ISBN-9783868450255].

    :param E:         energy array
    :param h_l:       left-side coupling Hamiltonian
    :param h_0:       channel Hamiltonian
    :param h_r:       right-side coupling Hamiltonian
    :param iterate:   iterate to stabilize TB matrix

    :return:          left- and right-side self-energies
    """

    h_list = [h_l, h_0 - E * np.identity(h_0.shape[0]), h_r]
    vals, vects = surface_greens_function_poles(h_list)
    vals = np.diag(vals)

    u_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    u_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))

    alpha = 0.0001

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

    if iterate:
        return iterate_gf(E, h_0, h_l, h_r, sgf_l, 2), iterate_gf(E, h_0, h_r, h_l, sgf_r, 2)

    return iterate_gf(E, h_0, h_l, h_r, sgf_l, 0), iterate_gf(E, h_0, h_r, h_l, sgf_r, 0)


def reduce_mode_space(energy, h_l, h_0, h_r, thr):

    # energy = np.linspace(2.0, 3.7, 50)

    label = '_' + "{0:.2f}".format(np.min(energy)) + '_' + "{0:.2f}".format(np.max(energy)) + '_' + str(len(energy))
    first_file = 'init_basis'+label+'.pkl'
    second_file = 'vals_for_plot'+label+'.pkl'

    if os.path.isfile(first_file) and os.path.isfile(second_file):
        # unpickle
        with open(first_file, 'rb') as infile:
            init_basis = pickle.load(infile)
        with open(second_file, 'rb') as infile:
            vals_for_plot = pickle.load(infile)

        num_of_states = vals_for_plot[::4, :].size - np.count_nonzero(np.isnan(vals_for_plot[::4, :]))
    else:
        init_basis, vals_for_plot, num_of_states = bs_vs_e(energy, h_l, h_0, h_r)
        # pickle
        with open(first_file, 'wb') as outfile:
            pickle.dump(init_basis, outfile)
        with open(second_file, 'wb') as outfile:
            pickle.dump(vals_for_plot, outfile)

    # orthogonalize initial basis

    eee, vvv = np.linalg.eig(init_basis.H * init_basis)
    init_basis = init_basis * vvv * np.matrix(np.diag(1.0/np.sqrt(eee)))
    init_basis = init_basis[:, np.where(eee > thr)[0]]

    # test reduced mode space
    h_l_reduced = init_basis.H * h_l * init_basis
    h_0_reduced = init_basis.H * h_0 * init_basis
    h_r_reduced = init_basis.H * h_r * init_basis

    _, vals_for_plot_1, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    while num_of_states != num_of_states1:

        extended_basis = (1.0 - init_basis * init_basis.H) * h_0 * init_basis
        extended_basis1 = (1.0 - init_basis * init_basis.H) * (h_l + h_r) * init_basis
        extended_basis = np.matrix(np.hstack((extended_basis, extended_basis1)))
        eee, vvv = np.linalg.eig(extended_basis.H * extended_basis)
        extended_basis = extended_basis * vvv * np.matrix(np.diag(1.0 / np.sqrt(eee)))
        extended_basis = extended_basis[:, np.where(eee > thr)[0]]
        x0 = 0.5*np.ones(extended_basis.shape[1])
        res = minimize(object_function,
                       x0,
                       args=(energy, init_basis, extended_basis, h_l, h_0, h_r, num_of_states),
                       method='COBYLA', options={'maxiter': 300})
        vec = np.matrix(res.x)

        extended_basis = np.matrix(np.array(1.0 / np.sqrt(vec * vec.H)) * np.array(extended_basis * vec.T))
        init_basis = np.hstack((init_basis, extended_basis))

        # test reduced mode space
        h_l_reduced = init_basis.H * h_l * init_basis
        h_0_reduced = init_basis.H * h_0 * init_basis
        h_r_reduced = init_basis.H * h_r * init_basis

        _, vals_for_plot2, num_of_states1 = bs_vs_e(energy, h_l_reduced, h_0_reduced, h_r_reduced)

    return h_l_reduced, h_0_reduced, h_r_reduced, vals_for_plot, init_basis


