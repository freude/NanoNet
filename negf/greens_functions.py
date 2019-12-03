"""
The module contains functions that computes Green's functions and their poles
"""
from __future__ import print_function, division
import pickle
from scipy.optimize import minimize
import os.path
import numpy as np
import scipy.linalg as linalg


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


def surface_greens_function(E, h_l, h_0, h_r, iterate=False, damp=0.0001j):
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
    :param damp:      damping

    :return:          left- and right-side self-energies
    """

    h_list = [h_l, h_0 - (E+damp) * np.identity(h_0.shape[0]), h_r]
    vals, vects = surface_greens_function_poles(h_list)
    vals = np.diag(vals)

    u_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    u_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))

    alpha = 0.01

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
