"""
The module contains functions that computes Green's functions and their poles
"""
from __future__ import print_function, division
import numpy as np
import scipy.linalg as linalg


def surface_greens_function_poles(h_list):
    """Computes eigenvalues and eigenvectors for the complex band structure problem.
    The eigenvalues correspond to the wave vectors as `exp(ik)`.

    Parameters
    ----------
    h_list : list
        List of the Hamiltonian blocks - blocks describes coupling
        with left-side lead, Hamiltonian of the device region and
        coupling with right-side lead

    Returns
    -------
    eigenvals : numpy.ndarray (dtype=numpy.complex)
        Array of sorted eigenvalues
    eigenvects : numpy.ndarray (dtype=numpy.complex)
        Eigenvector matrix
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

    betha[betha == 0] = 1e-19
    eigenvals = alpha / betha

    # sort absolute values
    ind = np.argsort(np.abs(eigenvals))
    eigenvals = eigenvals[ind]
    eigenvects = eigenvects[:, ind]
    eigenvects = eigenvects[matix_size:, :]

    return eigenvals, eigenvects


def group_velocity(eigenvector, eigenvalue, h_r):
    """Computes a group velocity of wave packets

    Parameters
    ----------
    eigenvector : numpy.ndarray (dtype=numpy.complex)
        Eigenvectors
    eigenvalue : numpy.ndarray (dtype=numpy.complex)
        Eigenvalues
    h_r : numpy.matrix numpy.ndarray (dtype=numpy.float)
        Coupling Hamiltonian

    Returns
    -------
    gv : float
        Group velocity of the wave package
    """

    num_vecs = eigenvector.shape[1]
    if num_vecs > 1:
        gv = np.zeros(num_vecs)
        for j in range(num_vecs):
            gv[j] = np.imag(np.dot(np.dot(np.dot(eigenvector[:, j].conj().T, h_r), eigenvalue[j]), eigenvector[:, j]))
    else:
        gv = np.imag(np.dot(np.dot(np.dot(eigenvector.conj().T, h_r), eigenvalue), eigenvector))

    return gv


def iterate_gf(E, h_0, h_l, h_r, gf, num_iter):
    """

    Parameters
    ----------
    E : numpy.ndarray (dtype=numpy.float)
        Energy array
    h_l : numpy.ndarray (dtype=numpy.float)
        Left-side coupling Hamiltonian
    h_0 : numpy.ndarray (dtype=numpy.float)
        Hamiltonian of the device region
    h_r : numpy.ndarray (dtype=numpy.float)
        Right-side coupling Hamiltonian
    se : numpy.ndarray (dtype=numpy.complex)
        Self-energy
    num_iter : int
        Number of iterations

    Returns
    -------
    se : numpy.ndarray (dtype=numpy.complex)
        Self-energy
    """

    for _ in range(num_iter):
        se = h_r.dot(np.linalg.pinv(E * np.identity(h_0.shape[0]) - h_0 - gf)).dot(h_l)

    return se


def surface_greens_function(E, h_l, h_0, h_r, iterate=False, damp=0.0001j):
    """Computes surface self-energies using the eigenvalue decomposition.
    The procedure is described in
    [M. Wimmer, Quantum transport in nanostructures: From computational concepts
    to spintronics in graphene and magnetic tunnel junctions, 2009, ISBN-9783868450255].

    Parameters
    ----------
    E : numpy.ndarray (dtype=numpy.float)
        Energy array
    h_l : numpy.ndarray (dtype=numpy.float)
        Left-side coupling Hamiltonian
    h_0 : numpy.ndarray (dtype=numpy.float)
        Hamiltonian of the device region
    h_r : numpy.ndarray (dtype=numpy.float)
        Right-side coupling Hamiltonian
    iterate : bool
        Iterate to stabilize TB matrix (Default value = False)
    damp :
        damping (Default value = 0.0001j)

    Returns
    -------
    sgf_l : numpy.ndarray (dtype=numpy.complex)
        Left-lead self-energy matrix
    sgf_r : numpy.ndarray (dtype=numpy.complex)
        Right-lead self-energy matrix
    """

    h_list = [h_l, h_0 - (E+damp) * np.identity(h_0.shape[0]), h_r]
    vals, vects = surface_greens_function_poles(h_list)
    num_eigs = len(vals)

    u_right = vects[:, :num_eigs//2]
    u_left = vects[:, num_eigs-1:num_eigs//2-1:-1]
    lambda_right = np.diag(vals[:num_eigs//2])
    lambda_left = np.diag(vals[num_eigs-1:num_eigs//2-1:-1])

    sgf_l = h_r.dot(u_right).dot(lambda_right).dot(np.linalg.pinv(u_right))
    sgf_r = h_l.dot(u_left).dot(np.linalg.inv(lambda_left)).dot(np.linalg.pinv(u_left))

    if iterate:
        return iterate_gf(E, h_0, h_l, h_r, sgf_l, 2), iterate_gf(E, h_0, h_r, h_l, sgf_r, 2)

    return sgf_l, sgf_r
