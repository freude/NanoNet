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
    h_list :
        list of the Hamiltonian blocks - blocks describes coupling
        with left-side lead, Hamiltonian of the device region and
        coupling with right-side lead

    Returns
    -------
    eigenvals : numpy.ndarray
        Non-zero value indicates error code, or zero on success.
    eigenvects : numpy.ndarray
        Human readable error message, or None on success.
    
    """

    # linearize polynomial eigenvalue problem
    pr_order = len(h_list) - 1
    matix_size = h_list[0].shape[0]
    full_matrix_size = pr_order * matix_size
    identity = np.identity(matix_size)

    main_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex128)
    overlap_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex128)

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

    betha[betha == 0] = 1e-9
    eigenvals = alpha / betha

    # sort absolute values
    ind = np.argsort(np.abs(eigenvals))
    eigenvals = eigenvals[ind]
    eigenvects = eigenvects[:, ind]
    eigenvects = eigenvects[matix_size:, :]

    # vals = np.copy(eigenvals)
    # mask1 = np.abs(vals) < 0.9999999
    # mask2 = np.abs(vals) > 1.0000001
    # vals = np.angle(vals)
    #
    # vals[mask1] = -5
    # vals[mask2] = 5
    # ind = np.argsort(vals, kind='mergesort')
    #
    # eigenvals = eigenvals[ind]
    # eigenvects = eigenvects[:, ind]

    ind = np.squeeze(np.where(np.abs(np.abs(eigenvals)-1.0) < 0.01))
    if len(ind) > 0:
        gv = group_velocity(eigenvects[:, ind], eigenvals[ind], h_list[2])
        print(gv)

    eigenvals = np.diag(eigenvals)

    norms = linalg.norm(eigenvects, axis=0)
    norms = np.array([1e30 if np.abs(norm) < 0.000001 else norm for norm in norms])
    eigenvects = eigenvects / norms[np.newaxis, :]

    return eigenvals, eigenvects


def group_velocity(eigenvector, eigenvalue, h_r):
    """Computes the group velocity of wave packets

    Parameters
    ----------
    eigenvector : numpy.matrix(dtype=numpy.complex)
        eigenvector
    eigenvalue :
        eigenvalue
    h_r : numpy.matrix
        coupling Hamiltonian

    Returns
    -------

    
    """

    num_vecs = eigenvector.shape[1]
    if num_vecs > 1:
        ans = np.zeros(num_vecs)
        for j in range(num_vecs):
            ans[j] = np.imag(np.dot(np.dot(np.dot(eigenvector[:, j].conj().T, h_r), eigenvalue[j]), eigenvector[:, j]))
    else:
        ans = np.imag(np.dot(np.dot(np.dot(eigenvector.conj().T, h_r), eigenvalue), eigenvector))

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

    for _ in range(num_iter):
        gf = h_r.dot(np.linalg.pinv(E * np.identity(h_0.shape[0]) - h_0 - gf)).dot(h_l)

    return gf


def surface_greens_function(E, h_l, h_0, h_r, iterate=False, damp=0.0001j):
    """Computes surface self-energies using the eigenvalue decomposition.
    The procedure is described in
    [M. Wimmer, Quantum transport in nanostructures: From computational concepts
    to spintronics in graphene and magnetic tunnel junctions, 2009, ISBN-9783868450255].

    Parameters
    ----------
    E :
        energy array
    h_l :
        left-side coupling Hamiltonian
    h_0 :
        channel Hamiltonian
    h_r :
        right-side coupling Hamiltonian
    iterate :
        iterate to stabilize TB matrix (Default value = False)
    damp :
        damping (Default value = 0.0001j)

    Returns
    -------

    
    """

    h_list = [h_l, h_0 - (E+damp) * np.identity(h_0.shape[0]), h_r]
    vals, vects = surface_greens_function_poles(h_list)
    num_eigs = len(vals)

    u_right = vects[:, :num_eigs//2]
    u_left = vects[:, num_eigs-1:num_eigs//2-1:-1]
    lambda_right = vals[:num_eigs//2, :num_eigs//2]
    lambda_left = vals[num_eigs-1:num_eigs//2-1:-1, num_eigs-1:num_eigs//2-1:-1]

    sgf_l = h_r.dot(u_right).dot(lambda_right).dot(np.linalg.pinv(u_right))
    sgf_r = h_l.dot(u_left).dot(np.linalg.inv(lambda_left)).dot(np.linalg.pinv(u_left))

    if iterate:
        return iterate_gf(E, h_0, h_l, h_r, sgf_l, 2), iterate_gf(E, h_0, h_r, h_l, sgf_r, 2)

    return iterate_gf(E, h_0, h_l, h_r, sgf_l, 0), iterate_gf(E, h_0, h_r, h_l, sgf_r, 0)
