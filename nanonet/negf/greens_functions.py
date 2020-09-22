"""
The module contains functions that computes Green's functions and their poles
"""
from __future__ import print_function, division
import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg


def surface_greens_function_poles(hl, h0, hr):
    """Computes eigenvalues and eigenvectors for the complex band structure problem.
    The eigenvalues correspond to the wave vectors as `exp(ik)`.

    Parameters
    ----------
    hl : numpy.ndarray (dtype=numpy.float)
        Left-side coupling Hamiltonian
    h0 : numpy.ndarray (dtype=numpy.float)
        Hamiltonian of the device region
    hr : numpy.ndarray (dtype=numpy.float)
        Right-side coupling Hamiltonian

    Returns
    -------
    eigenvals : numpy.ndarray (dtype=numpy.complex)
        Array of sorted eigenvalues
    eigenvects : numpy.ndarray (dtype=numpy.complex)
        Eigenvector matrix
    """

    # linearize polynomial eigenvalue problem

    shapes_d = []
    shapes_ud = []
    shapes_ld = []

    if isinstance(h0, list):
        num_blocks = len(h0)
        for j in range(len(h0)):
            shapes_d.append(h0[j].shape[0])
            shapes_ld.append(hl[j].shape)
            shapes_ud.append(hr[j].shape)
        matix_size = np.sum(shapes_d)
    else:
        num_blocks = 1
        shapes_d.append(h0.shape[0])
        shapes_ld.append(hl.shape)
        shapes_ud.append(hr.shape)
        matix_size = h0.shape[0]
        hl = [hl]
        h0 = [h0]
        hr = [hr]

    full_matrix_size = 2 * matix_size
    identity = np.identity(matix_size)
    main_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex)
    overlap_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex)
    main_matrix[0:matix_size, matix_size:2 * matix_size] = identity
    overlap_matrix[0:matix_size, 0:matix_size] = identity
    main_matrix[matix_size:matix_size+hl[-1].shape[0], matix_size-hl[-1].shape[1]:matix_size] = -hl[-1]
    overlap_matrix[2*matix_size-hr[-1].shape[0]:2*matix_size, matix_size:matix_size+hr[-1].shape[1]] = hr[-1]

    offfset = matix_size

    for j in range(num_blocks):

        main_matrix[offfset:offfset+h0[j].shape[0], offfset:offfset+h0[j].shape[0]] = -h0[j]

        if j > 0:
            main_matrix[offfset:offfset + hl[j - 1].shape[0], offfset - hl[j - 1].shape[1]:offfset] = hl[j - 1]
            main_matrix[offfset - hl[j - 1].shape[1]:offfset, offfset:offfset + hl[j - 1].shape[0]] = hr[j - 1]

        offfset += h0[j].shape[0]

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


def iterate_gf(E, h_0, h_l, h_r, se, num_iter):
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
        se = h_r.dot(linalg.pinv(E * np.identity(h_0.shape[0]) - h_0 - se)).dot(h_l)
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

    # linearize polynomial eigenvalue problem

    shapes_d = []
    shapes_ud = []
    shapes_ld = []

    if isinstance(h_0, list):
        num_blocks = len(h_0)
        for j in range(len(h_0)):
            shapes_d.append(h_0[j].shape[0])
            shapes_ld.append(h_l[j].shape)
            shapes_ud.append(h_r[j].shape)
        matix_size = np.sum(shapes_d)
    else:
        num_blocks = 1
        shapes_d.append(h_0.shape[0])
        shapes_ld.append(h_l.shape)
        shapes_ud.append(h_r.shape)
        matix_size = h_0.shape[0]
        h_l = [h_l]
        h_0 = [h_0]
        h_r = [h_r]

    full_matrix_size = 2 * matix_size
    identity = np.identity(matix_size)
    main_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex)
    overlap_matrix = np.zeros((full_matrix_size, full_matrix_size), dtype=np.complex)
    main_matrix[0:matix_size, matix_size:2 * matix_size] = identity
    overlap_matrix[0:matix_size, 0:matix_size] = identity
    main_matrix[matix_size:matix_size+h_l[-1].shape[0], matix_size-h_l[-1].shape[1]:matix_size] = -h_l[-1]
    overlap_matrix[2*matix_size-h_r[-1].shape[0]:2*matix_size, matix_size:matix_size+h_r[-1].shape[1]] = h_r[-1]

    offfset = matix_size

    for j in range(num_blocks):

        main_matrix[offfset:offfset+h_0[j].shape[0], offfset:offfset+h_0[j].shape[0]] = -h_0[j]

        if j > 0:
            main_matrix[offfset:offfset + h_l[j - 1].shape[0], offfset - h_l[j - 1].shape[1]:offfset] = -h_l[j - 1]
            main_matrix[offfset - h_l[j - 1].shape[1]:offfset, offfset:offfset + h_l[j - 1].shape[0]] = -h_r[j - 1]

        offfset += h_0[j].shape[0]

    main_matrix[matix_size:2*matix_size, matix_size:2*matix_size] = \
        (E + damp) * np.identity(matix_size) + main_matrix[matix_size:2*matix_size, matix_size:2*matix_size]

    alpha, betha, _, vects, _, _ = linalg.lapack.cggev(main_matrix, overlap_matrix)

    betha[betha == 0] = 1e-19
    vals = alpha / betha

    # sort absolute values
    ind = np.argsort(np.abs(vals))
    vals = vals[ind]
    vects = vects[:, ind]
    vects = vects[matix_size:, :]

    # vals, vects = surface_greens_function_poles(E, h_l, h_0, h_r, damp)
    num_eigs = len(vals)

    u_right = vects[:, :num_eigs // 2]
    u_left = vects[:, num_eigs - 1:num_eigs // 2 - 1:-1]
    lambda_right = np.diag(vals[:num_eigs // 2])
    lambda_left = np.diag(vals[num_eigs - 1:num_eigs // 2 - 1:-1])

    h0 = -main_matrix[matix_size:2*matix_size, matix_size:2*matix_size] + (E + damp) * np.identity(matix_size)
    hl = -main_matrix[matix_size:2*matix_size, 0:matix_size]
    hr = overlap_matrix[matix_size:2*matix_size, matix_size:2*matix_size]

    sgf_l = hr.dot(u_right).dot(lambda_right).dot(np.linalg.pinv(u_right))
    sgf_r = hl.dot(u_left).dot(np.linalg.inv(lambda_left)).dot(np.linalg.pinv(u_left))

    if iterate:
        sgf_l = iterate_gf(E, h0, hl, hr, sgf_l, 2)
        sgf_r = iterate_gf(E, h0, hr, hl, sgf_r, 2)

    s01, s02 = h_0[0].shape
    s11, s12 = h_0[-1].shape

    sgf_l = sgf_l[-s11:, -s12:]
    sgf_r = sgf_r[:s01, :s02]

    return sgf_r, sgf_l
