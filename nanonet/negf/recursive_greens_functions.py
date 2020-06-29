import copy
import numpy as np
import scipy.linalg as linalg


def mat_left_div(mat_a, mat_b):

    # ans, resid, rank, s = linalg.lstsq(mat_a, mat_b, lapack_driver='gelsy')
    ans, resid, rank, s = np.linalg.lstsq(mat_a, mat_b, rcond=None)

    return ans


def mat_mul(list_of_matrices):
    num_of_mat = len(list_of_matrices)

    unity = np.eye(list_of_matrices[num_of_mat - 1].shape[0])

    for j, item in enumerate(list_of_matrices):
        list_of_matrices[j] = item

    for j in range(9, -1, -1):
        unity = list_of_matrices[j] * unity

    return unity


def recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, s_in=0, s_out=0, damp=0.000001j):
    """
    The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355

    In order to get the electron correlation function output, the parameters s_in has to be set.
    For the hole correlation function, the parameter s_out has to be set.


    :param energy:                     energy
    :type energy:                      numpy array
    :param mat_d_list:                 list of diagonal blocks
    :type mat_d_list:                  list of numpy arrays
    :param mat_u_list:                 list of upper-diagonal blocks
    :type mat_u_list:                  list of numpy arrays
    :param mat_l_list:                 list of lower-diagonal blocks
    :type mat_l_list:                  list of numpy arrays

    :return grd, grl, gru, gr_left:    retarded Green's
                                       function: block-diagonal,
                                                 lower block-diagonal,
                                                 upper block-diagonal,
                                                 left-connected
    :rtype grd, grl, gru, gr_left:     list of numpy arrays
    """
    # -------------------------------------------------------------------
    # ---------- convert input arrays to the matrix data type -----------
    # ----------------- in case they are not matrices -------------------
    # -------------------------------------------------------------------

    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj] = item
        mat_d_list[jj] = mat_d_list[jj] - np.diag(energy * np.ones(mat_d_list[jj].shape[0]) + 1j*damp)

    # computes matrix sizes
    num_of_matrices = len(mat_d_list)  # Number of diagonal blocks.
    mat_shapes = [item.shape for item in mat_d_list]  # This gives the sizes of the diagonal matrices.

    # -------------------------------------------------------------------
    # -------------- compute retarded Green's function ------------------
    # -------------------------------------------------------------------

    # allocate empty lists of certain lengths
    gr_left = [None for _ in range(num_of_matrices)]
    gr_left[0] = mat_left_div(-mat_d_list[0], np.eye(mat_shapes[0][0]))  # Initialising the retarded left connected.

    for q in range(num_of_matrices - 1):  # Recursive algorithm (B2)
        gr_left[q + 1] = mat_left_div((-mat_d_list[q + 1] - mat_l_list[q].dot(gr_left[q]).dot(mat_u_list[q])),
                                      np.eye(mat_shapes[q + 1][0]))      # The left connected recursion.
    # -------------------------------------------------------------------

    grl = [None for _ in range(num_of_matrices-1)]
    gru = [None for _ in range(num_of_matrices-1)]
    grd = copy.copy(gr_left)                                             # Our glorious benefactor.
    g_trans = copy.copy(gr_left[len(gr_left)-1])

    for q in range(num_of_matrices - 2, -1, -1):                         # Recursive algorithm
        grl[q] = grd[q + 1].dot(mat_l_list[q]).dot(gr_left[q])           # (B5) We get the off-diagonal blocks for free.
        gru[q] = gr_left[q].dot(mat_u_list[q]).dot(grd[q + 1])           # (B6) because we need .Tthem.T for the next calc:
        grd[q] = gr_left[q] + gr_left[q].dot(mat_u_list[q]).dot(grl[q])  # (B4) I suppose I could also use the lower.
        g_trans = gr_left[q].dot(mat_u_list[q]).dot(g_trans)

    # -------------------------------------------------------------------
    # ------ compute the electron correlation function if needed --------
    # -------------------------------------------------------------------

    if isinstance(s_in, list):

        gin_left = [None for _ in range(num_of_matrices)]
        gin_left[0] = gr_left[0].dot(s_in[0]).dot(np.conj(gr_left[0]))

        for q in range(num_of_matrices - 1):
            sla2 = mat_l_list[q].dot(gin_left[q]).dot(np.conj(mat_u_list[q]))
            prom = s_in[q + 1] + sla2
            gin_left[q + 1] = np.real(gr_left[q + 1].dot(prom).dot(np.conj(gr_left[q + 1])))

        # ---------------------------------------------------------------

        gnl = [None for _ in range(num_of_matrices - 1)]
        gnu = [None for _ in range(num_of_matrices - 1)]
        gnd = copy.copy(gin_left)

        for q in range(num_of_matrices - 2, -1, -1):               # Recursive algorithm
            gnl[q] = grd[q + 1].dot(mat_l_list[q]).dot(gin_left[q]) +\
                     gnd[q + 1].dot(np.conj(mat_l_list[q])).dot(np.conj(gr_left[q]))
            gnd[q] = np.real(gin_left[q] +
                             gr_left[q].dot(mat_u_list[q]).dot(gnd[q + 1]).dot(np.conj(mat_l_list[q])).dot(np.conj(gr_left[q])) +
                             (gin_left[q].dot(np.conj(mat_u_list[q])).dot(np.conj(grl[q])) + gru[q].dot(mat_l_list[q]).dot(gin_left[q])))

            gnu[q] = np.conj(gnl[q])

    # -------------------------------------------------------------------
    # -------- compute the hole correlation function if needed ----------
    # -------------------------------------------------------------------
    if isinstance(s_out, list):

        gip_left = [None for _ in range(num_of_matrices)]
        gip_left[0] = gr_left[0].dot(s_out[0]).dot(np.conj(gr_left[0]))

        for q in range(num_of_matrices - 1):
            sla2 = mat_l_list[q].dot(gip_left[q]).dot(np.conj(mat_u_list[q]))
            prom = s_out[q + 1] + sla2
            gip_left[q + 1] = np.real(gr_left[q + 1].dot(prom).dot(np.conj(gr_left[q + 1])))

        # ---------------------------------------------------------------

        gpl = [None for _ in range(num_of_matrices - 1)]
        gpu = [None for _ in range(num_of_matrices - 1)]
        gpd = copy.copy(gip_left)

        for q in range(num_of_matrices - 2, -1, -1):               # Recursive algorithm
            gpl[q] = grd[q + 1].dot(mat_l_list[q]).dot(gip_left[q]) + gpd[q + 1].dot(np.conj(mat_l_list[q])).dot(np.conj(gr_left[q]))
            gpd[q] = np.real(gip_left[q] +
                             gr_left[q].dot(mat_u_list[q]).dot(gpd[q + 1]).dot(np.conj(mat_l_list[q])).dot(np.conj(gr_left[q])) +
                             gip_left[q].dot(np.conj(mat_u_list[q])).dot(np.conj(grl[q])) + gru[q].dot(mat_l_list[q]).dot(gip_left[q]))

            gpu[0] = gpl[0].conj().T

    # -------------------------------------------------------------------
    # -- remove energy from the main diagonal of th Hamiltonian matrix --
    # -------------------------------------------------------------------

    for jj, item in enumerate(mat_d_list):
        mat_d_list[jj] = mat_d_list[jj] + np.diag(energy * np.ones(mat_d_list[jj].shape[0]) + 1j*damp)

    # -------------------------------------------------------------------
    # ---- choose a proper output depending on the list of arguments ----
    # -------------------------------------------------------------------

    if not isinstance(s_in, list) and not isinstance(s_out, list):
        return g_trans, \
               grd, grl, gru, gr_left

    elif isinstance(s_in, list) and not isinstance(s_out, list):
        return g_trans, \
               grd, grl, gru, gr_left, \
               gnd, gnl, gnu, gin_left

    elif not isinstance(s_in, list) and isinstance(s_out, list):
        return g_trans, \
               grd, grl, gru, gr_left, \
               gpd, gpl, gpu, gip_left

    else:
        return g_trans, \
               grd, grl, gru, gr_left, \
               gnd, gnl, gnu, gin_left, \
               gpd, gpl, gpu, gip_left
