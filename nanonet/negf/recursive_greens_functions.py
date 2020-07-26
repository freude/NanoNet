import copy
import numpy as np
import scipy.linalg as linalg


def mat_left_div(mat_a, mat_b):
    """
    Solves the linear problem AX=B, where A is the linear operator,
    X is an unknown vector and B is the right-hand side column vector.

    Parameters
    ----------
    mat_a : numpy.ndarray (dtype=numpy.float)
        Linear operator
    mat_b : numpy.ndarray (dtype=numpy.float)
        Right-side column vector

    Returns
    -------
    ans : numpy.ndarray (dtype=numpy.float)
        Solution of the linear system
    """

    # ans, resid, rank, s = np.linalg.lstsq(mat_a, mat_b, rcond=1e-9)
    ans, resid, rank, s = linalg.lstsq(mat_a, mat_b, lapack_driver='gelsy')

    return ans


def _recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, s_in=0, s_out=0, damp=0.000001j):
    """The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355
    
    In order to get the electron correlation function output, the parameters s_in has to be set.
    For the hole correlation function, the parameter s_out has to be set.

    Parameters
    ----------
    energy : numpy.ndarray (dtype=numpy.float)
        Energy array
    mat_d_list : list of numpy.ndarray (dtype=numpy.float)
        List of diagonal blocks
    mat_u_list : list of numpy.ndarray (dtype=numpy.float)
        List of upper-diagonal blocks
    mat_l_list : list of numpy.ndarray (dtype=numpy.float)
        List of lower-diagonal blocks
    s_in :
         (Default value = 0)
    s_out :
         (Default value = 0)
    damp :
         (Default value = 0.000001j)

    Returns
    -------
    g_trans : numpy.ndarray (dtype=numpy.complex)
        Blocks of the retarded Green's function responsible for transmission
    grd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    grl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gru : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gr_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gnd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gnl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gnu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gin_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gpd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gpl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gpu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gip_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
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


def recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, left_se=None, right_se=None, s_in=0, s_out=0, damp=0.000001j):
    """The recursive Green's function algorithm is taken from
    M. P. Anantram, M. S. Lundstrom and D. E. Nikonov, Proceedings of the IEEE, 96, 1511 - 1550 (2008)
    DOI: 10.1109/JPROC.2008.927355

    In order to get the electron correlation function output, the parameters s_in has to be set.
    For the hole correlation function, the parameter s_out has to be set.

    Parameters
    ----------
    energy : numpy.ndarray (dtype=numpy.float)
        Energy array
    mat_d_list : list of numpy.ndarray (dtype=numpy.float)
        List of diagonal blocks
    mat_u_list : list of numpy.ndarray (dtype=numpy.float)
        List of upper-diagonal blocks
    mat_l_list : list of numpy.ndarray (dtype=numpy.float)
        List of lower-diagonal blocks
    s_in :
         (Default value = 0)
    s_out :
         (Default value = 0)
    damp :
         (Default value = 0.000001j)

    Returns
    -------
    g_trans : numpy.ndarray (dtype=numpy.complex)
        Blocks of the retarded Green's function responsible for transmission
    grd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    grl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gru : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gr_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gnd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gnl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gnu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gin_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    gpd : numpy.ndarray (dtype=numpy.complex)
        Diagonal blocks of the retarded Green's function
    gpl : numpy.ndarray (dtype=numpy.complex)
        Lower diagonal blocks of the retarded Green's function
    gpu : numpy.ndarray (dtype=numpy.complex)
        Upper diagonal blocks of the retarded Green's function
    gip_left : numpy.ndarray (dtype=numpy.complex)
        Left-conencted blocks of the retarded Green's function
    """

    if isinstance(left_se, np.ndarray):
        s01, s02 = mat_d_list[0].shape
        left_se = left_se[:s01, :s02]
        mat_d_list[0] = mat_d_list[0] + left_se

    if isinstance(right_se, np.ndarray):
        s11, s12 = mat_d_list[-1].shape
        right_se = right_se[-s11:, -s12:]
        mat_d_list[-1] = mat_d_list[-1] + right_se

    ans = _recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, s_in=s_in, s_out=s_out, damp=damp)

    if isinstance(left_se, np.ndarray):
        mat_d_list[0] = mat_d_list[0] - left_se

    if isinstance(right_se, np.ndarray):
        mat_d_list[-1] = mat_d_list[-1] - right_se

    return ans