import logging
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from nanonet.tb.hamiltonian import Hamiltonian
import nanonet.tb as tb
import nanonet.negf as negf
from nanonet.tb.aux_functions import fd
from nanonet.tb.diatomic_matrix_element import me
from nanonet.negf.recursive_greens_functions import recursive_gf
from nanonet.negf.hamiltonian_chain import HamiltonianChain
from ase.visualize import view


class Lead(Hamiltonian):

    def __init__(self, **kwargs):

        logging.info("Making leads.... ")

        super(Lead, self).__init__(**kwargs)
        self.initialize()

        period = kwargs.get('period', [[0, 0, 1.0]])
        if isinstance(period, float) or isinstance(period, int):
            self.set_periodic_bc([[0, 0, period]])
        else:
            self.set_periodic_bc(period)

        self._h_l, self._h_0, self._h_r = self.get_hamiltonians()

        logging.info("Done!")

    @property
    def h_0(self):
        return self._h_0

    @property
    def h_l(self):
        return self._h_l

    @property
    def h_r(self):
        return self._h_r

    def get_self_energy(self, energy, iterate=True, damp=0.00005j):

        flag = False

        if isinstance(energy, (int, float)):
            energy = [energy]
            flag = True

        sgf_l = []
        sgf_r = []

        for E in energy:
            L, R = negf.surface_greens_function(E, self._h_l, self._h_0, self._h_r, iterate=iterate, damp=damp)
            sgf_l.append(L)
            sgf_r.append(R)

        sgf_l = np.array(sgf_l)
        sgf_r = np.array(sgf_r)

        if flag:
            sgf_l[0] = sgf_l
            sgf_r[0] = sgf_r

        return sgf_l, sgf_r


class TwoTerminalDevice(object):

    def __init__(self, ll, dev, rl):

        self._device = dev
        self._left_lead = ll
        self._right_lead = rl
        self._field = None
        self._h_l = None
        self._h_r = None
        self._h_0 = None

        self._initialize()

    def _initialize(self):
        """Compute matrix elements of the Hamiltonian.

        Returns
        -------
        type Hamiltonian
            Returns the instance of the class Hamiltonian
        """

        self._h_0 = [self._left_lead.h_0, self._device.h_matrix, self._right_lead.h_0]

        self._h_l = np.zeros((self._left_lead.basis_size, self._device.basis_size), dtype=complex)
        self._h_r = np.zeros((self._device.basis_size, self._right_lead.basis_size), dtype=complex)

        # if self.compute_overlap:
        #     self.ov_matrix = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        #     self.ov_matrix_bc_add = np.zeros((self.basis_size, self.basis_size), dtype=complex)

        # loop over all nodes
        for j1 in range(self._left_lead.num_of_nodes):
            # find neighbours for each node
            list_of_neighbours = self._device.get_neighbours(self._left_lead._coords[j1])[1:]
            print(list_of_neighbours)

            for j2 in list_of_neighbours:
                # nearest neighbours interaction
                for l1 in range(self._left_lead.orbitals_dict[list(self._left_lead.atom_list.keys())[j1]].num_of_orbitals):
                    for l2 in range(self._device.orbitals_dict[list(self._device.atom_list.keys())[j2]].num_of_orbitals):
                        ind1 = self._left_lead.qn2ind([('atoms', j1), ('l', l1)], )
                        ind2 = self._device.qn2ind([('atoms', j2), ('l', l2)], )

                        atom_kind1 = self._left_lead._ind2atom(j1)
                        atom_kind2 = self._device._ind2atom(j2)

                        coords1 = np.array(list(self._left_lead.atom_list.values())[j1], dtype=float) - \
                                  np.array(list(self._device.atom_list.values())[j2], dtype=float)

                        norm = np.linalg.norm(coords1)

                        if self._device.int_radial_dependence is None:
                            which_neighbour = ""
                        else:
                            which_neighbour = self._device.int_radial_dependence(norm)

                        if self._device.radial_dependence is None:
                            factor = 1.0
                        else:
                            factor = self._device.radial_dependence(norm)

                        # compute directional cosines
                        if self._device.compute_angular:
                            coords1 /= norm
                        else:
                            coords1 = np.array([1.0, 0.0, 0.0])

                        self._h_l[ind1, ind2] = me(atom_kind1, l1, atom_kind2, l2, coords1, which_neighbour, overlap=False) * factor

        # loop over all nodes
        for j1 in range(self._right_lead.num_of_nodes):
            # find neighbours for each node
            list_of_neighbours = self._device.get_neighbours(self._right_lead._coords[j1])[1:]
            print(list_of_neighbours)

            for j2 in list_of_neighbours:
                # nearest neighbours interaction
                for l1 in range(self._right_lead.orbitals_dict[
                                    list(self._right_lead.atom_list.keys())[j1]].num_of_orbitals):
                    for l2 in range(self._device.orbitals_dict[
                                        list(self._device.atom_list.keys())[j2]].num_of_orbitals):
                        ind1 = self._right_lead.qn2ind([('atoms', j1), ('l', l1)], )
                        ind2 = self._device.qn2ind([('atoms', j2), ('l', l2)], )

                        atom_kind1 = self._right_lead._ind2atom(j1)
                        atom_kind2 = self._device._ind2atom(j2)

                        coords1 = np.array(list(self._right_lead.atom_list.values())[j1], dtype=float) - \
                                  np.array(list(self._device.atom_list.values())[j2], dtype=float)

                        norm = np.linalg.norm(coords1)

                        if self._device.int_radial_dependence is None:
                            which_neighbour = ""
                        else:
                            which_neighbour = self._device.int_radial_dependence(norm)

                        if self._device.radial_dependence is None:
                            factor = 1.0
                        else:
                            factor = self._device.radial_dependence(norm)

                        # compute directional cosines
                        if self._device.compute_angular:
                            coords1 /= norm
                        else:
                            coords1 = np.array([1.0, 0.0, 0.0])

                        self._h_r[ind2, ind1] = me(atom_kind1, l1, atom_kind2, l2, coords1,
                                                   which_neighbour, overlap=False) * factor

                        # if self.compute_overlap:
                        #     self.ov_matrix[ind1, ind2] = self._get_me(j1, j2, l1, l2, overlap=True)

        self._h_l = [self._h_l, self._h_r]
        self._h_r = [self._h_l[0].conj().T, self._h_r.conj().T]

        # self._h_r = [self._h_r, self._h_l.conj().T]
        # self._h_l = [self._h_r[0].conj().T, self._h_l]

    def sgf(self, energy, ef1, ef2, tempr, sgf_l, sgf_r):
        """ """

        sgf = [None for _ in range(len(self._h_0))]

        for jjj in range(len(self._h_0)):
            if jjj == 0:
                sgf[jjj] = -2.0 * np.imag(sgf_l) * fd(energy, ef1, tempr)
            elif jjj == len(self._h_0) - 1:
                sgf[jjj] = -2.0 * np.imag(sgf_r) * fd(energy, ef2, tempr)
            else:
                sgf[jjj] = np.zeros(self._h_0[jjj].shape)

        return sgf

    def comp_dos_tr(self, energy):

        dos = np.zeros(len(energy))
        tr = np.zeros(len(energy))

        for j, en in enumerate(energy):

            L, _ = self._left_lead.get_self_energy(en)
            _, R = self._right_lead.get_self_energy(en)

            L = L[0]
            R = R[0]

            self._h_0[-1] = self._h_0[-1] + R
            self._h_0[0] = self._h_0[0] + L

            g_trans, grd, grl, gru, gr_left = recursive_gf(en,
                                                           self._h_r,
                                                           self._h_0,
                                                           self._h_l,
                                                           damp=0.00005j)

            self._h_0[-1] = self._h_0[-1] - R
            self._h_0[0] = self._h_0[0] - L

            gamma_l = 1j * (L - L.conj().T)
            gamma_r = 1j * (R - R.conj().T)

            tr[j] = np.real(np.trace(gamma_r.dot(g_trans.conj().T).dot(gamma_l).dot(g_trans)))

            for jj in range(len(self._h_0)):
                dos[j] = dos[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].conj().T))) / len(self._h_0)

        return dos, tr

    def comp_dos_tr_dense(self, energy, ef1, ef2, tempr):

        dos = np.zeros(len(energy))
        tr = np.zeros(len(energy))
        dens = np.zeros((len(energy), len(self._h_0)))
        dens = np.zeros((len(energy), len(self._h_0[0]) + len(self._h_0[1]) + len(self._h_0[2])))

        for j, en in enumerate(energy):

            L, _ = self._left_lead.get_self_energy(en)
            _, R = self._right_lead.get_self_energy(en)

            L = L[0]
            R = R[0]

            self._h_0[-1] = self._h_0[-1] + R
            self._h_0[0] = self._h_0[0] + L

            g_trans, grd, grl, gru, gr_left, gnd, gnl, gnu, gn_left = recursive_gf(en,
                                                                                   self._h_r,
                                                                                   self._h_0,
                                                                                   self._h_l,
                                                                                   s_in=self.sgf(en, ef1, ef2, tempr, L, R),
                                                                                   damp=0.00005j)

            self._h_0[-1] = self._h_0[-1] - R
            self._h_0[0] = self._h_0[0] - L

            gamma_l = 1j * (L - L.conj().T)
            gamma_r = 1j * (R - R.conj().T)

            tr[j] = np.real(np.trace(gamma_r.dot(g_trans.conj().T).dot(gamma_l).dot(g_trans)))

            for jj in range(len(self._h_0)):
                dos[j] = dos[j] + np.real(np.trace(1j * (grd[jj] - grd[jj].conj().T))) / len(self._h_0)

            dens[j, :] = 2 * np.hstack((np.diag(gnd[0]), np.diag(gnd[1]), np.diag(gnd[2])))

        return dos, tr, dens

    # @property
    # def field(self):
    #     return self._field
    #
    # @property
    # def device(self):
    #     return self._device
    #
    # @property
    # def left_lead(self):
    #     return self._left_lead
    #
    # @property
    # def right_lead(self):
    #     return self._right_lead
    #
    # @field.setter
    # def field(self, value):
    #     pass
    #
    # @device.setter
    # def device(self, value):
    #     self._device = value
    #
    # @left_lead.setter
    # def left_lead(self, value):
    #     pass
    #
    # @right_lead.setter
    # def right_lead(self, value):
    #     pass
    #
    # @field.deleter
    # def field(self):
    #     pass
    #
    # @left_lead.deleter
    # def left_lead(self):
    #     pass
    #
    # @right_lead.deleter
    # def right_lead(self):
    #     pass
    #
    # @device.deleter
    # def device(self):
    #     pass

    def get_matrix(self):
        """ """

        ans = block_diag(*tuple(self._h_0))

        y0 = 0
        x0 = self._h_r[0].shape[1]

        for item in self._h_r:
            ans[x0:x0 + item.shape[0], y0:y0 + item.shape[1]] = item
            x0 = x0 + item.shape[0]
            y0 = y0 + item.shape[1]

        x0 = 0
        y0 = self._h_l[0].shape[0]

        for item in self._h_l:
            ans[x0:x0 + item.shape[0], y0:y0 + item.shape[1]] = item
            x0 = x0 + item.shape[0]
            y0 = y0 + item.shape[1]

        return ans

    def visualize(self):
        """ """

        import matplotlib.pyplot as plt
        from matplotlib import cm

        vals = np.zeros(len(self.fields[0])*len(self.fields[0][0]))

        for field in self.fields:
            vals += np.hstack(tuple([item for item in field]))

        vals /= np.max(vals)

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        norm = cm.colors.Normalize(vmax=np.abs(np.max(vals)), vmin=-np.abs(np.min(vals)))

        ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                   c=vals,
                   marker='o',
                   norm=norm,
                   cmap=cm.get_cmap(cm.coolwarm))

        plt.show()

    @property
    def coords(self):
        """ """
        if self.elem_length is not None:
            coords = [self._coords - jjj * self.elem_length for jjj in range(self.left_translations, 0, -1)] + \
                     [self._coords] + \
                     [self._coords + jjj * self.elem_length for jjj in range(1, self.right_translations + 1)]

            return np.concatenate(coords)
        else:
            return self._coords


def main():

    # --------------------- make orbitals ---------------------

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

    # --------------------- make leads ---------------------

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    5.0    4.0
                          B1    0.0    5.0    5.0
                          A2    0.0    6.0    4.0
                          B2    0.0    6.0    5.0
                       """)

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    0.0    0.0
                          B1    0.0    0.0    1.0
                          A2    0.0    1.0    0.0
                          B2    0.0    1.0    1.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""14
                                   H cell
                                   A1    0.0    0.0    2.0
                                   B1    0.0    0.0    3.0
                                   A2    0.0    1.0    2.0
                                   B2    0.0    1.0    3.0
                                   A3    0.0    2.0    2.0
                                   B3    0.0    2.0    3.0
                                   A4    0.0    3.0    2.0
                                   B4    0.0    3.0    3.0
                                   A5    0.0    4.0    2.0
                                   B5    0.0    4.0    3.0
                                   A6    0.0    5.0    2.0
                                   B6    0.0    5.0    3.0
                                   A7    0.0    6.0    2.0
                                   B7    0.0    6.0    3.0
                                """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    energy = np.linspace(-3.0, 1.5, 1700)
    dos, tr = a.comp_dos_tr(energy)

    plt.figure(1)
    plt.plot(np.log(tr))
    plt.figure(2)
    plt.plot(np.log(dos))
    plt.show()


def main0():

    # --------------------- make orbitals ---------------------

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

    # --------------------- make leads ---------------------

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    5.0    -6.0
                          B1    0.0    5.0    -5.0
                          A2    0.0    6.0    -6.0
                          B2    0.0    6.0    -5.0
                       """)

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    0.0    -10.0
                          B1    0.0    0.0    -9.0
                          A2    0.0    1.0    -10.0
                          B2    0.0    1.0    -9.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""14
                                   H cell
                                   A1    0.0    0.0    -8.0
                                   B1    0.0    0.0    -7.0
                                   A2    0.0    1.0    -8.0
                                   B2    0.0    1.0    -7.0
                                   A3    0.0    2.0    -8.0
                                   B3    0.0    2.0    -7.0
                                   A4    0.0    3.0    -8.0
                                   B4    0.0    3.0    -7.0
                                   A5    0.0    4.0    -8.0
                                   B5    0.0    4.0    -7.0
                                   A6    0.0    5.0    -8.0
                                   B6    0.0    5.0    -7.0
                                   A7    0.0    6.0    -8.0
                                   B7    0.0    6.0    -7.0
                                """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    energy = np.linspace(-3.0, 1.5, 1700)
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main1():
    # --------------------- make orbitals ---------------------

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

    # --------------------- make leads ---------------------

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    0.0    4.0
                          B1    0.0    0.0    5.0
                          A2    0.0    1.0    4.0
                          B2    0.0    1.0    5.0
                       """)

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    0.0    0.0
                          B1    0.0    0.0    1.0
                          A2    0.0    1.0    0.0
                          B2    0.0    1.0    1.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""4
                                   H cell
                                   A1    0.0    0.0    2.0
                                   B1    0.0    0.0    3.0
                                   A2    0.0    1.0    2.0
                                   B2    0.0    1.0    3.0
                                """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    energy = np.linspace(-3.0, 1.5, 1700)
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main1_1():
    # --------------------- make orbitals ---------------------

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

    # --------------------- make leads ---------------------

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    0.0    4.0
                          B1    0.0    0.0    5.0
                          A2    0.0    1.0    4.0
                          B2    0.0    1.0    5.0
                       """)

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0, 2.0]],
                   xyz="""4
                          H cell
                          A1    0.0    0.0    0.0
                          B1    0.0    0.0    1.0
                          A2    0.0    1.0    0.0
                          B2    0.0    1.0    1.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""4
                                   H cell
                                   A1    0.0    0.0    2.0
                                   B1    0.0    0.0    3.0
                                   A2    0.0    1.0    2.0
                                   B2    0.0    1.0    3.0
                                """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    energy = np.linspace(-3.0, 1.5, 1700)
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main2():
    # --------------------- make orbitals ---------------------

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)

    tb.Orbitals.orbital_sets = {'A': a}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5})

    # --------------------- make leads ---------------------

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0, 1.0]],
                   xyz="""1
                          A cell
                          A1    0.0    4.0    2.0
                       """)

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0, 1.0]],
                   xyz="""1
                          A cell
                          A1    0.0    10.0    0.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""9
                                   H cell
                                   A4    0.0    3.0    1.0
                                   A5    0.0    4.0    1.0
                                   A6    0.0    5.0    1.0
                                   A7    0.0    6.0    1.0
                                   A8    0.0    7.0    1.0
                                   A9    0.0    8.0    1.0
                                   A10    0.0    9.0    1.0
                                   A11    0.0    10.0    1.0
                                   A12    0.0    11.0    1.0
                                """)

    # device = tb.Hamiltonian(nn_distance=1.1,
    #                         xyz="""4
    #                                H cell
    #                                A1    0.0    0.0    2.0
    #                                B1    0.0    0.0    3.0
    #                                A2    0.0    1.0    2.0
    #                                B2    0.0    1.0    3.0
    #                             """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    energy = np.linspace(-3.0, 1.5, 1700)
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main3():
    # --------------------- make orbitals ---------------------

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.7)

    b = tb.Orbitals('B')
    b.add_orbital('s', -0.3)

    tb.Orbitals.orbital_sets = {'A': a, 'B': b}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.5}, PARAMS_A_B={'ss_sigma': -0.5}, PARAMS_B_B={'ss_sigma': -0.5})

    # --------------------- make leads ---------------------

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0, 1.0]],
                   xyz="""1
                          A cell
                          A1    0.0    4.0    2.0
                       """)

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0, 1.0]],
                   xyz="""1
                          A cell
                          A1    0.0    10.0    0.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""9
                                   H cell
                                   A4    0.0    3.0    1.0
                                   B5    0.0    4.0    1.0
                                   A6    0.0    5.0    1.0
                                   B7    0.0    6.0    1.0
                                   A8    0.0    7.0    1.0
                                   B9    0.0    8.0    1.0
                                   A10    0.0    9.0    1.0
                                   B11    0.0    10.0    1.0
                                   A12    0.0    11.0    1.0
                                """)

    # device = tb.Hamiltonian(nn_distance=1.1,
    #                         xyz="""4
    #                                H cell
    #                                A1    0.0    0.0    2.0
    #                                B1    0.0    0.0    3.0
    #                                A2    0.0    1.0    2.0
    #                                B2    0.0    1.0    3.0
    #                             """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    energy = np.linspace(-3.0, 1.5, 1700)
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main4(energy):
    # --------------------- make orbitals ---------------------

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.2)

    b = tb.Orbitals('B')
    b.add_orbital('s', 0.2)

    tb.Orbitals.orbital_sets = {'A': a, 'B': b}

    tb.set_tb_params(PARAMS_A_B={'ss_sigma': -0.5})

    # --------------------- make leads ---------------------

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""2
                          A cell
                          A    0.0    1.0    1.0
                          B    0.0    1.0    2.0
                       """)

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""2
                          A cell
                          B    0.0    1.0    12.0
                          A    0.0    1.0    13.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""9
                                   H cell
                                   A    0.0    1.0    3.0
                                   B    0.0    1.0    4.0
                                   A    0.0    1.0    5.0
                                   B    0.0    1.0    6.0
                                   A    0.0    1.0    7.0
                                   B    0.0    1.0    8.0
                                   A    0.0    1.0    9.0
                                   B    0.0    1.0    10.0
                                   A    0.0    1.0    11.0
                                """)

    # device = tb.Hamiltonian(nn_distance=1.1,
    #                         xyz="""4
    #                                H cell
    #                                A1    0.0    0.0    2.0
    #                                B1    0.0    0.0    3.0
    #                                A2    0.0    1.0    2.0
    #                                B2    0.0    1.0    3.0
    #                             """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main5(energy):
    # --------------------- make orbitals ---------------------

    a = tb.Orbitals('A')
    a.add_orbital('s', 0)

    tb.Orbitals.orbital_sets = {'A': a}

    tb.set_tb_params(PARAMS_A_A={'ss_sigma': -0.7})

    # --------------------- make leads ---------------------

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""1
                          A cell
                          A    0.0    1.0    2.0
                       """)

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""1
                          A cell
                          A    0.0    1.0    12.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""9
                                   H cell
                                   A    0.0    1.0    3.0
                                   A    0.0    1.0    4.0
                                   A    0.0    1.0    5.0
                                   A    0.0    1.0    6.0
                                   A    0.0    1.0    7.0
                                   A    0.0    1.0    8.0
                                   A    0.0    1.0    9.0
                                   A    0.0    1.0    10.0
                                   A    0.0    1.0    11.0
                                """)

    # device = tb.Hamiltonian(nn_distance=1.1,
    #                         xyz="""4
    #                                H cell
    #                                A1    0.0    0.0    2.0
    #                                B1    0.0    0.0    3.0
    #                                A2    0.0    1.0    2.0
    #                                B2    0.0    1.0    3.0
    #                             """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main6():
    # --------------------- make orbitals ---------------------

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.2)

    b = tb.Orbitals('B')
    b.add_orbital('s', 0.2)

    c = tb.Orbitals('C')
    c.add_orbital('s', 0)

    tb.Orbitals.orbital_sets = {'A': a, 'B': b, 'C': c}

    tb.set_tb_params(PARAMS_A_C={'ss_sigma': -0.61}, PARAMS_C_C={'ss_sigma': -0.7}, PARAMS_A_B={'ss_sigma': -0.5})

    # --------------------- make leads ---------------------

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""1
                          A cell
                          C    0.0    1.0    2.0
                       """)

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""1
                          A cell
                          C    0.0    1.0    20.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""17
                                   H cell
                                   A    0.0    1.0    3.0
                                   B    0.0    1.0    4.0
                                   A    0.0    1.0    5.0
                                   B    0.0    1.0    6.0
                                   A    0.0    1.0    7.0
                                   B    0.0    1.0    8.0
                                   A    0.0    1.0    9.0
                                   B    0.0    1.0    10.0
                                   A    0.0    1.0    11.0
                                   B    0.0    1.0    12.0
                                   A    0.0    1.0    13.0
                                   B    0.0    1.0    14.0
                                   A    0.0    1.0    15.0
                                   B    0.0    1.0    16.0
                                   A    0.0    1.0    17.0
                                   B    0.0    1.0    18.0
                                   A    0.0    1.0    19.0
                                """)

    # device = tb.Hamiltonian(nn_distance=1.1,
    #                         xyz="""4
    #                                H cell
    #                                A1    0.0    0.0    2.0
    #                                B1    0.0    0.0    3.0
    #                                A2    0.0    1.0    2.0
    #                                B2    0.0    1.0    3.0
    #                             """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    energy = np.linspace(-2.0, 2.0, 1700)
    dos, tr = a.comp_dos_tr(energy)

    return dos, tr


def main7(energy):
    # --------------------- make orbitals ---------------------

    a = tb.Orbitals('A')
    a.add_orbital('s', -0.2)

    b = tb.Orbitals('B')
    b.add_orbital('s', 0.2)

    c = tb.Orbitals('C')
    c.add_orbital('s', 0)

    tb.Orbitals.orbital_sets = {'A': a, 'B': b, 'C': c}

    tb.set_tb_params(PARAMS_A_C={'ss_sigma': -0.61}, PARAMS_C_C={'ss_sigma': -0.7}, PARAMS_A_B={'ss_sigma': -0.5})

    # --------------------- make leads ---------------------

    leads_l = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""1
                          A cell
                          C    0.0    1.0    2.0
                       """)

    leads_r = Lead(nn_distance=1.1,
                   period=[[0, 0.0, 1.0]],
                   xyz="""1
                          A cell
                          C    0.0    1.0    20.0
                       """)

    # -------------- make two terminal device --------------

    device = tb.Hamiltonian(nn_distance=1.1,
                            xyz="""17
                                   H cell
                                   A    0.0    1.0    3.0
                                   B    0.0    1.0    4.0
                                   A    0.0    1.0    5.0
                                   B    0.0    1.0    6.0
                                   A    0.0    1.0    7.0
                                   B    0.0    1.0    8.0
                                   A    0.0    1.0    9.0
                                   B    0.0    1.0    10.0
                                   A    0.0    1.0    11.0
                                   B    0.0    1.0    12.0
                                   A    0.0    1.0    13.0
                                   B    0.0    1.0    14.0
                                   A    0.0    1.0    15.0
                                   B    0.0    1.0    16.0
                                   A    0.0    1.0    17.0
                                   B    0.0    1.0    18.0
                                   A    0.0    1.0    19.0
                                """)

    # device = tb.Hamiltonian(nn_distance=1.1,
    #                         xyz="""4
    #                                H cell
    #                                A1    0.0    0.0    2.0
    #                                B1    0.0    0.0    3.0
    #                                A2    0.0    1.0    2.0
    #                                B2    0.0    1.0    3.0
    #                             """)

    device.initialize()

    a = TwoTerminalDevice(leads_l, device, leads_r)
    print(a._h_l)
    print(a._h_0)
    print(a._h_r)
    plt.imshow(np.real(a.get_matrix()))
    plt.show()
    ef1 = 0
    ef2 = 0
    tempr = 300
    dos, tr, dense = a.comp_dos_tr_dense(energy, ef1, ef2, tempr)

    return dos, tr, dense


if __name__ == '__main__':

    # main2()
    # dos, tr = main0()
    # dos1, tr1 = main1()
    #
    # plt.figure(1)
    # plt.plot(np.log(tr))
    # plt.plot(np.log(tr1))
    # plt.figure(2)
    # plt.plot(np.log(dos))
    # plt.plot(np.log(dos1))
    # plt.show()

    # dos, tr = main6()
    #
    # plt.figure(1)
    # plt.plot(tr)
    # # plt.yscale('log')
    #
    # plt.figure(2)
    # plt.plot(dos)
    # # plt.yscale('log')
    # plt.show()

    energy = np.linspace(-2.0, 2.0, 1700)

    dos1, tr1 = main4(energy)
    dos2, tr2 = main5(energy)

    plt.figure(1)
    plt.plot(tr1)
    plt.plot(tr2)
    plt.yscale('log')

    plt.figure(2)
    plt.plot(dos1)
    plt.plot(dos2)
    plt.yscale('log')
    plt.show()

    dos3, tr3, dens = main7(energy)

    plt.figure(3)
    plt.plot(tr3)
    plt.figure(4)
    plt.plot(dos3)
    plt.show()
