import numpy as np
from scipy.linalg import block_diag
from nanonet.tb.aux_functions import yaml_parser
from nanonet.negf.field import Field


def fd(energy, ef, temp):
    """

    Parameters
    ----------
    energy :
        
    ef :
        
    temp :
        

    Returns
    -------

    """
    kb = 8.61733e-5       # Boltzmann constant in eV
    return 1.0 / (1.0 + np.exp((energy - ef) / (kb * temp)))


class HamiltonianChain(object):
    """ """

    def __init__(self, h_l, h_0, h_r, coords):

        self.h_l = h_l
        self.h_0 = h_0
        self.h_r = h_r
        self._coords = coords
        self._z_coords_map = []

        self.num_sites = h_0.shape[0]

        self.elem_length = None
        self.left_translations = None
        self.right_translations = None
        self.fields = None
        self.sgf_l = None
        self.sgf_r = None

        self.energy = 0
        self.tempr = 0
        self.ef1 = 0
        self.ef2 = 0

    @property
    def sgf(self):
        """ """

        sgf = [None for _ in range(len(self.h_0))]

        for jjj in range(len(self.h_0)):
            if jjj == 0:
                sgf[jjj] = -2.0 * np.imag(self.sgf_l) * fd(self.energy, self.ef1, self.tempr)
            elif jjj == len(self.h_0) - 1:
                sgf[jjj] = -2.0 * np.imag(self.sgf_r) * fd(self.energy, self.ef2, self.tempr)
            else:
                sgf[jjj] = np.zeros(self.h_0[jjj].shape)

        return sgf

    def translate(self, period, left_translations, right_translations):
        """

        Parameters
        ----------
        period :
            
        left_translations :
            
        right_translations :
            

        Returns
        -------

        """

        self.elem_length = np.array(period)
        self.left_translations = left_translations
        self.right_translations = right_translations

        self.h_l = [self.h_l for _ in range(left_translations)] + \
                   [self.h_l for _ in range(right_translations)]

        self.h_0 = [self.h_0 for _ in range(left_translations)] + \
                   [self.h_0] + \
                   [self.h_0 for _ in range(right_translations)]

        self.h_r = [self.h_r for _ in range(left_translations)] + \
                   [self.h_r for _ in range(right_translations)]

    def add_field(self, field, eps=7.0):
        """

        Parameters
        ----------
        field :
            
        eps :
             (Default value = 7.0)

        Returns
        -------

        """

        field_buf = []

        for jjj in range(self.left_translations, 0, -1):
            field_buf.append(field.get_values(self._coords, translate=jjj * self.elem_length) / eps)

        field_buf.append(field.get_values(self._coords) / eps)

        for jjj in range(1, self.right_translations + 1):
            field_buf.append(field.get_values(self._coords, translate=-jjj * self.elem_length) / eps)

        field_buf = np.array(field_buf)

        if not isinstance(self.fields, list):
            self.fields = []

        self.fields.append(field_buf)

        for jjj in range(len(self.h_0)):
            self.h_0[jjj] = self.h_0[jjj] - np.diag(field_buf[jjj])

    def remove_field(self):
        """ """

        if isinstance(self.fields, list):
            for item in self.fields:
                for jjj in range(len(self.h_0)):
                    self.h_0[jjj] = self.h_0[jjj] - np.diag(item[jjj])

        self.fields = None

    def add_self_energies(self, sgf_l, sgf_r, energy=0, tempr=0, ef1=0, ef2=0):
        """

        Parameters
        ----------
        sgf_l :
            
        sgf_r :
            
        energy :
             (Default value = 0)
        tempr :
             (Default value = 0)
        ef1 :
             (Default value = 0)
        ef2 :
             (Default value = 0)

        Returns
        -------

        """

        self.energy = energy
        self.tempr = tempr
        self.ef1 = ef1
        self.ef2 = ef2

        self.sgf_l = sgf_l
        self.sgf_r = sgf_r

        self.h_0[-1] = self.h_0[-1] + sgf_r
        self.h_0[0] = self.h_0[0] + sgf_l

    def remove_self_energies(self):
        """ """

        self.h_0[-1] = self.h_0[-1] - self.sgf_r
        self.h_0[0] = self.h_0[0] - self.sgf_l

        self.sgf_l = None
        self.sgf_r = None

    def translate_self_energies(self, sgf_l, sgf_r):
        """

        Parameters
        ----------
        sgf_l :
            
        sgf_r :
            

        Returns
        -------

        """

        mat_list = [item * 0.0 for item in self.h_0]

        return block_diag(*tuple([sgf_r] + mat_list[1:])), block_diag(*tuple(mat_list[:-1] + [sgf_l]))

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

    def z_coords_map(self, ind):
        """

        Parameters
        ----------
        ind :
            

        Returns
        -------

        """

        if len(self._z_coords_map) == 0:

            coords = self.coords[:, 2]
            unique_coords = self.z_coords()
            coord_map = np.zeros(coords.shape)

            for j, item in enumerate(unique_coords):
                coord_map[np.abs(coords-item) < 0.001] = j

            self._z_coords_map = coord_map

        return int(self._z_coords_map[ind])

    def z_coords(self):
        """ """

        unique_coords = np.sort(np.array(list(set(self.coords[:, 2]))))

        return unique_coords

    def get_matrix(self):
        """ """

        if isinstance(self.h_0, list):
            matrix = block_diag(*tuple(self.h_0))
        else:
            return self.h_0

        for j in range(len(self.h_l)):

            s1, s2 = self.h_0[j].shape

            matrix[j * s1:(j + 1) * s1, (j + 1) * s2:(j + 2) * s2] = self.h_r[j]
            matrix[(j + 1) * s1:(j + 2) * s1, j * s2:(j + 1) * s2] = self.h_l[j]

        return matrix

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


class HamiltonianChainComposer(HamiltonianChain):
    """ """

    def __init__(self, h_l, h_0, h_r, coords, params):
        super(HamiltonianChainComposer, self).__init__(h_l, h_0, h_r, coords)

        self.translate(params['unit_cell'], params['left_translations'], params['right_translations'])
        self.mol_length = 0
        uc = np.array(params['unit_cell'])

        self.dict_of_fields = {}

        if 'fields' in params:

            for item in params['fields']['xyz']:

                field = list(item.keys())[0]

                if field in self.dict_of_fields:
                    self.dict_of_fields[field].set_origin(self.dict_of_fields[field].mean_coords + np.array(item[field]))
                else:
                    self.dict_of_fields[field] = Field(path=params['fields'][field])
                    # angle = 1.13446
                    angle = params['fields']['angle']

                    self.dict_of_fields[field].rotate('x', angle)  # 65 degrees
                    self.dict_of_fields[field].rotate('y', np.pi / 2.0)

                    size_x_max = np.max(coords[:, 0])
                    size_y_min = np.min(coords[:, 1])
                    size_y_max = np.max(coords[:, 1])

                    _, mol_coords = self.dict_of_fields[field].get_atoms()
                    mol_y_length0 = np.max(mol_coords[:, 1]) - np.min(mol_coords[:, 1])
                    mol_y_length = mol_y_length0 * np.sin(angle)
                    mol_z_length = mol_y_length0 * np.cos(angle)

                    self.dict_of_fields[field].mean_coords = np.array([0.5 * (size_x_max - np.abs(size_y_min)),
                                                                  size_y_max + 0.5 * mol_y_length +
                                                                  params['fields']['spacing'],
                                                                  0.5 * mol_z_length])

                    self.dict_of_fields[field].set_origin(self.dict_of_fields[field].mean_coords + np.array(item[field]))

                if isinstance(params['fields']['eps'], list):
                    self.dict_of_fields[field].add_screening(params['fields']['eps'], mol_y_length0, params['fields']['spacing'])
                    # self.add_field(self.dict_of_fields[field], eps=params['fields']['eps'][3])
                    self.add_field(self.dict_of_fields[field], eps=1.0)
                else:
                    self.add_field(self.dict_of_fields[field], eps=params['fields']['eps'])


if __name__ == '__main__':

    # fields_config = """
    #
    # unit_cell:        [[0, 0, 5.50]]
    #
    # left_translations:     3
    # right_translations:    3
    #
    # fields:
    #
    #     eps = 3.8
    #
    #     cation:      '/home/mk/tetracene_dft_wB_pcm_38_32_cation.cube'
    #
    #     angle:       1.13446
    #     spacing:     5.0
    #
    #     xyz:
    #         - cation:       [0.0000000000,    0.0000000000,    0.0000000000]
    #
    # """
    #
    # params = yaml_parser(fields_config)
    # print('hi')
    import numpy as np
    from matplotlib import pyplot

    fig = pyplot.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), autoscale_on=False)

    # Empty data plot
    points = ax.scatter([], [], color='r', zorder=2)
    # ax properties
    ax.set_xlim(-10, 10)
    ax.set_ylim(5e-2, 5e3)
    ax.set_yscale("log")

    # Example data points
    x_data = [-5, -3, 0, 3, 5]
    y_data = [1, 10, 1000, 10, 1]
    # Set data points
    points.set_offsets(np.hstack((x_data, y_data)).T)

    pyplot.show()