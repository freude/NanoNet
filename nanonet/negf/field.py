import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc


def _getline(cube):
    """Read a line from cube file where first field is an int
    and the remaining fields are floats.

    Parameters
    ----------
    cube :
        file object of the cube file

    Returns
    -------
    type
        

    """
    line = cube.readline().strip().split()
    return int(line[0]), np.array(list(map(float, line[1:])))


def read_cube(fname):
    """Read cube file into numpy array

    Parameters
    ----------
    fname :
        filename of cube file

    Returns
    -------
    type
        

    """
    bohr = 0.529177
    meta = {}
    with open(fname, 'r') as cube:
        cube.readline()
        cube.readline()  # ignore comments
        natm, meta['org'] = _getline(cube)
        nx, meta['xvec'] = _getline(cube)
        ny, meta['yvec'] = _getline(cube)
        nz, meta['zvec'] = _getline(cube)

        if nx > 0:
            meta['xvec'] = meta['xvec'] * bohr
        if ny > 0:
            meta['yvec'] = meta['yvec'] * bohr
        if nz > 0:
            meta['zvec'] = meta['zvec'] * bohr

        meta['atoms'] = [_getline(cube) for i in range(natm)]
        data = np.zeros((nx * ny * nz))
        idx = 0
        for line in cube:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1

    data = np.reshape(data, (nx, ny, nz))
    return data, meta


class Field(object):
    """ """

    def __init__(self, path):

        self._origin_shift = np.array([0.0, 0.0, 0.0])
        self._cube = []
        self._origin_has_changed = False
        self._rot_mat = []
        self._atoms = []

        if path.endswith('.cube') or path.endswith('.cub'):
            data, meta = read_cube(path)
            data = np.swapaxes(data, 1, 2)

            shape = data.shape
            x = np.linspace(0.0, meta['xvec'][0] * shape[0], shape[0]) - 0.5 * meta['xvec'][0] * shape[0]
            z = np.linspace(0.0, meta['yvec'][1] * shape[1], shape[1]) - 0.5 * meta['yvec'][1] * shape[1]
            y = np.linspace(0.0, meta['zvec'][2] * shape[2], shape[2]) - 0.5 * meta['zvec'][2] * shape[2]

            self._cube = [[x[0], y[0], z[0]], [x[-1], y[-1], z[-1]]]

            self.origin = (0, 0, 0)
            self._shape = shape
            self._atoms = meta['atoms']

        data[data > 100] = 100
        data[data < -100] = -100
        self._interpolant = RegularGridInterpolator((x, y, z), data, bounds_error=False)

    def set_origin(self, origin):
        """Set the coordinates of the center of the molecule

        Parameters
        ----------
        origin :
            return:

        Returns
        -------

        """

        if isinstance(origin, list):
            origin = np.array(origin)

        self._origin_shift = origin
        self._origin_has_changed = True

    def rotate(self, axis, theta):
        """Set the coordinates of the center of the molecule

        Parameters
        ----------
        origin :
            return:
        axis :
            
        theta :
            

        Returns
        -------

        """
        if axis == 'x':
            rot_mat = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(theta), -np.sin(theta)],
                                [0.0, np.sin(theta), np.cos(theta)]])
        elif axis == 'y':
            rot_mat = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(theta), 0.0, np.cos(theta)]])
        elif axis == 'z':
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 1.0]])
        else:
            raise ValueError('Wrong axis')

        self._rot_mat.append(rot_mat)

    def reset_rotations(self):
        """ """

        self._rot_mat = []

    def _transform(self, coords1, translate):
        """

        Parameters
        ----------
        coords1 :
            
        translate :
            

        Returns
        -------

        """

        coords = coords1.copy()

        if len(coords.shape) < 2:
            coords = [coords]

        for j in range(len(coords)):
            coords[j] += self.origin

            if self._origin_has_changed:
                coords[j] = coords[j] - self._origin_shift

            if isinstance(translate, np.ndarray):
                coords[j] = coords[j] - np.squeeze(translate)

            if len(self._rot_mat) > 0:
                for item in self._rot_mat:
                    coords[j] = np.dot(item, coords[j].T).T[0]

        return coords

    def _inv_transform(self, coords1, translate):
        """

        Parameters
        ----------
        coords1 :
            
        translate :
            

        Returns
        -------

        """

        coords = coords1.copy()

        if len(coords.shape) < 2:
            coords = [coords]

        for j in range(len(coords)):

            if len(self._rot_mat) > 0:
                for item in reversed(self._rot_mat):
                    coords[j] = np.dot(np.linalg.inv(item), coords[j].T).T[0]

            if isinstance(translate, np.ndarray):
                coords[j] = coords[j] + np.squeeze(translate)

            if self._origin_has_changed:
                coords[j] = coords[j] + self._origin_shift

            coords[j] -= self.origin

        return coords

    def get_values(self, coords1, translate=None):
        """

        Parameters
        ----------
        coords1 :
            
        translate :
             (Default value = None)

        Returns
        -------

        """

        coords = self._transform(coords1, translate)
        values = self._interpolant(coords)

        return np.nan_to_num(values)

    def get_atoms(self):
        """ """

        ind = []
        coords = []

        for item in self._atoms:
            ind.append(item[0])
            coords.append(item[1][1:])

        return np.array(ind), np.array(coords)

    def add_screening(self, eps, mol_y_length, spacing):
        """

        Parameters
        ----------
        eps :
            
        mol_y_length :
            
        spacing :
            

        Returns
        -------

        """

        # from negf.poisson import laplacian, PoissonSolver

        num_points = 256

        x = np.linspace(-19, 19, num_points - 1)
        y = np.linspace(-19, 19, num_points - 1)
        z = np.linspace(-19, 19, num_points - 1)

        X = np.meshgrid(x, y, z)

        coords = np.vstack((X[0].flatten(), X[1].flatten(), X[2].flatten())).T

        vec = np.zeros(3)
        eps[1] = mol_y_length * 0.5
        vec[eps[0]] = 1.0
        vec = np.array(vec)

        rot_mat = np.identity(3)
        # rot_mat[:2, :2] = np.matrix(np.sqrt(1.0 - self._rot_mat[0][1:, 1:]**2))
        rot_mat[:2, :2] = self._rot_mat[0][1:, 1:]
        rot_mat[0, 1] = -rot_mat[0, 1]
        rot_mat[1, 0] = -rot_mat[1, 0]
        # rot_mat = np.linalg.inv(rot_mat)
        # rot_mat = np.sqrt(1.0 - rot_mat ** 2)
        # rot_mat[2, 2] = 1.0
        # rot_mat[0, 0] = -rot_mat[0, 0]
        # rot_mat[1, 1] = -rot_mat[1, 1]

        vec = np.dot(rot_mat, vec.T).T[0]

        dist = X[0] * vec[0] + X[1] * vec[1] + X[2] * vec[2] + eps[1] + \
               0.5 * mol_y_length * (1.0 - np.sin(1.13446)) + \
               spacing

        perm = eps[2] * (1.0 - 0.5 * np.tanh(eps[1] * dist) - 0.5) \
               + eps[3] * (0.5 * np.tanh(eps[1] * dist) + 0.5)

        # perm = np.ones(perm.shape)

        pot = self._interpolant(np.vstack((X[0].flatten(), X[1].flatten(), X[2].flatten())).T).reshape(X[0].shape)
        pot1 = pot / eps
        # dens = -laplacian(pot, x[2] - x[1]) / (4 * np.pi)

        # ps = PoissonSolver()
        # ps.solve_mesh(X[0], X[1], X[2], pot1, dens, perm)

        # pot1 = pot1 + 0.1 * perm / np.max(perm)

        # pot1 = np.swapaxes(pot1, 1, 0)
        self._interpolant = RegularGridInterpolator((x, y, z), pot1, bounds_error=False)


class Field1D(object):
    """ """

    def __init__(self, coord_dependence, axis=2):

        self._func = coord_dependence
        self._axis = axis

    def get_values(self, coords1, translate=None):
        """

        Parameters
        ----------
        coords1 :
            
        translate :
             (Default value = None)

        Returns
        -------

        """

        coords = coords1.copy()
        values = np.zeros(len(coords))

        if len(coords.shape) < 2:
            coords = [coords]

        for j in range(len(coords)):
            if isinstance(translate, np.ndarray):
                coords[j] -= translate

            values[j] = self._func(coords[j][self._axis])

        return values


def main():
    """ """
    import matplotlib.pyplot as plt

    fl = Field(path='/home/mk/gaussian_swarm/gauss_comp/out_neutral.cube')

    x = np.linspace(fl._cube[0][0], fl._cube[1][0], fl._shape[0])
    y = np.linspace(fl._cube[0][1], fl._cube[1][1], fl._shape[1])
    z = np.linspace(fl._cube[0][2], fl._cube[1][2], fl._shape[2])

    x = np.linspace(0, 60, 100)
    y = np.linspace(0, 11, 100)
    z = np.linspace(0, 5.5, 100)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # data = fl.get_values(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T, translate=np.array([0,0,15]))
    # data = data.reshape(X.shape)
    # plt.imshow(data[:, :, 50])
    # # plt.contour(data[:, 50, :], levels=(0.04, 0.06, 0.08, 0.1, 0.2))
    # plt.show()

    fl.set_origin(np.array([20, 11, 2.75]))

    data = fl.get_values(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)
    data = data.reshape(X.shape)
    plt.imshow(data[:, :, 50])

    # plt.contour(data[:, 50, :], levels=(0.04, 0.06, 0.08, 0.1, 0.2))

    plt.show()


def laplacian(input_mat, dx):
    """

    Parameters
    ----------
    input_mat :
        
    dx :
        

    Returns
    -------

    """
    top = input_mat[0:-2, 1:-1, 1:-1]
    left = input_mat[1:-1, 0:-2, 1:-1]
    bottom = input_mat[2:, 1:-1, 1:-1]
    right = input_mat[1:-1, 2:, 1:-1]
    center = input_mat[1:-1, 1:-1, 1:-1]
    back = input_mat[1:-1, 1:-1, 0:-2]
    forward = input_mat[1:-1, 1:-1, 2:]
    return np.pad(top + left + bottom + right + back + forward - 6 * center,
                  ((1, 1), (1, 1), (1, 1)),
                  'constant',
                  constant_values=((0, 0), (0, 0), (0, 0))) / dx ** 2


def main1():
    """ """
    import matplotlib.pyplot as plt

    # fl = Field(path='/home/mk/gaussian_swarm/gauss_comp/wB_ion.cube')
    # fl = Field(path='../cubefil.cube')
    # fl1 = Field(path='/home/mk/gaussian_swarm/gauss_comp/cam_ion.cube')

    fl = Field(path='/home/mk/tetracene_dft_wB_pcm_38_32.cube')
    fl1 = Field(path='/home/mk/tetracene_dft_wB_pcm_38_32_wpcm.cube')

    x = np.linspace(fl._cube[0][0], fl._cube[1][0], fl._shape[0])
    y = np.linspace(fl._cube[0][1], fl._cube[1][1], fl._shape[1])
    z = np.linspace(fl._cube[0][2], fl._cube[1][2], fl._shape[2])

    x = np.linspace(-30, 30, 200)
    y = np.linspace(-30, 30, 200)
    z = np.linspace(-30, 30, 200)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # data = fl.get_values(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T, translate=np.array([0,0,15]))
    # data = data.reshape(X.shape)
    # plt.imshow(data[:, :, 50])
    # # plt.contour(data[:, 50, :], levels=(0.04, 0.06, 0.08, 0.1, 0.2))
    # plt.show()

    # fl.set_origin(np.array([0.0, 0.0, 1.0]))
    # fl1.set_origin(np.array([0.0, 0.0, 1.0]))

    fl.set_origin(np.array([6.36, 11.86, 2.75]))
    fl.rotate('z', 1.13446)  # 65 degrees
    data = fl.get_values(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)
    data = data.reshape(X.shape)

    data1 = fl1.get_values(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)
    data1 = data1.reshape(X.shape)

    cut_level = 0.015

    data1[data1 > cut_level] = cut_level
    data[data > cut_level] = cut_level

    from matplotlib import cm
    norm = cm.colors.Normalize(vmax=cut_level, vmin=-cut_level)
    # cmap = cm.PRGn
    # cmap = cm.seismic
    cmap = cm.coolwarm
    levels = np.arange(-cut_level * 1.1, cut_level * 1.1, 0.002)
    cset = plt.contourf(X[:, :, 100], Y[:, :, 100], data[:, :, 100], levels, norm=norm,
                        cmap=cm.get_cmap(cmap, len(levels) - 1))

    from matplotlib.patches import Rectangle
    currentAxis = plt.gca()
    someX = -5
    someY = -5
    width = 10
    height = 10
    currentAxis.add_patch(Rectangle((someX, someY), width, height,
                                    alpha=1, fill=None))

    plt.show()

    print('hi')


# class nf(float):
#     def __repr__(self):
#         str = '%.3f eV' % (self.__float__(),)
#         if str[-1] == '0':
#             return '%.2f eV' % self.__float__()
#         else:
#             return '%.3f eV' % self.__float__()


class nf(float):
    """ """
    def __repr__(self):
        if self.__float__() == 0.0:
            return '0.0'
        else:
            return '%.1e eV' % (self.__float__(),)


def plot(x, y, data, data1, cut_level, n_contours):
    """

    Parameters
    ----------
    x :
        
    y :
        
    data :
        
    data1 :
        
    cut_level :
        
    n_contours :
        

    Returns
    -------

    """
    X, Y = np.meshgrid(x, y, indexing='ij')

    data[data > cut_level] = cut_level

    from matplotlib import cm
    norm = cm.colors.Normalize(vmax=cut_level, vmin=-cut_level)
    # cmap = cm.PRGn
    # cmap = cm.seismic
    cmap = cm.coolwarm
    levels = np.arange(-cut_level * 1.1, cut_level * 1.1, 2 * cut_level / n_contours)
    # levels = np.linspace(-0.1, 0.1, 21)
    # levels = np.linspace(-0.01, 0.01, 21)
    fig, ax = plt.subplots(figsize=(10, 10))
    cset = ax.contourf(X, Y, data, levels, norm=norm,
                       cmap=cm.get_cmap(cmap, len(levels) - 1))
    cset = ax.contour(X, Y, data, levels, norm=norm,
                      colors='r', linewidths=1)

    cset = ax.contour(X, Y, data1, levels, norm=norm, colors='k')

    # Recast levels to new class
    cset.levels = [nf(val) for val in cset.levels]

    # Label levels with specially formatted floats

    fmt = '%r'

    ax.clabel(cset, cset.levels[9:-9], inline=True, fmt=fmt, fontsize=14)
    # ax.clabel(cset, cset.levels, inline=True, fmt=fmt, fontsize=14)
    ax.axis('off')
    l = mlines.Line2D([10, 15], [-15, -15], linewidth=3, color='k')
    ax.add_line(l)
    rc('font', **{'family': 'monospace', 'weight': 'bold'})
    rc('text', usetex=True)
    ax.text(12, -14.4, r'$5 \mathring{A}$', fontsize=16)


if __name__ == '__main__':
    main1()

    # path = '/home/mk/tetracene_dft_wB_pcm_38_32.cube'
    # data, meta = read_cube(path)
    # path = '/home/mk/tetracene_dft_wB_pcm_38_32_wpcm.cube'
    # data1, meta = read_cube(path)
    # x = np.linspace(0.0, meta['xvec'][0] * data.shape[0], data.shape[0]) - 0.5 * meta['xvec'][0] * data.shape[0]
    # data = data.reshape((len(x), len(x), len(x)))
    # cut_level = 0.015
    # cut_level = 0.01
    # plot(x, x, data[:, 100, :] / 3.8, data1[:, 100, :] / 3.8, cut_level, 20)
    # # plot(x, x, data1[:, 100, :] / 3.8, cut_level, 20)
    # plt.show()
