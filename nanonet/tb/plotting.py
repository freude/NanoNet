import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_atom_positions(atom_list, virtual_and_interfacial_atoms, radial_dep):
    """

    Parameters
    ----------
    atom_list :
        
    virtual_and_interfacial_atoms :
        
    radial_dep :
        

    Returns
    -------

    """

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # coordinates_to_plot = np.asarray(list(self.atom_list.values()))
    # ax.scatter(coordinates_to_plot[:, 0], coordinates_to_plot[:, 1], coordinates_to_plot[:, 2], c='red', s=100)
    #
    # map1 = [item.startswith('*_') for item in list(self.ct.virtual_and_interfacial_atoms.keys())]
    # map2 = [item.startswith('**_') for item in list(self.ct.virtual_and_interfacial_atoms.keys())]
    #
    # coordinates_to_plot = np.asarray(list(self.ct.virtual_and_interfacial_atoms.values()))
    # ax.scatter(coordinates_to_plot[map1, 0], coordinates_to_plot[map1, 1], coordinates_to_plot[map1, 2],
    #            c='green', s=70)
    # ax.scatter(coordinates_to_plot[map2, 0], coordinates_to_plot[map2, 1], coordinates_to_plot[map2, 2],
    #            s=20)
    #
    # ax.set_xlim(-6, 6)
    # ax.set_ylim(-6, 6)
    # ax.set_zlim(-6, 6)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    coordinates_of_atoms_in_unit_cell = np.asarray(list(atom_list.values()))
    ax.scatter(coordinates_of_atoms_in_unit_cell[:, 0], coordinates_of_atoms_in_unit_cell[:, 1], coordinates_of_atoms_in_unit_cell[:, 2], c='k', s=60)

    coordinates_of_atoms_outside_of_unit_cell = np.asarray(list(virtual_and_interfacial_atoms.values()))
    for ii in range(len(atom_list)):
        for jj in range(len(virtual_and_interfacial_atoms)):
            if radial_dep(coordinates_of_atoms_outside_of_unit_cell[jj] - coordinates_of_atoms_in_unit_cell[ii]) == 1:
                ax.scatter(coordinates_of_atoms_outside_of_unit_cell[jj, 0],
                           coordinates_of_atoms_outside_of_unit_cell[jj, 1],
                           coordinates_of_atoms_outside_of_unit_cell[jj, 2],
                           c='r', s=30, alpha=0.3)
            elif radial_dep(coordinates_of_atoms_outside_of_unit_cell[jj] - coordinates_of_atoms_in_unit_cell[ii]) == 2:
                ax.scatter(coordinates_of_atoms_outside_of_unit_cell[jj, 0],
                           coordinates_of_atoms_outside_of_unit_cell[jj, 1],
                           coordinates_of_atoms_outside_of_unit_cell[jj, 2],
                           c='g', s=20, alpha=0.3)
            elif radial_dep(coordinates_of_atoms_outside_of_unit_cell[jj] - coordinates_of_atoms_in_unit_cell[ii]) == 3:
                ax.scatter(coordinates_of_atoms_outside_of_unit_cell[jj, 0],
                           coordinates_of_atoms_outside_of_unit_cell[jj, 1],
                           coordinates_of_atoms_outside_of_unit_cell[jj, 2],
                           c='b', s=10, alpha=1.0)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
    plt.show()


def plot_bs_split(kk, vba, cba):
    """

    Parameters
    ----------
    kk :
        
    vba :
        
    cba :
        

    Returns
    -------

    """

    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylim(-1.0, -0.3)
    ax[0].plot(kk, np.sort(np.real(vba)), 'k')
    ax[0].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax[0].set_ylabel(r'Energy (eV)')
    ax[0].set_title('Valence band')
    # plt.savefig('bs_vb.pdf')

    ax[1].set_ylim(2.0, 2.7)
    ax[1].plot(kk, np.sort(np.real(cba)), 'k')
    ax[1].set_xlabel(r'Wave vector ($\frac{\pi}{a}$)')
    ax[1].set_ylabel(r'Energy (eV)')
    ax[1].set_title('Conduction band')
    fig.tight_layout()
    # plt.savefig('bs_cb.pdf')
    plt.show()


def plot_atom_positions1(h, virtual_and_interfacial_atoms, radial_dep):
    """

    Parameters
    ----------
    h :
        
    virtual_and_interfacial_atoms :
        
    radial_dep :
        

    Returns
    -------

    """

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # coordinates_to_plot = np.asarray(list(self.atom_list.values()))
    # ax.scatter(coordinates_to_plot[:, 0], coordinates_to_plot[:, 1], coordinates_to_plot[:, 2], c='red', s=100)
    #
    # map1 = [item.startswith('*_') for item in list(self.ct.virtual_and_interfacial_atoms.keys())]
    # map2 = [item.startswith('**_') for item in list(self.ct.virtual_and_interfacial_atoms.keys())]
    #
    # coordinates_to_plot = np.asarray(list(self.ct.virtual_and_interfacial_atoms.values()))
    # ax.scatter(coordinates_to_plot[map1, 0], coordinates_to_plot[map1, 1], coordinates_to_plot[map1, 2],
    #            c='green', s=70)
    # ax.scatter(coordinates_to_plot[map2, 0], coordinates_to_plot[map2, 1], coordinates_to_plot[map2, 2],
    #            s=20)
    #
    # ax.set_xlim(-6, 6)
    # ax.set_ylim(-6, 6)
    # ax.set_zlim(-6, 6)
    # plt.show()

    atom_list = h.atom_list
    virtual_and_interfacial_atoms = h.ct.virtual_and_interfacial_atoms

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    neigbours_in_unit_cell = [h.get_neighbours(coord) for coord in list(atom_list.values())]
    coordinates_of_atoms_in_unit_cell = [list(atom_list.values())[item] for item in np.array(neigbours_in_unit_cell).flatten()]
    coordinates_of_atoms_in_unit_cell = np.array(coordinates_of_atoms_in_unit_cell)
    ax.scatter(coordinates_of_atoms_in_unit_cell[:, 0], coordinates_of_atoms_in_unit_cell[:, 1], coordinates_of_atoms_in_unit_cell[:, 2], c='k', s=60)

    neigbours_out_unit_cell = [h.ct.get_neighbours(coord) for coord in list(atom_list.values())]

    coordinates_of_atoms_out_unit_cell1 = [list(virtual_and_interfacial_atoms.values())[item] for item in
                                          np.array(neigbours_out_unit_cell)[0]]

    coordinates_of_atoms_out_unit_cell2 = [list(virtual_and_interfacial_atoms.values())[item] for item in
                                          np.array(neigbours_out_unit_cell)[1]]
    coordinates_of_atoms_outside_of_unit_cell1 = np.array(coordinates_of_atoms_out_unit_cell1)
    coordinates_of_atoms_outside_of_unit_cell2 = np.array(coordinates_of_atoms_out_unit_cell2)
    # coordinates_of_atoms_outside_of_unit_cell = np.asarray(list(virtual_and_interfacial_atoms.values()))
    ax.scatter(coordinates_of_atoms_outside_of_unit_cell1[:, 0], coordinates_of_atoms_outside_of_unit_cell1[:, 1],
               coordinates_of_atoms_outside_of_unit_cell1[:, 2], c='r', s=30)
    ax.scatter(coordinates_of_atoms_outside_of_unit_cell2[:, 0], coordinates_of_atoms_outside_of_unit_cell2[:, 1],
               coordinates_of_atoms_outside_of_unit_cell2[:, 2], c='b', s=60)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
    plt.show()



