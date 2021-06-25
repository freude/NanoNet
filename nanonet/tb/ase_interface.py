import numpy as np
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase import Atoms, Atom


def make_atoms(**kwargs):

    if 'coords' in kwargs:
        coords = kwargs['coords']
        if isinstance(coords, str):
            lines = coords.split(sep='\n')[2:]
            labels = []
            coo = []
            for items in lines:
                if len(items) != 0:
                    data = items.split()
                    label = ''.join(i for i in data[0] if not i.isdigit())
                    labels.append(label)
                    coo.append(np.array(data[1:], dtype=float))
        else:
            coo = coords
            if 'lables' in kwargs:
                labels = kwargs['labels']
            else:
                labels = 'A' * len(coo)

        atoms = Atoms(labels, positions=coo)

        atoms.set_positions(coo)
        atoms.set_chemical_symbols(labels)

    else:
        atoms = Atoms()

    if 'period' in kwargs:
        atoms.set_cell(kwargs['period'])

    return atoms


def unfold_atoms(atoms):
    pass


def ham2atoms(ham):

    labels = []
    for item in list(ham.atom_list.keys()):
        labels.append(''.join(i for i in item if not i.isdigit()))

    atoms = Atoms(labels, list(ham.atom_list.values()))

    virtual = Atoms()
    interface = Atoms()

    if ham.ct is not None:
        vac = np.array([50, 50, 50])
        mask = np.nonzero(np.sum(np.abs(ham.ct.pcv), axis=0))[0]
        vac_diag = np.diag(np.array([50, 50, 50]))
        vac[mask] = 0
        cell = np.vstack((ham.ct.pcv, vac_diag[np.nonzero(vac)[0]]))
        atoms.set_cell(cell)

        for key, item in ham.ct.virtual_and_interfacial_atoms.items():
            if key.startswith('*'):
                virtual += Atom('He', item)
            else:
                interface += Atom('Ne', item)

    return atoms, virtual, interface


def visualize_hamiltonian(atoms):
    fig, ax = plt.subplots()
    plot_atoms(atoms, ax, radii=0.3, rotation=('0x,0y,0z'))
    plt.view()



