import numpy as np
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.visualize import view
from ase import Atoms


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

    atoms = Atoms()
    atoms.set_positions(np.array(ham.atom_list.values()).T)
    atoms.set_chemical_symbols(list(ham.atom_list.keys()))

    return atoms

def visualize_hamiltonian(atoms):
    fig, ax = plt.subplots()
    plot_atoms(atoms, ax, radii=0.3, rotation=('0x,0y,0z'))
    plt.view()



