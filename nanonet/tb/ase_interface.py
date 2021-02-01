import numpy as np
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase.visualize import view
from ase import Atoms


def ham2atoms(ham):
    atoms = Atoms()
    atoms.set_positions(np.array(ham.atom_list.values()).T)
    atoms.set_chemical_symbols(list(ham.atom_list.keys()))
    view(atoms)
    return atoms


def atoms2ham(atoms):
    pass

def visualize_hamiltonian():
    fig, ax = plt.subplots()
    plot_atoms(atoms, ax, radii=0.3, rotation=('0x,0y,0z'))
    plt.view()



