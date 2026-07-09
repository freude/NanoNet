import numpy as np
from ase import Atoms


def hamiltonian2aroms(hamiltonian):

    atoms = Atoms(positions=np.array(list(hamiltonian.atom_list.values())))

    if hamiltonian.ct is not None:
        pcv = hamiltonian.ct.pcv

        if pcv.shape[1] < 3:
            pcv = np.pad(pcv, ((0, 0), (0, 3 - pcv.shape[1])), mode='constant')

        pbc = (pcv != 0).any(axis=0).astype(int)
        pbc1 = (pcv != 0).any(axis=0).astype(int)
        row_idx = np.where(pbc1 == 0)[0]
        col_idx = np.where(pbc == 0)[0]
        pairs = np.array(np.meshgrid(row_idx, col_idx)).T.reshape(-1, 2)

        if pcv.shape[0] < 3:
            pcv = np.pad(pcv, ((0, 3 - pcv.shape[0]), (0, 0)), mode='constant')
            default_size_supercel = np.max(pcv) * 10
            for item in pairs:
                pcv[item[0], item[1]] = default_size_supercel

        atoms.set_cell(pcv)
        atoms.set_pbc(pbc)

        try:
            atoms.set_chemical_symbols(list(hamiltonian.atom_list.keys()))
        except KeyError:
            pass  # ignore missing key and continue

    return atoms


if __name__=='__main__':

    from pathlib import Path
    import sys
    from ase.visualize import view

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from examples.graphene_bilayer_rect import make_hamiltonian

    h = make_hamiltonian(0)
    atoms = hamiltonian2aroms(h)
    view(atoms)
