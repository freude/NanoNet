"""
Module contains the class `Orbitals` that allows to generate any user defined basis set based on
the linear combination of atomic orbitals (LCAO). Also, the module contains a set of predefined basis sets
`SiliconSP3D5S`, `HydrogenS`, `Bismuth`.
"""
import sys
from nanonet.tb.aux_functions import print_table


class Orbitals(object):
    """This is the parent class for all basis sets for all atoms. It also contains a factory function,
    which generates objects of the class Orbitals from a list of labels and
    the dictionary `orbital_sets` making a correspondence between an atom and its basis set

    Parameters
    ----------

    Returns
    -------

    """

    orbital_sets = {}

    def __init__(self, title):
        self.title = title
        self.orbitals = []
        self.num_of_orbitals = 0
        Orbitals.orbital_sets[self.title] = self

    def add_orbital(self, title, energy=0.0, principal=0, orbital=0, magnetic=0, spin=0):
        """Adds an orbital to the set of orbitals

        Parameters
        ----------
        title :
            a string representing an orbital label,
            it usually specifies its symmetry, e.g. `s`, `px`, `py` etc.
        energy :
            energy of the orbital (Default value = 0.0)
        principal :
            principal quantum number `n-1` (Default value = 0)
        orbital :
            orbital quantum number `l` (Default value = 0)
        magnetic :
            magnetic quantum number `m` (Default value = 0)
        spin :
            spin quantum number `s` (Default value = 0)

        Returns
        -------

        """

        orbital = {'title': title,
                   'energy': energy,
                   'n': principal,
                   'l': orbital,
                   'm': magnetic,
                   's': spin}

        self.orbitals.append(orbital)
        self.num_of_orbitals += 1

        Orbitals.orbital_sets[self.title] = self

    def generate_info(self):
        """ """

        return print_table(self.orbitals)

    @staticmethod
    def atoms_factory(labels):
        """Taking a list of labels creates a dictionary of `Orbitals` objects
        from those labels. The set of orbitals for each atom and corresponding class is
        specified in the class variable `orbital_sets`

        Parameters
        ----------
        labels : list(str)
            list of labels

        Returns
        -------
        type
            dictionary of `Orbitals` objects

        """

        output = {}

        for label in labels:

            try:
                key = ''.join([i for i in label if not i.isdigit()])
                atom = Orbitals.orbital_sets[key]
                if not isinstance(atom, Orbitals):
                    raise KeyError
            except KeyError:
                # TODO: simplify these statements below
                if label.lower().startswith('si'):
                    atom = getattr(sys.modules[__name__], Orbitals.orbital_sets['Si'])()
                elif label.lower().startswith('h'):
                    atom = getattr(sys.modules[__name__], Orbitals.orbital_sets['H'])()
                elif label.lower().startswith('b'):
                    atom = getattr(sys.modules[__name__], Orbitals.orbital_sets['Bi'])()
                else:
                    raise ValueError("There is no library entry for the atom " + label)

            output[atom.title] = atom

        return output


class SiliconSP3D5S(Orbitals):
    """Class defines the `sp3d5s*` basis set for the silicon atom"""

    def __init__(self):

        super(SiliconSP3D5S, self).__init__("Si")

        self.add_orbital("s", energy=-2.0196, spin=0)
        self.add_orbital("c", energy=19.6748, principal=1, spin=0)
        self.add_orbital("px", energy=4.5448, orbital=1, magnetic=-1, spin=0)
        self.add_orbital("py", energy=4.5448, orbital=1, magnetic=1, spin=0)
        self.add_orbital("pz", energy=4.5448, orbital=1, magnetic=0, spin=0)
        self.add_orbital("dz2", energy=14.1836, orbital=2, magnetic=-1, spin=0)
        self.add_orbital("dxz", energy=14.1836, orbital=2, magnetic=-2, spin=0)
        self.add_orbital("dyz", energy=14.1836, orbital=2, magnetic=2, spin=0)
        self.add_orbital("dxy", energy=14.1836, orbital=2, magnetic=1, spin=0)
        self.add_orbital("dx2my2", energy=14.1836, orbital=2, magnetic=0, spin=0)


class HydrogenS(Orbitals):
    """Class defines the simplest basis set for the hydrogen atom,
    consisting of a single s-orbital

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self):

        super(HydrogenS, self).__init__("H")

        self.add_orbital("s", energy=0.9998)
        # self.add_orbital("s", energy=0.9998, spin=1)


class Bismuth(Orbitals):
    """Class defines the `sp3` basis set for the bismuth atom"""

    def __init__(self):

        super(Bismuth, self).__init__("Bi")

        self.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic=0, spin=0)
        self.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=0)
        self.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic=1, spin=0)
        self.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic=0, spin=0)
        self.add_orbital("s", energy=-10.906, principal=0, orbital=0, magnetic=0, spin=1)
        self.add_orbital("px", energy=-0.486, principal=0, orbital=1, magnetic=-1, spin=1)
        self.add_orbital("py", energy=-0.486, principal=0, orbital=1, magnetic=1, spin=1)
        self.add_orbital("pz", energy=-0.486, principal=0, orbital=1, magnetic=0, spin=1)
