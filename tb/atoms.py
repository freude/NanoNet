"""
Module contains classes describing atoms with different number of basis functions
"""
import sys


class Atom(object):
    """
    This is the parent class for all basis sets for all atoms. It also contains a factory function,
    which generates objects of Atom from a list of labels and the dictionary `orbital_sets` making
    a correspondence between an atom and its basis set
    """

    orbital_sets = {}

    def __init__(self, title):
        self.title = title
        self.orbitals = []
        self.num_of_orbitals = 0

    def add_orbital(self, title, energy=0.0, principal=0, orbital=0, magnetic=0, spin=0):
        """
        Adds an orbital to the set of orbitals

        :param title:        a string representing an orbital label,
                             it usually specifies its symmetry, e.g. 's', 'px', 'py' etc.
        :param energy:       energy of the orbital
        :param principal:    principal quantum number `n`
        :param orbital:      orbital quantum number `l`
        :param magnetic:     magnetic quantum number `m`
        :param spin:         spin quantum number `s`
        """

        orbital = {'title': title,
                   'energy': energy,
                   'n': principal,
                   'l': orbital,
                   'm': magnetic,
                   's': spin}

        self.orbitals.append(orbital)
        self.num_of_orbitals += 1

    @staticmethod
    def atoms_factory(labels):
        """
        Taking a list of labels creates a dictionary of `Atom` objects
        from those labels. The set of orbitals for each atom and corresponding class is
        specified in the class variable `orbital_sets`

        :param labels:   list of labels
        :type labels:    list(str)
        :return:         dictionary of `Atom` objects
        """

        output = {}

        for label in labels:

            try:
                key = ''.join([i for i in label if not i.isdigit()])
                atom = Atom.orbital_sets[key]
                if not isinstance(atom, Atom):
                    raise KeyError
            except KeyError:
                # TODO: simplify these statements below
                if label.lower().startswith('si'):
                    atom = getattr(sys.modules[__name__], Atom.orbital_sets['Si'])()
                elif label.lower().startswith('h'):
                    atom = getattr(sys.modules[__name__], Atom.orbital_sets['H'])()
                elif label.lower().startswith('b'):
                    atom = getattr(sys.modules[__name__], Atom.orbital_sets['Bi'])()
                else:
                    raise ValueError("There is no library entry for the atom " + label)

            output[atom.title] = atom

        return output


class SiliconSP3D5S(Atom):
    """
    Class defines the `sp3d5s*` basis set for the silicon atom
    """

    def __init__(self):

        super(SiliconSP3D5S, self).__init__("Si")

        self.add_orbital("s", energy=-2.0196)
        self.add_orbital("c", energy=19.6748, principal=1)
        self.add_orbital("px", energy=4.5448, orbital=1, magnetic=-1)
        self.add_orbital("py", energy=4.5448, orbital=1, magnetic=1)
        self.add_orbital("pz", energy=4.5448, orbital=1, magnetic=0)
        self.add_orbital("dz2", energy=14.1836, orbital=2, magnetic=-1)
        self.add_orbital("dxz", energy=14.1836, orbital=2, magnetic=-2)
        self.add_orbital("dyz", energy=14.1836, orbital=2, magnetic=2)
        self.add_orbital("dxy", energy=14.1836, orbital=2, magnetic=1)
        self.add_orbital("dx2my2", energy=14.1836, orbital=2, magnetic=0)


class HydrogenS(Atom):
    """
    Class defines the simplest basis set for the hydrogen atom,
    consisting of a single s-orbital
    """

    def __init__(self):

        super(HydrogenS, self).__init__("H")

        self.add_orbital("s", energy=0.9998)


class Bismuth(Atom):
    """
    Class defines the `sp3` basis set for the bismuth atom
    """

    def __init__(self):

        super(Bismuth, self).__init__("Bi")

        self.add_orbital("s", energy=-10.906, principal=0)

        # self.add_orbital("s1", energy=-10.906, principal=0, spin=1)

        self.add_orbital("px", energy=-0.486, orbital=1, magnetic=-1)
        self.add_orbital("py", energy=-0.486, orbital=1, magnetic=1)
        self.add_orbital("pz", energy=-0.486, orbital=1, magnetic=0)

        # self.add_orbital("px1", energy=-0.486, orbital=1, magnetic=-1, spin=1)
        # self.add_orbital("py1", energy=-0.486, orbital=1, magnetic=1, spin=1)
        # self.add_orbital("pz1", energy=-0.486, orbital=1, magnetic=0, spin=1)
