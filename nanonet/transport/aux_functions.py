import numpy as np
from nanonet.tb.constants import kB_eV


def fd(energy, ef, temp):
    """
    Fermi-Dirac function

    Parameters
    ----------
    energy :

    ef :

    temp :


    Returns
    -------

    """

    return 1.0 / (1.0 + np.exp((energy - ef) / (kB_eV * temp)))


def fermi_window(energy, tempr):
    """

    Parameters
    ----------
    energy :

    tempr :


    Returns
    -------

    """

    return (1.0 / (4.0 * kB_eV * tempr)) / np.cosh(energy / (2 * kB_eV * tempr))