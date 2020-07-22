import numpy as np


def fermi_window(energy, tempr):
    """

    Parameters
    ----------
    energy :
        
    tempr :
        

    Returns
    -------

    """

    kb = 8.61733e-5  # Boltzmann constant in eV

    return (1.0 / (4.0 * kb * tempr)) / np.cosh(energy / (2 * kb * tempr))


def tr2cond(energy, tr, tempr=300):
    """

    Parameters
    ----------
    energy :
        
    tr :
        
    tempr :
         (Default value = 300)

    Returns
    -------

    """

    tr = np.pad(tr, 30000, 'edge')
    energy = np.pad(energy, 30000,'linear_ramp')
    ans = np.convolve(tr, fermi_window(energy, tempr), mode='same')

    return energy, ans