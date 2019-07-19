import numpy as np


def fermi_window(energy, tempr):

    kb = 8.61733e-5  # Boltzmann constant in eV

    return (1.0 / (4.0 * kb * tempr)) / np.cosh(energy / (2 * kb * tempr))


def tr2cond(energy, tr, tempr=300):

    tr = np.pad(tr, 'edge')
    energy = np.pad(energy, 'linear_ramp')
    ans = np.convolve(tr, fermi_window(energy, tempr), mode='same')

    return energy, ans