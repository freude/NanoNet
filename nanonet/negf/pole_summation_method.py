import numpy as np
import scipy.linalg as linalg

def pole_maker(Emin,ChemPot,kT,reltol):
    """This is an alternate pole summation method implemented by Areshkin-Nikolic [CITE].
    Similar to the continued-fraction representation of Ozaki, this method allows for
    efficient computation of the density matrix by complex pole summation. Both methods
    approximate the Fermi-Dirac distribution function so that it is more easily manipulable.

    One of the exciting uses of this method is for finite differences, and the computation
    of quasi-equilibrium solutions for use in efficient non-equilibrium density matrices
    as the user can reuse certain poles but with different weights for little extra cost.

    What A-N does is replaces f(E,mu) with one that approximates it on the real line:
    f(E,mu,kT) -> f(E,1j*mu_im,1j*kT_im)*( f(E,mu,kT) - f(Emin, mu_re,kT_re) ) but has
    drastic changes in the upper complex plane. By judicious choice of mu_im to be at
    least p*kT away from the real line and mu_re to be at least p*kT away from the
    minimum eigenvalue Emin, this gives a controlled relative error of e^-p.

    This function automatically switches from first order to second order when
    appropriate. This is when the interval divided by the temperature exceeds 10^3.
    To-do: add third order method.

     Parameters
    ----------
    Emin    : scalar (dtype=numpy.float)
           Minimum occupied state, e.g. a band edge.
    ChemPot : scalar (dtype=numpy.float)
           The chemical potential
    kT      : scalar (dtype=numpy.float)
           The temperature (in units of energy)
    reltol : scalar (dtype=numpy.float)
           The desired relative tolerance. p = -np.log(reltol)

    """
    p = -np.log(reltol)  # Compute the exponent for the relative tolerance desired.

    # When energy exceeds 10^3, switch to second order poles.
    z = (ChemPot-Emin)/kT

    if z < 10**3:
        poles, residues = pole_order_one(Emin, ChemPot, kT, p)
    else:
        poles, residues = pole_order_two(Emin, ChemPot, kT, p)

    return poles, residues


def pole_order_one(Emin,ChemPot,kT, p):

    poles = 1
    residues = 1

    return poles, residues


def pole_order_two(Emin, ChemPot, kT, p):

    poles = 1
    residues = 1

    return poles, residues


