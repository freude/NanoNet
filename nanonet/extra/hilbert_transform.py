"""
This function generates the weights required for the discrete
Hilbert transform. Re[G] = -Im[G]*phi, Im[G] = Re[G]*phi.

The reason for a discrete hilbert transform is that the common
fast fourier transform method erroneously pins the the real
part at the edges leading to significant deviation from the
true result. For a given energy grid, this method generates
a matrix to which you multiply your signal, after which the
method returns the Hilbert transform of that function. This
method assumes that the functions are localised to the domain
window, if there is an approximately constant tail in the
positive/negative direction, an additional term must be added.
If the function is actually periodic we recommend you use the
FFT method, as it will be better suited to your problem.

If used, please cite
Jesse Vaitkus' thesis 2020
"""


import numpy as np


def make_hilbert_weights(E, **kwargs):
    """This function generates the weights required for the discrete
       Hilbert transform. This method assumes that each site can be
       describe using linearly interpolating basis functions. Other
       basis functions are planned.

       Re[G] = -Im[G]*phi, Im[G] = Re[G]*phi.

        Parameters
        ----------
        E : numpy array
            Energy grid to compute weights for
        
        kwargs : dict
              additional named arguments:
                eta : float (Default value = sqrt(eps) )
                      Imaginary part to be used for stability, should
                      be as small as possible without breaking

        Returns
        -------
        numpy.matrix

            #Guide for usage.

            A user provides an energy grid and the method returns an
            array phi such that Re[G] = -Im[G]*phi, Im[G] = Re[G]*phi.
        """

    eta = kwargs.get('eta', np.sqrt(np.finfo(float).eps))

    numE = np.size(E)
    dE = (E[-1] - E[0])/(numE - 1)  # This is the average separation of points,
    Eneg = E[0] - dE                # which we use to generate the ghost points
    Epos = E[-1] + dE               # Eneg/Epos to handle the domain boundaries.

    phi = np.empty( (numE, numE), dtype=complex)

    phi[0, :] = (E + 1j*eta - Eneg)/(E[0] - Eneg)*np.log((Eneg - E - 1j*eta)/(E[0] - E - 1j*eta)) + \
                (E + 1j*eta - E[1])/(E[0] - E[1])*np.log((E + 1j*eta - E[0])/(E + 1j*eta - E[1]))

    for ii in range(1, numE-1):
        phi[ii, :] = (E + 1j * eta - E[ii-1])/(E[ii] - E[ii-1])*np.log((E[ii-1] - E - 1j*eta)/(E[ii] - E - 1j*eta)) + \
                     (E + 1j * eta - E[ii+1])/(E[ii] - E[ii+1])*np.log((E + 1j*eta - E[ii])/(E + 1j * eta - E[ii+1]))

    phi[-1, :] = (E + 1j * eta - E[-2]) / (E[-1] - E[-2]) * np.log((E[-2] - E - 1j * eta) / (E[-1] - E - 1j * eta)) + \
                 (E + 1j * eta - Epos) / (E[-1] - Epos) * np.log((E + 1j * eta - E[-1]) / (E + 1j * eta - Epos))

    return np.real(phi)/np.pi  # Discard imaginary part used for stability in computation.
