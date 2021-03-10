"""
Example function to transform
analytical transform is known
for 1/(1+x^2) is x/(x^2 + 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from nanonet.extra.hilbert_transform import make_hilbert_weights


W = 30
numE = 1501

# This is making a randomly spaced energy grid
# over the energy range we desire, this is to
# show that the method and its solution are
# not sensitive to varying grid shapes.
E = np.random.rand(numE) + 0.25  # 0.25 to avoid excessively small steps
E = W*(np.cumsum(E)/np.sum(E) - 0.5)

phi = make_hilbert_weights(E)

realfun = 1/(E**2 + 1)      # The function we are going to transform
imagfun = E/(E**2 + 1)      # Its analytical hilbert transform
hilbfun = realfun.dot(phi)  # Our numerical hilbert transform

# To compute using the fourier transform we need the data on an
# equally spaced grid, using the same linear interpolation
# hypothesis we resample the data on these axes, note this is
# not evaluating the function at these points, but its interpo-
# lation because we don't "know" the values there.
equalE = np.linspace(-W/2, W/2, 2*numE+1)  # We'll double the sampling to show
realfun2 = np.interp(equalE, E, realfun)   # that it doesn't improve anything
hilbfun2 = np.imag(hilbert(realfun2))  # FFT-based Hilbert transform

plt.plot(E, imagfun, label="Analytical", color='#00AEC7', linestyle='solid')
plt.plot(E, hilbfun, label="Our method", color='#464646', linestyle='dashed')
plt.plot(equalE, hilbfun2, label="FFT method", color='#D40F7D', linestyle=':')
plt.legend()

plt.show()  # Note the slight deviations at the end are
            # due to the function not being exactly zero
            # at the edges, this may be amended by mak-
            # ing some approximation as to the function-
            # al form at the boundaries or by widening
            # the interval until the function has decay-
            # ed a satisfactory amount. Note how the FFT
            # method erroneously clamps the function to
            # the edges/zero to maintain periodicity
