import numpy as np
import matplotlib.pyplot as plt
from SiNWfunction import bs

"""
below is a test function using the rectangular rule
low and high are the bounds of integration, f is f(x), 
N is the number of rectangles. Test function has been commented out,
while the density of states function is below it.   
"""
# def rectangular_rule(low, high, f, N):
#     E = np.linspace(low, high, N)
#     area = 0
#     for r in range(10000):
#         area += f(low+r*(high-low)/N)*(high-low)/N
#     return (area)
#
#
# def polynomial(x):
#     return(3*x**2)
# rectangular_rule(0, 100, polynomial, 10000)
#
# print rectangular_rule(0, 100, polynomial, 10000)
"below is the density of states function and delta function"


def delta(E, eps, h):
    if E - h < eps < E + h:
        return 1
    else:
        return 0


def dos(E, bs, kk, h):

    dos = np.zeros(E.shape)
    for j, en in enumerate(E):
        for j1 in range(len(kk)):
            dos[j] += delta(en, bs[j1], h)
    return dos


if __name__ == '__main__':

    num_points = 100
    kk = np.linspace(0, 0.57, num_points, endpoint=True)
    Eg, bstruct = bs(path='c:\users\sammy\desktop\NanoNet\input_samples', kk=kk, flag=True)

    E = np.linspace(-2, 0, 200)
    h = 0.01

    dos1 = np.zeros(E.shape)

    for j in range(bstruct[0].shape[1]):
        dos1 += dos(E, bstruct[0][:, j], kk, h)
    print(dos1)
