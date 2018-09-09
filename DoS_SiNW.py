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


def delta(e, eps, h):
    if e - (1 / (2 * h)) < eps < e + (1 / (2 * h)):
        return 1
    else:
        return 0


def dos(eps, kk):
    h = 0.001
    e = np.linspace(eps, kk, h)
    dos = np.zeros(e.shape)
    for j, en in enumerate(e):
        for j1, kk in enumerate(kk):
            dos[j] += delta(en, eps[j], h)
    return dos

