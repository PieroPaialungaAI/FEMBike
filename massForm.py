import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def massMat(den, crossA, elemL, secM):
    # den = material density
    # crossA = cross sectional area
    # elemL = length of element
    # secM = second moment of area
    rV = secM/crossA
    aV = elemL/2
    a2V = aV**2
    mass = np.zeros((12, 12), dtype=float)
    mass[np.array([0, 6]), np.array([0, 6])] = 70*np.ones((1, 2))
    mass[np.array([1, 2, 7, 8]), np.array([1, 2, 7, 8])] = 78 * np.ones((1, 4))
    mass[np.array([3, 9]), np.array([3, 9])] = (70*rV) * np.ones((1, 2))
    mass[np.array([4, 5, 10, 11]), np.array([4, 5, 10, 11])] = (8*a2V) * np.ones((1, 4))
    mass[np.array([0, 6]), np.array([6, 0])] = 35 * np.ones((1, 2))
    mass[np.array([1, 8, 5, 10]), np.array([5, 10, 1, 8])] = (22*aV) * np.ones((1, 4))
    mass[np.array([1, 2, 7, 8]), np.array([7, 8, 1, 2])] = 27 * np.ones((1, 4))
    mass[np.array([1, 4, 11, 8]), np.array([11, 8, 1, 4])] = (-13*aV) * np.ones((1, 4))
    mass[np.array([2, 7, 4, 11]), np.array([4, 11, 2, 7])] = (-22*aV) * np.ones((1, 4))
    mass[np.array([2, 5, 10, 7]), np.array([10, 7, 2, 5])] = (13*aV) * np.ones((1, 4))
    mass[np.array([3, 9]), np.array([9, 3])] = (-35*rV) * np.ones((1, 2))
    mass[np.array([4, 5, 10, 11]), np.array([10, 11, 4, 5])] = (-6*a2V) * np.ones((1, 4))
    mass = (den*crossA*aV/105)*mass
    return mass

