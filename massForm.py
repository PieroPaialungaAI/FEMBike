import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from elemForm import frameForm

# Major Nodes in Bike Frame
mNodes = np.array([[0, 0, 0], [-3, 1, 1], [-3, 1, -1], [0, 4, 0], [6, 3, 0], [6, 4, 0]], dtype=float)
mNodes = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
# Major Node Pairs defining the Bike Frame
mPairs = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [3, 4], [4, 6], [5, 6]], dtype=int)
mPairs = np.array([[1, 2]], dtype=int)
# Number of Elements in each piece of the Bike Frame
nElem = np.array([20, 20, 20, 20, 20, 20, 20, 20], dtype=int)
nElem = np.array([2], dtype=int)

# Define element constants
eM = 1  # Elastic Modulus
sM = 1  # Shear Modulus
rho = 1  # Density
area = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # Element Areas
mI = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Element Moment of Inertia
mJ = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # Element Polar Moment of Inertia

def massMat(rho, area, aV, mJ):
    # den: material density
    # area: cross sectional area
    # elemL: length of element
    # mJ: second moment of area
    rV = mJ/area
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
    mass = (rho*area*aV/105)*mass
    return mass

def massMat2(rho, area, aV, mJ):
    mass = np.ones((12, 12), dtype=float)
    return mass

def massGlob(nodeDF, elemDF):
    nNum = len(nodeDF)
    massG = np.zeros((nNum*6, nNum*6), dtype=float)
    for index, row in elemDF.iterrows():
        temp = massMat2(row['rho'], row['A'], row['a'], row['Jx'])
        node1 = int(row['Node 1'])
        node2 = int(row['Node 2'])
        massG[6*node1:6*(node1+1), 6*node1:6*(node1+1)] += temp[:6, :6]
        # print('break')
        # print(massG[6*node2:6*(node2+1), 6*node2:6*(node2+1)])
        massG[6*node2:6*(node2+1), 6*node2:6*(node2+1)] += temp[6:, 6:]
        massG[6*node1:6*(node1+1), 6*node2:6*(node2+1)] += temp[:6, 6:]
        massG[6*node2:6*(node2+1), 6*node1:6*(node1+1)] += temp[6:, :6]
    return massG


df1, df2 = frameForm(mNodes, mPairs, nElem, eM, sM, rho, area, mI, mJ)
print(df1)
print(df2)
print(massGlob(df1, df2))
