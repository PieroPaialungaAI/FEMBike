import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from elemForm import frameForm


# Major Nodes in Bike Frame
mNodes = np.array([[0, 0, 0], [-3, 1, 1], [-3, 1, -1], [0, 4, 0], [6, 3, 0], [6, 4, 0]], dtype=float)
# mNodes = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
# Major Node Pairs defining the Bike Frame
mPairs = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [3, 4], [4, 6], [5, 6]], dtype=int)
# mPairs = np.array([[1, 2]], dtype=int)
# Number of Elements in each piece of the Bike Frame
nElem = np.array([20, 20, 20, 20, 20, 20, 20, 20], dtype=int)
# nElem = np.array([2], dtype=int)

# Define element constants
eM = 1  # Elastic Modulus
sM = 1  # Shear Modulus
rho = 1  # Density
area = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # Element Areas
mI = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Element Moment of Inertia
mJ = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # Element Polar Moment of Inertia


def frameForm(mNodes, mPairs, nElem, eM, sM, rho, area, mI, mJ):
    nPairs = np.shape(mPairs)[0]
    unitV = np.zeros((nPairs, 3), dtype=float)
    distV = np.zeros((nPairs, 1), dtype=float)
    nodeDF = pd.DataFrame(data=np.concatenate((mNodes, np.zeros((np.sum(nElem) - len(nElem), 3)))),
                          columns=['X', 'Y', 'Z'])
    elemDF = pd.DataFrame(data=np.zeros((np.sum(nElem), 11)),
                          columns=['Node 1', 'Node 2', 'A', 'E', 'G', 'rho', 'a', 'Ix', 'Iy', 'Iz', 'Jx'])
    nInd = len(mNodes)
    eInd = 0
    for ind in range(nPairs):
        ind1 = mPairs[ind, 0] - 1
        ind2 = mPairs[ind, 1] - 1
        temp1 = mNodes[ind1, :]
        temp2 = mNodes[ind2, :]
        unitV = temp2 - temp1
        distV = np.sqrt(np.sum(np.square(unitV)))
        unitV = unitV / distV
        eNum = nElem[ind]
        dD = distV / eNum
        conLst = [area[ind], eM, sM, rho, dD/2]+list(mI[ind, :])+[mJ[ind]]
        for node in range(eNum):
            if node == 0:
                elemDF.loc[eInd] = [ind1, nInd]+conLst
                eInd += 1
            else:
                if node == eNum - 1:
                    elemDF.loc[eInd] = [nInd, ind2]+conLst
                    eInd += 1
                else:
                    elemDF.loc[eInd] = [nInd, nInd + 1]+conLst
                    eInd += 1
                nodeDF.loc[nInd] = temp1 + dD * unitV * node
                nInd += 1
    return nodeDF, elemDF

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

def kMat(A, E, G, a, Iy, Iz, Jx):
    a2V = a**2
    a3V = a**3
    k = np.zeros((12, 12), dtype=float)
    k[0, 0] = A*E/(2*a)
    k[0, 6] = -A*E/(2*a)
    k[1, 1] = 3*E*Iz/(2*a3V)
    k[1, 5] = 3*E*Iz/(2*a2V)
    k[1, 7] = -k[1, 1]
    k[1, 11] = k[1, 5]
    k[2, 2] = 3*E*Iy/(2*a3V)
    k[2, 4] = 3*E*Iy/(2*a2V)
    k[2, 8] = -k[2, 2]
    k[2, 10] = -3*E*Iy/(2*a2V)
    k[3, 3] = G*Jx/(2*a)
    k[3, 9] = -k[3, 3]
    k[4, 4] = 2*E*Iy/a
    k[4, 8] = 3*E*Iy/(2*a2V)
    k[4, 10] = E*Iy/a
    k[5, 5] = 2*E*Iz/a
    k[5,7] = -3*E*Iz/(2*a2V)
    k[5,11] = E*Iz/a
    k[6,6] = A*E/(2*a)
    k[7,7] = 3*Iz*E/(2*a3V)
    k[7,11] = -3*Iz*E/(2*a2V)
    k[8,8] = k[7,7]
    k[8,10] = -k[7,11]
    k[9,9] = k[3,3]
    k[10,10] = k[4,4]
    k[11,11] = k[5,5]
    k = np.tril(k.T) + np.triu(k, 1)
    # k[np.array([0, 6]), np.array([0, 6])] = A*E/(2*a)*np.ones((1, 2))
    # k[np.array([1, 2, 7, 8]), np.array([1, 2, 7, 8])] = 3*E*Iz/(2*a3V)* np.ones((1, 4))
    # k[np.array([3, 9]), np.array([3, 9])] = G*Jx/(2*a) * np.ones((1, 2))
    # k[np.array([4, 5, 10, 11]), np.array([4, 5, 10, 11])] = 2*E*Iy/a * np.ones((1, 4))
    # k[np.array([0, 6]), np.array([6, 0])] = -A*E/(2*a) * np.ones((1, 2))
    # k[np.array([1, 8, 5, 10, 1, 4, 11, 8]), np.array([5, 10, 1, 8, 11, 8, 1, 4])] = 3*E*Iz/(2*a2V) * np.ones((1, 8))
    # k[np.array([1, 2, 7, 8]), np.array([7, 8, 1, 2])] = -3*E*Iz/(2*a3V) * np.ones((1, 4))
    # k[np.array([2, 7, 4, 11]), np.array([4, 11, 2, 7])] = 3*E*Iy/(2*a2V) * np.ones((1, 4))
    # k[np.array([2, 5, 10, 7]), np.array([10, 7, 2, 5])] = 0 * np.ones((1, 4))
    # k[np.array([3, 9]), np.array([9, 3])] = 0 * np.ones((1, 2))
    # k[np.array([4, 5, 10, 11]), np.array([10, 11, 4, 5])] = 0 * np.ones((1, 4))
    return k


def glob(nodeDF, elemDF):
    nNum = len(nodeDF)
    massG = np.zeros((nNum*6, nNum*6), dtype=float)
    kG = np.zeros((nNum*6, nNum*6), dtype=float)
    for index, row in elemDF.iterrows():
        tempM = massMat(row['rho'], row['A'], row['a'], row['Jx'])
        tempK = kMat(row['A'], row['E'], row['G'], row['a'], row['Iy'], row['Iz'], row['Jx'])
        node1 = int(row['Node 1'])
        node2 = int(row['Node 2'])
        #Mass Matrix Assembly
        massG[6*node1:6*(node1+1), 6*node1:6*(node1+1)] += tempM[:6, :6]
        massG[6*node2:6*(node2+1), 6*node2:6*(node2+1)] += tempM[6:, 6:]
        massG[6*node1:6*(node1+1), 6*node2:6*(node2+1)] += tempM[:6, 6:]
        massG[6*node2:6*(node2+1), 6*node1:6*(node1+1)] += tempM[6:, :6]
        #Stiffness Matrix Assembly
        kG[6 * node1:6 * (node1 + 1), 6 * node1:6 * (node1 + 1)] += tempK[:6, :6]
        kG[6 * node2:6 * (node2 + 1), 6 * node2:6 * (node2 + 1)] += tempK[6:, 6:]
        kG[6 * node1:6 * (node1 + 1), 6 * node2:6 * (node2 + 1)] += tempK[:6, 6:]
        kG[6 * node2:6 * (node2 + 1), 6 * node1:6 * (node1 + 1)] += tempK[6:, :6]
    return massG, kG


df1, df2 = frameForm(mNodes, mPairs, nElem, eM, sM, rho, area, mI, mJ)
print(df1)
print(df2)

mM, kM = glob(df1, df2)
print(np.shape(mM))
print(np.shape(kM))