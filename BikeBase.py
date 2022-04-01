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

def massMat(rho, A, aV, Jx):
    # den: material density
    # A: cross sectional A
    # elemL: length of element
    # Jx: second moment of A
    rV = Jx/A
    a2V = aV**2
    mass = np.zeros((12, 12), dtype=float)
    mass[np.array([0, 6]), np.array([0, 6])] += 70
    mass[np.array([1, 2, 7, 8]), np.array([1, 2, 7, 8])] += 78
    mass[np.array([3, 9]), np.array([3, 9])] += 70*rV
    mass[np.array([4, 5, 10, 11]), np.array([4, 5, 10, 11])] += 8*a2V
    mass[np.array([0]), np.array([6])] += 35
    mass[np.array([1, 8, 2, 7]), np.array([5, 10, 4, 11])] += 22*aV
    mass[np.array([1, 2]), np.array([7, 8])] += 27
    mass[np.array([2, 5, 1, 4]), np.array([10, 7, 11, 8])] += 13*aV
    mass[np.array([3]), np.array([9])] += -35*rV
    mass[np.array([4, 5]), np.array([10, 11])] += -6*a2V
    mass[np.array([1, 4, 2, 7]), np.array([11, 8, 4, 11])] *= -1
    mass *= rho*A*aV/105
    mass = np.tril(mass.T) + np.triu(mass, 1)
    return mass

def kMat(A, E, G, a, Iy, Iz, Jx):
    k = np.zeros((12, 12), dtype=float)
    # Set numerators
    k[np.array([0, 6, 0]), np.array([0, 6, 6])] += A*E
    k[np.array([1, 7, 5, 1, 7, 1, 1]), np.array([1, 7, 7, 5, 11, 7, 11])] += 3*E*Iz
    k[np.array([2, 8, 2, 8, 4, 2, 2]), np.array([2, 8, 4, 10, 8, 8, 10])] += 3*E*Iy
    k[np.array([3, 9, 3]), np.array([3, 9, 9])] += G*Jx
    k[np.array([4, 10, 4]), np.array([4, 10, 10])] += 2*E*Iy
    k[np.array([5, 11, 5]), np.array([5, 11, 11])] += 2*E*Iz
    # Set denominators
    k[np.array([0, 3, 6, 9, 0, 3, 4, 5]), np.array([0, 3, 6, 9, 6, 9, 10, 11])] *= 1/(2*a)
    k[np.array([2, 5, 8, 1, 4, 7, 2, 1]), np.array([4, 7, 10, 5, 8, 11, 10, 11])] *= 1/(2*a**2)
    k[np.array([1, 2, 7, 8, 1, 2]), np.array([1, 2, 7, 8, 7, 8])] *= 1/(2*a**3)
    k[np.array([4, 5, 10, 11]), np.array([4, 5, 10, 11])] *= 1/a
    # Set negative values
    k[np.array([2, 5, 7, 0, 1, 2, 3, 2]), np.array([4, 7, 11, 6, 7, 8, 9, 10])] *= -1
    k = np.tril(k.T) + np.triu(k, 1)
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