import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def frameForm(propFile):
    baseDF = pd.read_csv(propFile, header=0, skiprows=[1])
    nPairs = len(baseDF)
    nodeDF = pd.DataFrame(columns=['X', 'Y', 'Z'], dtype=float)
    elemDF = pd.DataFrame(columns=['Node 1', 'Node 2', 'A', 'E', 'G', 'rho', 'a', 'Ix', 'Iy', 'Iz', 'J', 'xV', 'yV', 'zV'], dtype=float)
    elemDF = elemDF.astype({'Node 1': int, 'Node 2': int})
    eInd = 0
    print(baseDF.columns)
    nInd = int(baseDF[['p1', 'p2']].to_numpy().max())
    for ind in range(nPairs):
        ind1 = int(baseDF['p1'].iloc[ind] - 1)
        ind2 = int(baseDF['p2'].iloc[ind] - 1)
        eNum = baseDF['numElem'].iloc[ind]
        temp1 = baseDF[['x1', 'y1', 'z1']].iloc[ind].to_numpy()
        temp2 = baseDF[['x2', 'y2', 'z2']].iloc[ind].to_numpy()
        unitV = temp2 - temp1
        distV = np.sqrt(np.dot(unitV, unitV))
        unitV = unitV / distV
        dD = distV / eNum
        conLst = baseDF[['A', 'E', 'G', 'rho', 'Ix', 'Iy', 'Iz', 'J']].iloc[ind].tolist()
        conLst.insert(4, dD/2)
        conLst += list(unitV)
        nodeDF.loc[ind1] = temp1
        nodeDF.loc[ind2] = temp2
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
    return nodeDF.sort_index(), elemDF.sort_index()

def massMat(rho, A, a, J):
    rV = J/A
    a2 = a**2
    mass = np.zeros((12, 12), dtype=float)
    mass[np.array([0, 6]), np.array([0, 6])] += 70
    mass[np.array([1, 2, 7, 8]), np.array([1, 2, 7, 8])] += 78
    mass[np.array([3, 9]), np.array([3, 9])] += 70*rV
    mass[np.array([4, 5, 10, 11]), np.array([4, 5, 10, 11])] += 8*a2
    mass[np.array([0]), np.array([6])] += 35
    mass[np.array([1, 8, 2, 7]), np.array([5, 10, 4, 11])] += 22*a
    mass[np.array([1, 2]), np.array([7, 8])] += 27
    mass[np.array([2, 5, 1, 4]), np.array([10, 7, 11, 8])] += 13*a
    mass[np.array([3]), np.array([9])] += -35*rV
    mass[np.array([4, 5]), np.array([10, 11])] += -6*a2
    mass[np.array([1, 4, 2, 7]), np.array([11, 8, 4, 11])] *= -1
    mass *= rho*A*a/105
    mass = np.tril(mass.T) + np.triu(mass, 1)
    return mass

def kMat(A, E, G, a, Iy, Iz, J):
    k = np.zeros((12, 12), dtype=float)
    # Set numerators
    k[np.array([0, 6, 0]), np.array([0, 6, 6])] += A*E
    k[np.array([1, 7, 5, 1, 7, 1, 1]), np.array([1, 7, 7, 5, 11, 7, 11])] += 3*E*Iz
    k[np.array([2, 8, 2, 8, 4, 2, 2]), np.array([2, 8, 4, 10, 8, 8, 10])] += 3*E*Iy
    k[np.array([3, 9, 3]), np.array([3, 9, 9])] += G*J
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

def transM(locUV):
    v2 = locUV + [0, 0, 1]
    v1 = np.cross(locUV, v2)
    v2 = np.cross(locUV, v1)
    v1 = v1/np.sqrt(np.dot(v1, v1))
    v2 = v2 / np.sqrt(np.dot(v2, v2))
    locC = np.array([locUV, v1, v2])
    globC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tVal = np.zeros((3, 3))
    for g in range(3):
        for l in range(3):
            tVal[l, g] = np.dot(locC[l, :], globC[g, :])
    tMat = np.zeros((12, 12))
    tMat[0:3, 0:3] = tVal
    tMat[3:6, 3:6] = tVal
    tMat[6:9, 6:9] = tVal
    tMat[9:12, 9:12] = tVal
    return tMat

def findClose(pt, nodeDF):
    dist = nodeDF.apply(lambda row: np.abs(np.linalg.norm(row[['X', 'Y', 'Z']].to_numpy()-pt)), axis=1)
    ind = dist.idxmin()
    return ind, nodeDF.iloc[ind]

def glob(nodeDF, elemDF):
    nNum = len(nodeDF)
    massG = np.zeros((nNum*6, nNum*6), dtype=float)
    kG = np.zeros((nNum*6, nNum*6), dtype=float)
    for index, row in elemDF.iterrows():
        # Calculate mass and stiffness matrices
        tempM = massMat(row['rho'], row['A'], row['a'], row['J'])
        tempK = kMat(row['A'], row['E'], row['G'], row['a'], row['Iy'], row['Iz'], row['J'])
        # Coordinate transformation
        tMat = transM(row[['xV', 'yV', 'zV']].to_numpy())
        tMatT = np.transpose(tMat)
        tempM = tMatT*tempM*tMat
        tempK = tMatT*tempK*tMat
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

def appBound(boundFile, nodeDF):
    boundDF = pd.read_csv(boundFile, header=0, skiprows=[1])
    indLst = []
    for index, row in boundDF.iterrows():
        # Locate closest point
        ind, _ = findClose(row[['X', 'Y', 'Z']].to_numpy(), nodeDF)
        # Define fixed degrees of freedom
        lInd = np.where(row[['DOF 1', 'DOF 2', 'DOF 3', 'DOF 4', 'DOF 5', 'DOF 6']].to_numpy())[0].tolist()
        if len(lInd) > 0:
            indLst.extend(6*ind+lInd)
    indLst.sort()
    return indLst

def forceVect(forceFile, nodeDF):
    forceDF = pd.read_csv(forceFile, header=0, skiprows=[1])
    forceV = np.zeros((6*len(nodeDF),), dtype=float)
    for index, row in forceDF.iterrows():
        # Calculated unit direction vector
        temp = row[['xV', 'yV', 'zV']].to_numpy()
        dV = np.sqrt(np.dot(temp, temp))
        temp = temp/dV
        # Locate closest point
        ind, _ = findClose(row[['X', 'Y', 'Z']].to_numpy(), nodeDF)
        forceV[6*ind+[0, 1, 2]] = row['mag']*temp
    return forceV

frameFile = r"C:\Users\Natalie\OneDrive - University of Cincinnati\Documents\UCFiles\Spring2022\AEEM7052\Final Project\FrameProperties.csv"
boundFile = r"C:\Users\Natalie\OneDrive - University of Cincinnati\Documents\UCFiles\Spring2022\AEEM7052\Final Project\boundaryCond.csv"
forceFile = r"C:\Users\Natalie\OneDrive - University of Cincinnati\Documents\UCFiles\Spring2022\AEEM7052\Final Project\appForce.csv"
nodeDF, elemDF = frameForm(frameFile)
print(nodeDF)
print(elemDF)

mM, kM = glob(nodeDF, elemDF)
print(np.shape(mM))
print(np.shape(kM))

indLst = appBound(boundFile, nodeDF)
print(indLst)

forceV = forceVect(forceFile, nodeDF)
print(forceV)

