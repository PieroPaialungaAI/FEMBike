import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.linalg as la


def frameForm(propFile):
    baseDF = pd.read_csv(propFile, header=0, skiprows=[1])
    nPairs = len(baseDF)
    nodeDF = pd.DataFrame(columns=['X', 'Y', 'Z'], dtype=float)
    elemDF = pd.DataFrame(columns=['Node 1', 'Node 2', 'A', 'E', 'G', 'rho', 'a', 'Ix', 'Iy', 'Iz', 'J', 'xV', 'yV', 'zV'], dtype=float)
    elemDF = elemDF.astype({'Node 1': int, 'Node 2': int})
    eInd = 0
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
                if node == eNum - 1:
                    elemDF.loc[eInd] = [ind1, ind2] + conLst
                else:
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
    mass += np.tril(mass.T, -1)
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
    k += np.tril(k.T, -1)
    return k


def transM(locUV, globC, dim2=False):
    # Find local y and z directions
    if dim2:
        v2 = locUV+[1, 2, 0]
    else:
        v2 = locUV+[1, 2, 3]
    v1 = np.cross(locUV, v2)
    v2 = np.cross(locUV, v1)
    # Convert to unit vector
    v1 = v1 / np.sqrt(np.dot(v1, v1))
    v2 = v2 / np.sqrt(np.dot(v2, v2))
    locUV = locUV / np.sqrt(np.dot(locUV, locUV))
    # Define local and global coordinate systems
    locC = np.array([locUV, v2, v1])
    tVal = np.zeros((3, 3))
    for gC in range(3):
        for lC in range(3):
            tVal[lC, gC] = np.dot(locC[lC, :], globC[gC, :])
    if dim2:
        tVal[:, 2] = [0, 0, 1]
        tVal[2, :] = [0, 0, 1]
        tVal[1, :] = np.abs(tVal[1, :])
        tVal[:, 1] = np.abs(tVal[:, 1])
    print(tVal)
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


def glob(nodeDF, elemDF, globC, dim2=False):
    nNum = len(nodeDF)
    massG = np.zeros((nNum*6, nNum*6), dtype=float)
    kG = np.zeros((nNum*6, nNum*6), dtype=float)
    for index, row in elemDF.iterrows():
        # Calculate mass and stiffness matrices
        tempM = massMat(row['rho'], row['A'], row['a'], row['J'])
        tempK = kMat(row['A'], row['E'], row['G'], row['a'], row['Iy'], row['Iz'], row['J'])
        # Coordinate transformation
        tMat = transM(row[['xV', 'yV', 'zV']].to_numpy(), globC, dim2)
        tMatT = np.transpose(tMat)
        tempM = np.matmul(np.matmul(tMatT, tempM), tMat)
        tempK = np.matmul(np.matmul(tMatT, tempK), tMat)
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
    if dim2:
        delInd = np.tile([2, 3, 4], nNum) + 6 * np.repeat(np.arange(nNum), 3)
        kG_R = np.delete(kG, delInd, axis=0)
        kG_R = np.delete(kG_R, delInd, axis=1)
        massG_R = np.delete(massG, delInd, axis=0)
        massG_R = np.delete(massG_R, delInd, axis=1)
        return massG_R, kG_R

    else:
        return massG, kG


def appBound(boundFile, nodeDF, dim2 = False):
    boundDF = pd.read_csv(boundFile, header=0, skiprows=[1])
    indLst = []
    for index, row in boundDF.iterrows():
        # Locate closest point
        ind, _ = findClose(row[['X', 'Y', 'Z']].to_numpy(), nodeDF)
        if dim2:
            # Define fixed degrees of freedom
            lInd = np.where(row[['DOF 1', 'DOF 2', 'DOF 6']].to_numpy())[0].tolist()
            if len(lInd) > 0:
                indLst.extend(3*ind+lInd)
        else:
            # Define fixed degrees of freedom
            lInd = np.where(row[['DOF 1', 'DOF 2', 'DOF 3', 'DOF 4', 'DOF 5', 'DOF 6']].to_numpy())[0].tolist()
            if len(lInd) > 0:
                indLst.extend(6 * ind + lInd)
    indLst.sort()
    return indLst


def lamDef(mag, scale):
    return lambda t: mag*scale*np.sin(t)


def forceVect(forceFile, nodeDF, dim2 = False):
    forceDF = pd.read_csv(forceFile, header=0, skiprows=[1])
    dynamicFunc = {}
    if dim2:
        staticFV = np.zeros((3 * len(nodeDF),), dtype=float)
        for index, row in forceDF.iterrows():
            # Calculated unit direction vector
            temp = row[['xV', 'yV', 'zV']].to_numpy()
            dV = np.sqrt(np.dot(temp, temp))
            temp = temp / dV
            # Locate closest point
            ind, _ = findClose(row[['X', 'Y', 'Z']].to_numpy(), nodeDF)
            if row['freq'] > 0:
                for dirI in range(2):
                    dynamicFunc[3 * ind + dirI] = lamDef(row['mag'], temp[dirI])
            else:
                staticFV[3 * ind + [0, 1]] = row['mag'] * temp[:2]
    else:
        staticFV = np.zeros((6 * len(nodeDF),), dtype=float)
        for index, row in forceDF.iterrows():
            # Calculated unit direction vector
            temp = row[['xV', 'yV', 'zV']].to_numpy()
            dV = np.sqrt(np.dot(temp, temp))
            temp = temp/dV
            # Locate closest point
            ind, _ = findClose(row[['X', 'Y', 'Z']].to_numpy(), nodeDF)
            if row['freq'] > 0:
                for dirI in range(3):
                    dynamicFunc[6 * ind + dirI] = lamDef(row['mag'], temp[dirI])
            else:
                staticFV[6 * ind + [0, 1, 2]] = row['mag'] * temp

    return staticFV, dynamicFunc


def staticSolve(kM, forceV, boundInd):
    # Apply boundary conditions
    kM_R = np.delete(kM, boundInd, axis=0)
    kM_R = np.delete(kM_R, boundInd, axis=1)
    # print(kM_R)
    print(np.shape(kM_R))
    fV_R = np.delete(forceV, boundInd)
    print(np.shape(fV_R))
    # Solve for nodal displacement
    nDis_R = la.solve(kM_R, fV_R, assume_a='sym')
    nDis = np.zeros(np.shape(forceV))
    nDis[boundInd] = np.zeros(len(boundInd))
    mapInd = np.delete(np.arange(len(forceV)), boundInd)
    nDis[mapInd] = nDis_R
    nFor = np.matmul(kM, nDis)
    return nDis, nFor


# Verification (HW 5)
pFolder = r"C:\Users\Natalie\OneDrive - University of Cincinnati\Documents\UCFiles\Spring2022\AEEM7052\Final Project"
frameFile = pFolder + r"\FramePropertiesVer.csv"
boundFile = pFolder + r"\boundaryCondVer.csv"
forceFile = pFolder + r"\appForceVer.csv"

globC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

nodeDF, elemDF = frameForm(frameFile)
nodeDF.to_csv("nodesVer.csv")
elemDF.to_csv("elementsVer.csv")
mM, kM = glob(nodeDF, elemDF, globC, True)

boundInd = appBound(boundFile, nodeDF, True)

staticFV, dynamicFunc = forceVect(forceFile, nodeDF, True)

nDis, nFor = staticSolve(kM, staticFV, boundInd)
nd_df = pd.DataFrame(data=np.array_split(nDis, len(nodeDF)), columns=["DOF 1 Displacement", "DOF 2 Displacement", "DOF 6 Displacement"])
nf_df = pd.DataFrame(data=np.array_split(nFor, len(nodeDF)), columns=["DOF 1 Force", "DOF 2 Force", "DOF 6 Force"])
pd.concat([nodeDF, nd_df, nf_df], axis=1).to_csv("Verification.csv")

# Bike Problem
pFolder = r"C:\Users\Natalie\OneDrive - University of Cincinnati\Documents\UCFiles\Spring2022\AEEM7052\Final Project"
frameFile = pFolder + r"\FramePropertiesBase.csv"
boundFile = pFolder + r"\boundaryCondBase.csv"
forceFile = [pFolder + r"\appForceBase_" + s + ".csv" for s in ["S1", "S2", "D1"]]

globC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

nodeDF, elemDF = frameForm(frameFile)
nodeDF.to_csv("nodes.csv")
elemDF.to_csv("elements.csv")
mM, kM = glob(nodeDF, elemDF, globC)

boundInd = appBound(boundFile, nodeDF)
print(boundInd)

solI = 1
for file in forceFile:
    staticFV, dynamicFunc = forceVect(file, nodeDF)
    if bool(dynamicFunc):
        print("Dynamic Case")
    else:
        print("Static Case")
        nDis, nFor = staticSolve(kM, staticFV, boundInd)
        nd_df = pd.DataFrame(data=np.array_split(nDis, len(nodeDF)), columns=["DOF 1 Displacement", "DOF 2 Displacement", "DOF 3 Displacement", "DOF 4 Displacement", "DOF 5 Displacement", "DOF 6 Displacement"])
        nf_df = pd.DataFrame(data=np.array_split(nFor, len(nodeDF)), columns=["DOF 1 Force", "DOF 2 Force", "DOF 3 Force", "DOF 4 Force", "DOF 5 Force", "DOF 6 Force"])
        strT = "staticRes"+str(solI)+".csv"
        pd.concat([nodeDF, nd_df, nf_df], axis=1).to_csv(strT)
        solI += 1
        print(nDis[:6])
        print(nFor[:6])

