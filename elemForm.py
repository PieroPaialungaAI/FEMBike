import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Major Nodes in Bike Frame
mNodes = np.array([[0, 0, 0], [-3, 1, 1], [-3, 1, -1], [0, 4, 0], [6, 3, 0], [6, 4, 0]], dtype=float)
# Major Node Pairs defining the Bike Frame
mPairs = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [3, 4], [4, 6], [5, 6]], dtype=int)
# Number of Elements in each piece of the Bike Frame
nElem = np.array([20, 20, 20, 20, 20, 20, 20, 20], dtype=int)


def frameForm(mNodes, mPairs, nElem):
    nPairs = np.shape(mPairs)[0]
    unitV = np.zeros((nPairs, 3), dtype=float)
    distV = np.zeros((nPairs, 1), dtype=float)
    nodeDF = pd.DataFrame(data=np.concatenate((mNodes, np.zeros((np.sum(nElem)-len(nElem), 3)))), columns=['X', 'Y', 'Z'])
    elemDF = pd.DataFrame(data=np.zeros((np.sum(nElem), 11)), columns=['Node 1', 'Node 2', 'A', 'E', 'G', 'rho', 'a', 'Ix', 'Iy', 'Iz', 'Jx'])
    nInd = len(mNodes)
    eInd = 0
    for ind in range(nPairs):
        ind1 = mPairs[ind, 0] - 1
        ind2 = mPairs[ind, 1] - 1
        temp1 = mNodes[ind1, :]
        temp2 = mNodes[ind2, :]
        unitV = temp2 - temp1
        distV = np.sqrt(np.sum(np.square(unitV)))
        unitV = unitV/distV
        eNum = nElem[ind]
        dD = distV/eNum
        for node in range(eNum):
            if node == 0:
                elemDF.loc[eInd] = [ind1, nInd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                eInd += 1
            else:
                if node == eNum-1:
                    elemDF.loc[eInd] = [nInd, ind2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    eInd += 1
                else:
                    elemDF.loc[eInd] = [nInd, nInd+1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    eInd += 1
                nodeDF.loc[nInd] = temp1+dD*unitV*node
                nInd += 1
    return nodeDF, elemDF


df1, df2 = frameForm(mNodes, mPairs, nElem)
print(df1)
print(df2)



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax.scatter(mNodes[:, 0], mNodes[:, 1], mNodes[:, 2])
# ax.scatter(tNodes[:, 0], tNodes[:, 1], tNodes[:, 2])
# plt.show()
