import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Major Nodes in Bike Frame
mNodes = np.array([[0, 0, 0], [-3, 1, 1], [-3, 1, -1], [0, 4, 0], [6, 3, 0], [6, 4, 0]], dtype=float)
# Major Node Pairs defining the Bike Frame
mPairs = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [3, 4], [4, 6], [5, 6]], dtype=int)
# Number of frame piece in the Bike Frame
nPairs = np.shape(mPairs)[0]
# Number of Elements in each piece of the Bike Frame
nElem = np.array([20, 20, 20, 20, 20, 20, 20, 20])

unitV = np.zeros((nPairs, 3), dtype=float)
distV = np.zeros((nPairs, 1), dtype=float)
nodeDict = {}

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# ax.scatter(mNodes[:, 0], mNodes[:, 1], mNodes[:, 2])

for ind in range(nPairs):
    temp1 = mNodes[mPairs[ind, 0] - 1, :]
    temp2 = mNodes[mPairs[ind, 1] - 1, :]
    unitV = temp2 - temp1
    distV = np.sqrt(np.sum(np.square(unitV)))
    unitV = unitV/distV
    eNum = nElem[ind]
    dD = distV/eNum
    tNodes = np.zeros((eNum+1, 3), dtype=float)
    tNodes[0, :] = temp1
    tNodes[-1, :] = temp2
    for node in range(1, eNum):
        tNodes[node, :] = tNodes[node-1, :]+dD*unitV
    ax.scatter(tNodes[:, 0], tNodes[:, 1], tNodes[:, 2])
    nodeDict[ind] = tNodes

plt.show()
