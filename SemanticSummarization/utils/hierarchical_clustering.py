import timeit

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

data = np.genfromtxt("data/ExpRawData-E-TABM-84-A-AFFY-44.tab", names=True, usecols=tuple(range(1, 30)), dtype=float,
                     delimiter="\t")

print(len(data))
print(len(data.dtype.names))

data_array = data.view((np.float, len(data.dtype.names)))
data_array = data_array.transpose()

print(data_array)

data_dist = pdist(data_array)  # computing the distance
data_link = linkage(data_dist)  # computing the linkage

dendrogram(data_link, labels=data.dtype.names)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);

# Compute and plot first dendrogram.
fig = plt.figure(figsize=(8, 8))
# x ywidth height
ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.6])
Y = linkage(data_dist, method='single')
Z1 = dendrogram(Y, orientation='right', labels=data.dtype.names)  # adding/removing the axes
ax1.set_xticks([])

# Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
Z2 = dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

# Compute and plot the heatmap
axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = squareform(data_dist)
D = D[idx1, :]
D = D[:, idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
plt.colorbar(im, cax=axcolor)

#
# from fastcluster import *
# %timeit data_link = linkage(data_array, method='single', metric='euclidean', preserve_input=True)
# dendrogram(data_link,labels=data.dtype.names)
# plt.xlabel('Samples')
# plt.ylabel('Distance')
# plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);
# plt.show()
#
# from Bio.Cluster import *
# handle = open("../data/ExpRawData-E-TABM-84-A-AFFY-44.tab")
# record = read(handle)
#
# genetree = record.treecluster(method='s')
# genetree.scale()
# exptree = record.treecluster(dist='u', transpose=1)
# record.save("../results/biopython_clustering", genetree, exptree)
