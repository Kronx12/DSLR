import numpy as np
import matplotlib.pyplot as plt
import math
from libs.dataset_reader import *
from libs.lib_stat import *

header, data, raw_header, raw = open_dataset("datasets/dataset_train.csv")

cm = plt.cm.get_cmap('RdYlBu_r')

fig, axs = plt.subplots(2, math.ceil(len(data[0]) / 2))
fig.tight_layout()
for i in range(len(data[0])):
    n, bins, patches = axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].hist(data[:,i])
    axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].set_title(header[i])
    axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].set_xlabel('Value')
    axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].set_ylabel('Frequency')

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

axs[-1, -1].axis('off')
fig.set_size_inches(20, 8)
plt.savefig("exports/export_history.png", dpi=100)
plt.show()