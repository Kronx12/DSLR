import numpy as np
import matplotlib.pyplot as plt
import math
from libs.dataset_reader import *
from libs.lib_stat import *

header, data, raw_header, raw = open_dataset("datasets/dataset_train.csv")

fig, axs = plt.subplots(2, math.ceil(len(data[0]) / 2))
fig.tight_layout()
for i in range(len(data[0])):
    x = list(range(len(data[:,i])))
    axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].scatter(x, data[:,i], s=.5)
    axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].set_title(header[i])
    axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].set_ylabel('Note')
    axs[math.floor(i / math.ceil(len(data[0]) / 2))][i % math.ceil(len(data[0]) / 2)].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axs[-1, -1].axis('off')
fig.set_size_inches(20, 8)

plt.savefig("exports/export_scatter_plot.png", dpi=100)
plt.show()