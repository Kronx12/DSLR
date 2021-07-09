from os import removedirs

from numpy import NaN
from numpy.lib.function_base import append
from libs.lib_neuron import neuron
from libs.dataset_reader import *
import matplotlib.pyplot as plt

# Extract data
header, data, raw_header, raw_data = open_dataset("./datasets/dataset_train.csv")

house_set = raw_data[:, 1]

astronomy_set = data[:, 1]
herbology_set = data[:, 2]

ravenclaw_set = []
slytherin_set = []
gryffindor_set = []
hufflepuff_set = []

# Format data
mask = []
mask_a = ~np.isnan(herbology_set)
mask_b = ~np.isnan(astronomy_set)
for i in range(len(mask_a)):
    if (not mask_a[i] or not mask_b[i]):
        mask.append(False)
    else:
        mask.append(True)

herbology_set = herbology_set[np.where(mask)]
astronomy_set = astronomy_set[np.where(mask)]
house_set = house_set[np.where(mask)]

# Fill result array
for i in house_set:
    if i == "Ravenclaw":
        ravenclaw_set.append(1)
        slytherin_set.append(0)
        gryffindor_set.append(0)
        hufflepuff_set.append(0)
    elif i == "Slytherin":
        ravenclaw_set.append(0)
        slytherin_set.append(1)
        gryffindor_set.append(0)
        hufflepuff_set.append(0)
    elif i == "Gryffindor":
        ravenclaw_set.append(0)
        slytherin_set.append(0)
        gryffindor_set.append(1)
        hufflepuff_set.append(0)
    elif i == "Hufflepuff":
        ravenclaw_set.append(0)
        slytherin_set.append(0)
        gryffindor_set.append(0)
        hufflepuff_set.append(1)

data = []
for i in range(len(astronomy_set)):
    data.append([astronomy_set[i], herbology_set[i]])

data = np.array(data)

# Create neurons
neurons = []
neurons.append(neuron("ravenclaw", data, ravenclaw_set))
neurons.append(neuron("slytherin", data, slytherin_set))
neurons.append(neuron("gryffindor", data, gryffindor_set))
neurons.append(neuron("hufflepuff", data, hufflepuff_set))

for i in neurons:
    i.plot_prediction()
    i.plot_data()
    # i.gradient_descent(200000, 0.01, progression=True)
    # i.plot_history()
    i.debug()