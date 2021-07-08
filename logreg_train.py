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
neurons.append(neuron(data, ravenclaw_set))
neurons.append(neuron(data, slytherin_set))
neurons.append(neuron(data, gryffindor_set))
neurons.append(neuron(data, hufflepuff_set))

# neurons[0].insertBias(np.array([-8392.34582622, 1050078.19329623, -6003389.881203]))
neurons[0].insertBias(np.array([ -89.57835927, 8644.23936994, -58484.33683975]))
neurons[0].gradient_descent(1000, 0.001, progression=True)
# neurons[0].gradient_descent(1000, 1, progression=True)
# neurons[0].gradient_descent(1000, 0.1, progression=True)
# neurons[0].gradient_descent(1000, 0.01, progression=True)
# neurons[0].gradient_descent(1000, 0.001, progression=True)
# neurons[0].gradient_descent(1000, 0.0001, progression=True)
neurons[0].debug()

neurons[0].plot_history()
neurons[0].plot_show()

neurons[0].plot_prediction()
neurons[0].plot_show()

# neurons[0].plot_theta()
neurons[0].plot_data()
neurons[0].plot_show()

# xlist = list(range(len(neurons[0].getMean())))
# plt.plot(xlist, np.array(neurons[0].getMean()))
# plt.show()

# xlist = list(range(int(min(astronomy_set)), int(max(astronomy_set))))

# plt.plot(xlist, neurons[0].theta[0] * xlist + neurons[0].theta[1] * xlist + neurons[0].theta[2])
# for i in neurons:
#     i.gradient_descent(1000, 0.1)
#     i.debug()
#     plt.plot(xlist, i.theta[1] * xlist + i.theta[0])

# colors = []
# for i in range(len(house_set)):
#     if (ravenclaw_set[i] == 1):
#         colors.append("red")
#     elif (slytherin_set[i] == 1):
#         colors.append("green")
#     elif (gryffindor_set[i] == 1):
#         colors.append("blue")
#     elif (hufflepuff_set[i] == 1):
#         colors.append("orange")
# plt.scatter(astronomy_set, herbology_set * 10000, s=2, c=colors)
# plt.show()