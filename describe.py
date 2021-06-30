import numpy as np
import sys
import libs.lib_stat as lib_stat
from libs.dataset_reader import *

header, data, raw_header, raw = open_dataset(sys.argv[1])
width = 15

# Headers
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, ""), end="")
    else:
        print("% *.*s | " % (width, width, header[i]), end="")
print("")

# Count
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "Count"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getCount(tmp[~np.isnan(tmp)])), end="")
print("")

# Mean
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "Mean"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getMean(tmp[~np.isnan(tmp)])), end="")
print("")

# Variance
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "Variance"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getVariance(tmp[~np.isnan(tmp)])), end="")
print("")

# Std
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "Std"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getStd(tmp[~np.isnan(tmp)])), end="")
print("")

# Min
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "Min"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getMin(tmp[~np.isnan(tmp)])), end="")
print("")

# 25%
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "25%"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getQ1(tmp[~np.isnan(tmp)])), end="")
print("")

# 50%
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "50%"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getMedian(tmp[~np.isnan(tmp)])), end="")
print("")

# 75%
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "75%"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getQ3(tmp[~np.isnan(tmp)])), end="")
print("")

# Max
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s | " % (width, "Max"), end="")
    else:
        tmp = data[:,i]
        print("%*f | " % (width, lib_stat.getMax(tmp[~np.isnan(tmp)])), end="")
print("")