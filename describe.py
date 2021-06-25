import numpy as np
import sys, math

def getCount(data):
    return (len(data))

def getMean(data):
    return (float(data.sum()) / len(data))

def getSS(data):
    return (((data - getMean(data)) ** 2).sum())

def getVariance(data):
    return (getSS(data) / (len(data) - 1))

def getStd(data):
    return (math.sqrt(getVariance(data)))

def getMin(data):
    s = False
    tmp = 0
    for i in data:
        if not s or i < tmp:
            tmp = i
            s = True
    return (tmp)

def getMax(data):
    s = False
    tmp = 0
    for i in data:
        if not s or i > tmp:
            tmp = i
            s = True
    return (tmp)

def getQ1(data):
    return (np.sort(data)[int((len(data) + (4 - len(data) % 4)) / 4) - 1])

def getMedian(data):
    data = np.sort(data)
    if (len(data) % 2 == 1):
        return (data[math.floor(len(data) / 2)])
    else:
        return ((data[int(len(data) / 2 - 1)] + data[int(len(data) / 2)]) / 2)

def getQ3(data):
    return (np.sort(data)[int((len(data) * 3) / 4)])

data = np.genfromtxt(sys.argv[1], delimiter=",")[1:]
data = data[:,~np.isnan(data).all(axis=0)]

width = 18

# Headers
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, ""), end="")
    else:
        print("%*s" % (width, "Feature" + str(i)), end="")
print("")

# Count
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "Count"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getCount(tmp[~np.isnan(tmp)])), end="")
print("")

# Mean
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "Mean"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getMean(tmp[~np.isnan(tmp)])), end="")
print("")

# Variance
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "Variance"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getVariance(tmp[~np.isnan(tmp)])), end="")
print("")

# Std
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "Std"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getStd(tmp[~np.isnan(tmp)])), end="")
print("")

# Min
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "Min"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getMin(tmp[~np.isnan(tmp)])), end="")
print("")

# 25%
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "25%"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getQ1(tmp[~np.isnan(tmp)])), end="")
print("")

# 50%
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "50%"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getMedian(tmp[~np.isnan(tmp)])), end="")
print("")

# 75%
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "75%"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getQ3(tmp[~np.isnan(tmp)])), end="")
print("")

# Max
for i in range(len(data[0])):
    if (i == 0):
        print("%-*s" % (width, "Max"), end="")
    else:
        tmp = data[:,i]
        print("%*f" % (width, getMax(tmp[~np.isnan(tmp)])), end="")
print("")