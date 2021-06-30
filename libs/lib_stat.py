import math
import numpy as np

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
