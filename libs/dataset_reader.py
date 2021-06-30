import numpy as np
import csv

def open_dataset(path):
    file = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            file.append(row)
    header = np.array(file)[0, 6:]
    raw_header = np.array(file)[0, :6]
    raw = np.array(file)[1:, :6]

    file = np.array(file)
    data = np.array(file[1:,6:])
    data = np.where(data == '', np.nan, data)
    data = data.astype(np.float64)

    header = header[~np.isnan(data).all(axis=0)]
    data = data[:,~np.isnan(data).all(axis=0)]
    return (header, data, raw_header, raw)
