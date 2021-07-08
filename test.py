import numpy as np
import matplotlib.pyplot as plt

def debugMatrix(name, x):
    print(name, ":\n", x)
    print(name, "shape:", x.shape)
    print("___")

def sigmoid(z):
    if -z > np.log(np.finfo(type(z)).max):
        return 0.0    
    a = np.exp(-z)
    return 1.0/ (1.0 + a)

def model(x, theta):
    return (x.dot(theta))

X = np.random.randint(50, size=(100, 2))
X = np.concatenate((X, np.random.randint(60, 100, size=(100, 2))))
# X = np.hstack((X, np.ones((X.shape[0], 1))))
n, m = X.shape
print(X.shape)

Y = np.concatenate((np.ones((100, 1)), np.zeros((100, 1))))

debugMatrix("X", X)
debugMatrix("Y", Y)

# Y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
# Y = Y.reshape((Y.shape[0], 1))

theta = np.zeros((m, 1))

stats = []
learnrate = 0.1
sigmoid_v = np.vectorize(sigmoid)
for i in range(10000):
    tmp = X.T.dot(sigmoid_v(model(X, theta)) - Y)
    theta = theta - (learnrate / m) * tmp
    if i % 100 == 0:
        stats.append(sum(tmp)[0])

# for i in range(len(stats)):
#     print(stats[i][0], stats[i][1], stats[i][2])
stats = np.abs(stats)
colors = []
for i in Y:
    if i == 1:
        colors.append("red")
    else:
        colors.append("blue")
xlist = list(range(int(min(X[:,0])), int(max(X[:,0])) + 1))

# plt.plot(stats)
plt.plot(xlist, theta[0] * xlist + theta[1] * xlist + theta[2])
plt.scatter(X[:,0], X[:,1], c=colors)

print(theta)
print(stats)

plt.show()

