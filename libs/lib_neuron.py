import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class neuron:
    theta = []
    X = []
    Y = []

    m, n = 0, 0

    history = []

    def __init__(self, x_data, y_data):
        self.X = np.array(x_data)
        self.X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))
        self.Y = np.array(y_data)
        self.Y = self.Y.reshape((self.Y.shape[0], 1))
        self.n, self.m = self.X.shape
        self.theta = np.zeros((self.m, 1))

    def plot_theta(self):
        # xlist = np.array(np.rint(self.X[:,0]))
        xlist = np.array(list(range(int(min(self.X[:,0])), int(max(self.X[:,0])) + 1)))
        ylist = xlist * self.theta[0] + xlist * self.theta[1] + self.theta[2]
        
        plt.plot(xlist, ylist)

        # zlist = []

        # for i in range(len(ylist)):
            # zlist.append(xlist * self.theta[0] + ylist[i] * self.theta[1] + self.theta[2])

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.plot_surface(self.X[:,0], self.X[:,1], self.model(), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    def plot_data(self):
        xlist = self.X[:, 0]
        ylist = self.X[:, 1]

        print(ylist)
        colors = []
        for i in self.Y:
            if i == 1:
                colors.append("red")
            else:
                colors.append("blue")
        plt.scatter(xlist, ylist, c=colors, s=2)

    def plot_history(self):
        xlist = np.array(list(range(len(self.history))))
        plt.plot(self.history)

    def plot_prediction(self):
        sigmoid_v = np.vectorize(self.sigmoid)
        tmp = sigmoid_v(self.model())
        xlist = list(range(len(self.Y)))
        ylist = np.sort(tmp)

        print(ylist)
        colors = []
        for i in self.Y:
            if i == 1:
                colors.append("red")
            else:
                colors.append("blue")
        plt.scatter(xlist, ylist, c=colors, s=1)

    def plot_show(self):
        plt.show()

    def debugMatrix(self, name, x):
        print(name, ":\n", x)
        print(name, "shape:", x.shape)
        print("___")

    def debug(self):
        self.debugMatrix("X", self.X)
        self.debugMatrix("Y", self.Y)
        self.debugMatrix("theta", self.theta)

    def getAccuracy(self):
        sigmoid_v = np.vectorize(self.sigmoid)
        tmp = np.rint(sigmoid_v(self.model()))
        ok = 0
        for i in range(len(self.Y)):
            if tmp[i] == self.Y[i]:
                ok += 1
        return (ok * 100 / len(self.Y))

    def insertBias(self, theta):
        self.theta = theta.reshape((self.m, 1))

    def sigmoid(self, z):
        if -z > np.log(np.finfo(type(z)).max):
            return 0.0    
        a = np.exp(-z)
        return 1.0 / (1.0 + a)

    def model(self):
        return (self.X.dot(self.theta))

    def predict(self):
        sigmoid_v = np.vectorize(self.sigmoid)
        return (sigmoid_v(self.model()))

    def cost(self):
        epsilon = 1e-5
        predictions = self.predict()
        class1_cost = -self.Y * np.log(predictions + epsilon)
        class2_cost = (1 - self.Y) * np.log(1 - predictions + epsilon)
        cost = class1_cost - class2_cost
        cost = cost.sum() / len(self.Y)
        return (cost)
        # sigmoid_v = np.vectorize(self.sigmoid)
        # return (self.X.T.dot(np.rint(sigmoid_v(self.model())) - self.Y))

    def update_weights(self, lr):
        N = len(self.theta)
        predictions = self.predict()
        gradient = np.dot(self.X.T,  predictions - self.Y)
        gradient /= N
        gradient *= lr
        self.theta -= gradient
        return (self.theta)

    def gradient_descent(self, n, learnrate, progression=False):
        for i in range(n):
            self.update_weights(learnrate)
            cost = self.cost()
            self.history.append(cost)
            # if i % 100 == 0:
                # print("iter: "+str(i) + " cost: "+str(cost))
            # self.history.append(self.getAccuracy())
            # self.theta = self.theta - (learnrate / self.n) * self.cost()
            if i % 10 == 0 and (progression):
                print("Accuracy: ", cost, "%", i, "/", n, " " * 10, end="\r")
        return (self.theta)

if __name__=="__main__":
    x = [[2, 3], [1, 2], [4, 5], [0, 1]]
    y = [0, 1, 0, 1]
    n = neuron(x, y)
    n.debug()
    n.gradient_descent(100000, 0.1, progression=True)

    n.plot_theta()
    n.plot_data()
    n.plot_show()

    n.plot_history()
    n.plot_show()