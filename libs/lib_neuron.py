import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class neuron:
    name = "default_name"

    theta = []
    X = []
    Y = []

    m, n = 0, 0

    history = []

    def __init__(self, name, x_data, y_data):
        self.name = name
        self.X = np.array(x_data)
        self.X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))
        self.Y = np.array(y_data)
        self.Y = self.Y.reshape((self.Y.shape[0], 1))
        self.n, self.m = self.X.shape
        try:
            self.restoreBias()
        except:
            self.theta = np.zeros((self.m, 1))

    def plot_data(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        xlist = self.X[:, 0]
        ylist = self.X[:, 1]
        zlist = self.Y
        colors = []
        for i in self.Y:
            if i == 1:
                colors.append("red")
            else:
                colors.append("blue")
        xlist_surface = self.X[:,0]
        ylist_surface  = self.X[:,1]
        xlist_surface, ylist_surface = np.meshgrid(xlist_surface, ylist_surface)
        zlist_surface = xlist_surface * self.theta[0] + ylist_surface  * self.theta[1] + self.theta[2]
        ax.scatter(xlist, ylist, zlist, c=colors)
        ax.plot_surface(xlist_surface, ylist_surface, zlist_surface, cmap="winter")
        ax.set_xlabel("X0")
        ax.set_ylabel("X1")
        ax.set_zlabel("Model Values")
        ax.set_title(self.name)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    def plot_history(self):
        xlist = np.array(list(range(len(self.history))))
        plt.plot(self.history)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        
    def plot_prediction(self):
        sigmoid_v = np.vectorize(self.sigmoid)
        tmp = sigmoid_v(self.model())
        xlist = list(range(len(self.Y)))
        ylist = np.sort(tmp)
        colors = []
        for i in self.Y:
            if i == 1:
                colors.append("red")
            else:
                colors.append("blue")
        plt.scatter(xlist, ylist, c=colors, s=1)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        
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
        
    def restoreBias(self):
        self.theta = np.loadtxt("export_" + self.name + ".csv", delimiter=",")
        self.theta = np.reshape(self.theta, (self.m, 1))
    
    def saveBias(self):
        np.savetxt("export_" + self.name + ".csv", self.theta, delimiter=",")

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
            if i % 10 == 0 and (progression):
                print("Cost: ", cost, "%", i, "/", n, " " * 10, end="\r")
        self.saveBias()
        return (self.theta)

if __name__=="__main__":
    x = [[2, 3], [1, 2], [4, 5], [0, 1]]
    y = [0, 1, 0, 1]
    n = neuron("defaut", x, y)
    n.gradient_descent(1000, 0.1, progression=True)
    n.debug()
    
    n.plot_data()
    n.plot_history()
    n.plot_data()
    n.plot_prediction()