import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PolynomialRegression:
    c = 1
    w = []

    def __init__(self, x, y, ridge, max_degree):
        self.ridge = ridge
        self.xtrain = np.copy(x)
        self.ytrain = np.copy(y)
        self.dist = np.ones((2, self.xtrain.shape[1]+1))
        self.max_degree = max_degree

    def normalize(self, x):
        for i in range(x.shape[1]):
            self.dist[0, i] = np.mean(x[:, i])
            self.dist[1, i] = np.std(x[:, i]) + 0.00001
            x[:, i] = (x[:, i] - self.dist[0, i])/self.dist[1, i]
        return x

    def weights(self, x, y, complexity):
        phi_matrix = self.features(x, complexity)
        phi = np.dot(np.transpose(phi_matrix), phi_matrix)
        mat = phi + self.ridge * np.eye(x.shape[1]*complexity+1)
        w = np.dot(np.dot(np.linalg.inv(mat), np.transpose(phi_matrix)), y)
        return w

    def features(self, x, complexity):
        x = np.asmatrix(x)
        phi = np.zeros((x.shape[0], x.shape[1]*complexity+1))
        for c in range(complexity+1):
            for input in range(x.shape[1]):
                for i in range(x.shape[0]):
                    phi[i, input+c] = np.power(np.sum(x[i, input]), c)
        return phi

    def rmse(self, w, x, y, c):
        y = np.asmatrix(y)
        f = np.dot(self.features(x, c), w)
        sqr = np.zeros((f.size, 1))
        for i in range(f.size):
            sqr[i, 0] = pow((f[i] - y[i, 0]), 2)
        rmse = np.sqrt((np.sum(sqr) + self.ridge*np.dot(np.transpose(w), w)) / x.size)
        return rmse

    def eval_complexity(self):
        rmse_val = np.zeros(self.max_degree)
        for i in range(1, self.max_degree+1):
            w = self.weights(self.xtrain, self.ytrain, i)
            rmse_val[i-1] = self.rmse(w, self.xtrain, self.ytrain, i)
        return np.argmin(rmse_val)+1

    def train(self):
        self.xtrain = self.normalize(self.xtrain)
        self.c = self.eval_complexity()
        print("Optimal Complexity = ", self.c)
        self.w = self.weights(self.xtrain, self.ytrain, self.c)
        return

    def predict_value(self, x):
        x = np.asmatrix(np.copy(x))
        for i in range(x.shape[1]):
           x[:, i] = (x[:, i] - self.dist[0, i])/self.dist[1, i]
        phi = self.features(x, self.c)
        return np.dot(phi, self.w)


'''
x = np.random.sample((50, 4))
x = np.sort(x, axis=0)
y_wahr = np.sum(np.multiply(x, x), axis=1)
y = np.sum(np.multiply(x, x), axis=1) + np.random.sample(50)*0.001

est = PolynomialRegression(x, y, pow(10, -5))
est.train()
'''
