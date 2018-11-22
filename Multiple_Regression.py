import numpy as np


class MultipleRegression:

    def __init__(self, x, y):
        self.xtrain = x
        self.ytrain = y
        self.theta = np.zeros((1, self.xtrain.shape[1]+1))

    def multiple_reg(self):
        ones = np.ones([self.xtrain.shape[0],1])
        self.xtrain = np.concatenate((ones, self.xtrain), axis=1)
        alpha = 0.01
        iters = 1000
        self.theta, cost = self.__gradient_descent(self.theta, iters, alpha)
        return cost

    def __compute_cost(self, theta):
        tobesummed = np.power(((self.xtrain @ theta.T) - self.ytrain), 2)
        return np.sum(tobesummed)/(2*len(self.xtrain))

    def __gradient_descent(self, theta, iters, alpha):
        cost = np.zeros(iters)
        for i in range(iters):
            theta = theta - (alpha/len(self.xtrain)) * np.sum(np.dot(self.xtrain.T, (self.xtrain @ theta.T - self.ytrain)), axis=0)
            cost[i] = self.__compute_cost(theta)
        return theta, cost

    def predict_value(self, x):
        x = np.asmatrix(np.insert(x, 0, 1))
        return self.theta*x.T
