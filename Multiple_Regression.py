import numpy as np
import matplotlib.pyplot as plt


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
        self.theta, cost = self.__gradientDescent(self.theta, iters, alpha)
        return cost

    def __computeCost(self, theta):
        tobesummed = np.power(((self.xtrain @ theta.T) - self.ytrain), 2)
        return np.sum(tobesummed)/(2*len(self.xtrain))

    def __gradientDescent(self, theta, iters, alpha):
        cost = np.zeros(iters)
        for i in range(iters):
            theta = theta - (alpha/len(self.xtrain)) * np.sum(np.dot(self.xtrain.T, (self.xtrain @ theta.T - self.ytrain)), axis=0)
            cost[i] = self.__computeCost(theta)
        return theta, cost

    def predictValue(self, x):
        x = np.asmatrix(np.insert(x, 0, 1))
        return self.theta*x.T


'''
x = np.matrix([[1, 2, 3, 4], [2, 3, 4, 5], [5, 3, 4, 6], [1, 3, 2, 4], [7, 5, 3, 1], [1, 1, 1, 1], [2, 2, 2, 2]])
y = np.transpose(np.asmatrix([11, 15, 19, 11, 15, 5, 9]))
estimator = MultipleRegression(x, y)
w = estimator.multiple_reg()

plt.plot(np.arange(10), w, 'r')
plt.show()

new_x = np.array([2, 5, 6, 7])
print("Should be: ", np.sum(new_x) + 1)
print(estimator.predictValue(new_x))
'''
