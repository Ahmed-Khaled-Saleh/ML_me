
from distutils.log import debug
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')

class LinearRegression():

    #hyperparameters
    theta = np.zeros((2, 1))
    iterations = 1000
    alpha = 0.01
    m = 0

    #Accumelators
    final_thetas = []
    J_vals = []

    def __init__(self):
        '''
        pass a list of hyperparameters as follows : 

        thetas : initial weights.
        iterations : number of iterations.
        alpha : learning rate.
        m = number of training examples.
        n = number of features.

        '''
        pass

    def costFunction(self, m, theta, X, y):
        self.m = m =  y.shape[0]
        self.theta = theta = np.zeros((X.shape[1], 1))
        #print(m ,theta, X, y)
        return np.sum(np.square(np.dot(X, theta) - y)) / (2*m)

    
    def GD(self, iterations, alpha,theta,  m, X, y, debug=False):

        for _ in range(iterations):
            theta = theta - (alpha/m)*np.dot(X.T,(np.dot(X, theta) - y))
            self.final_thetas.append(theta)
            self.J_vals.append(self.costFunction(self.m, self.theta, X, y))
        if debug: self.Debug(X, y)
        return self.final_thetas, self.J_vals


    def Debug(self,X, y):
        '''
            This method is used to plot #iterations vs J(theta)
            As described by Andrew NG.
        '''
        x_is = np.arange(self.iterations)
        y_is = self.J_vals
        x_is = x_is[:, np.newaxis]
        plt.plot(x_is, y_is)
        plt.xlabel('# iterations')
        plt.ylabel('J(0)')
        plt.show()

    def Fit(self, X, y, debug=False):
        self.theta = np.zeros((X.shape[1], 1))
        return self.GD(self.iterations, self.alpha, self.theta, y.shape[0], X, y, debug=True)
    
    def Predict(self, x):
        return np.dot(X, self.theta)
    
    def NormalEquation(self):
        pass

    def VisualizeTrainingProcess(self):
        pass


df = pd.read_csv('../../data/week_1-ex_1.txt')
X = df['Population']
y = df['Profit']

X = X[np.newaxis, :]
y = y[:, np.newaxis]
ones = np.ones((1, 97))
X = np.vstack((ones, X))
X = X.T

LR = LinearRegression()
LR.Fit(X, y, debug=True)
#LR.Fit(X, y)
print(X.shape, y.shape, LR.theta.shape)
#print(LR.costFunction(LR.m, LR.theta, X, y))

#LR.Debug()



