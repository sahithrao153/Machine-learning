def cost_function(X, y, theta):
    m = float(X.shape[0])
    cost = (1./(2.*m))*(X*theta-y).T*(X*theta-y)
    return cost.flat[0]

def gradient(X, y, theta, iter, alpha):
    theta_iter = [] #record theta for each iteration
    cost_iter = []  #record cost for each iteration
    m = float(X.shape[0])

    for i in range(iter):
        #update theta
        theta = theta-(alpha/m)*X.T*(X*theta-y)
        theta_iter.append(theta)
        cost_iter.append(cost_function(X,y,theta))

    return(theta, theta_iter, cost_iter)

import matplotlib.pyplot as plt
import numpy as np

#load data
data = np.loadtxt('svar-set2.dat')

#set initial variables
x = np.ones(data.shape)
x[:,1] = data[:,0]
y = np.zeros(shape=(data.shape[0],1))
y[:,0] = data[:,1]
theta = np.matrix([[0.],[0.]])
alpha = 0.03
iter = 100000

#gradient descent
theta, theta_iter, cost_iter = gradient(x, y, theta, iter, alpha)

#plot result
result = x*theta
plt.plot(data[:,0], result)
plt.scatter(data[:,0], data[:,1], marker='o', c='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()




