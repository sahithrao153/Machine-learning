'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
X,Y = np.loadtxt('svar-set2.dat', unpack=True, usecols=[0,1])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
coef = np.polyfit(X_test,Y_test,4)
poly_func = np.poly1d(coef,variable="x")
ys = poly_func(X_test)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X_test,Y_test)
plt.plot(X_test,ys,color='red')
plt.show()
'''
from numpy import loadtxt, zeros, ones
from pylab import plot,title, xlabel, ylabel


#Evaluate the linear regression
def cost_function(X, y, theta):
    m = y.size
    predictions = X.dot(theta).flatten()
    square_Errors = (predictions - y) ** 2
    J = (1.0 / (2 * m)) * square_Errors.sum()
    return J


data = loadtxt('svar-set1.dat')


plot(data[:, 0], data[:, 1], marker='o', c='b')
title('Single variable')
xlabel('X')
ylabel('Y')

X = data[:, 0]
y = data[:, 1]

#training data size
m = y.size

#Adding ones to X (interception data)
it = ones(shape=(m, 2))
it[:, 1] = X
#Initialize theta parameters
theta = zeros(shape=(2, 1))

#compute and display initial cost
print cost_function(it, y, theta)
