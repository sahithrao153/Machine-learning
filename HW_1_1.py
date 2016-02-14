import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#X,Y is to load data from the given dataset

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

