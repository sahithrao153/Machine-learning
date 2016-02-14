import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#X,Y is to load data from the given dataset
lm = LinearRegression()
X,Y = np.loadtxt('svar-set2.dat', unpack=True, usecols=[0,1])


coef = np.polyfit(X,Y,1)
poly_func = np.poly1d(coef,variable="x")
ys = poly_func(X)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,Y)
plt.plot(X,ys,'->',color='red')
plt.show()



