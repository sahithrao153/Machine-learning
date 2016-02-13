import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#X,Y is to load data from the given dataset
lm = LinearRegression()
X,Y = np.loadtxt('svar-set1.dat', unpack=True, usecols=[0,1])
"""lm.fit(X[:,np.newaxis],Y)

plt.scatter(X,Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear regression")
plt.plot(X, lm.predict(X[:,np.newaxis]), color='brown')
plt.show()

"""
coef = np.polyfit(X,Y,1)

poly_func = np.poly1d(coef,variable="x")

print poly_func
#plt.scatter(X,Y)

#plt.show()

