import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#X,Y is to load data from the given dataset
lm = LinearRegression()
X,Y = np.loadtxt('svar-set2.dat', unpack=True, usecols=[0,1])
"""lm.fit(X[:,np.newaxis],Y)

plt.scatter(X,Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear regression")
plt.plot(X, lm.predict(X[:,np.newaxis]), color='brown')
plt.show()

"""
op3 = np.polyfit(X,Y,6)
print op3
yf = np.polyval(np.poly1d(op3), Y)
#print yf

print max(Y-yf)


