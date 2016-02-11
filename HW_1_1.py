import numpy as np
import matplotlib.pyplot as plot

#myarray is to load data from the given dataset
from numpy.ma import shape

X,Y = np.loadtxt('svar-set1.dat', unpack=True, usecols=[0,1])
plot.plot(X,Y,"o")
plot.xlabel("X")
plot.ylabel("Y")
plot.title("Linear regression")
plot.show()




