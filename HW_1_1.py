import numpy as np
import matplotlib.pyplot as plot
#X,Y is to load data from the given dataset
X,Y = np.loadtxt('svar-set3.dat', unpack=True, usecols=[0,1])
plot.plot(X,Y,"o")
plot.xlabel("X")
plot.ylabel("Y")
plot.title("Linear regression")
plot.show()




