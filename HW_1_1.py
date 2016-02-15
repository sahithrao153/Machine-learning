import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

files_list = ['svar-set1.dat','svar-set2.dat','svar-set3.dat','svar-set4.dat']
print 'files list is :\n',files_list
f_index = int(raw_input('enter the filename index from the list from 1 to 4 '))
degree_poly = int(raw_input('enter the degree of the polynomial from 1 to n '))
X,Y = np.loadtxt(files_list[f_index-1], unpack=True, usecols=[0,1])


def split_test_data(x,y):
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.25, random_state=42)
    coef = np.polyfit(X_test,Y_test,degree_poly)
    poly_func = np.poly1d(coef,variable="x")
    ys = poly_func(X_test)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(X_test,Y_test)
    plt.plot(X_test,ys,color='green')
    plt.show()

def cross_valid_k():
    print "k"

def find_MSE():
    print "MSE"

split_test_data(X,Y)

cross_valid_k()

find_MSE()



