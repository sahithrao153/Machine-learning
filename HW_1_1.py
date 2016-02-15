import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

files_list = ['svar-set1.dat', 'svar-set2.dat', 'svar-set3.dat', 'svar-set4.dat']
print 'files list is :\n', files_list
f_index = int(raw_input('enter the filename index from the list from 1 to 4 '))
degree_poly = int(raw_input('enter the degree of the polynomial from 1 to n '))
test_amount = float(raw_input('enter the test data size from 0 to 1 '))
X, Y = np.loadtxt(files_list[f_index - 1], unpack=True, usecols=[0, 1])
plt.title('data')
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X, Y)
plt.show()

def split_train_test_data(x, y ,testsize):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_amount, random_state=20)
    # to plot train data
    coef_t = np.polyfit(X_test, Y_test, degree_poly)
    poly_func_t = np.poly1d(coef_t, variable="x")
    ys = poly_func_t(X_train)
    plt.title('train data')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(X_train, Y_train)
    plt.plot(X_train, ys, color='green')
    plt.show()
    # to plot test data
    coef = np.polyfit(X_test, Y_test, degree_poly)
    poly_func = np.poly1d(coef, variable="x")
    ys = poly_func(X_test)
    plt.title('test data')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(X_test, Y_test)
    plt.plot(X_test, ys, color='green')
    plt.show()


def cross_valid_k():
    print "k"


def find_mse():
    print "MSE"


split_train_test_data(X, Y, test_amount)

cross_valid_k()

find_mse()
