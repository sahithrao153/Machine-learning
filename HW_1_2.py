import numpy as np
import matplotlib.pyplot as plt


def find_mse_test_data(Z, theta, Y, test_size):
    Z = Z[int(len(Y) - (len(Y) * test_size)):len(Y)]
    Y = Y[int(len(Y) - (len(Y) * test_size)):len(Y)]
    ys = np.dot(theta.T, Z.T)
    diff = (ys.T - Y) ** 2
    msError = sum(diff) / len(Y)
    return msError


def find_mse_train_data(Z, theta, Y, train_size):
    Z = Z[0:int(train_size * len(Y))]
    Y = Y[0:int(train_size * len(Y))]
    ys = np.dot(theta.T, Z.T)
    diff = (ys.T - Y) ** 2
    msError = sum(diff) / len(Y)
    return msError


def poly_function(X, Y, z, degree_poly, test_size):
    for i in range(degree_poly - 1):
        X = X * X
        z = np.append(z, X, axis=1)
    z_train = z[0:int(len(Y) - (len(Y) * test_size))]
    Y = Y[0:int(len(Y) - (len(Y) * test_size))]
    theta = np.dot(np.linalg.inv(np.dot(z_train.T, z_train)), np.dot(z_train.T, Y))

    return z, theta


def plot_data(Z, theta, X, Y, test_size):
    Z_test = Z[int(len(Y) - (len(Y) * test_size)):len(Y)]
    X_test = X[int(len(Y) - (len(Y) * test_size)):len(Y)]
    Y_test = np.dot(theta.T, Z_test.T)
    Y_test = Y_test.reshape(Y_test.size, 1)
    plt.scatter(X_test, Y[int(len(Y) - (len(Y) * test_size)):len(Y)], color="blue")
    plt.plot(X_test, Y_test, 'go-', )
    plt.show()


def main():
    files_list = ['datasets/mvar-set1.dat', 'datasets/mvar-set2.dat', 'datasets/mvar-set3.dat',
                  'datasets/mvar-set4.dat']
    print 'files list is :\n', files_list
    f_index = int(raw_input('enter the filename index from the list from 1 to 4 '))
    degree_poly = int(raw_input('enter the degree of the polynomial from 1 to n '))
    test_size = float(raw_input('enter the test size from 0.0 to 1.0 '))
    train_size = 1.0 - test_size
    multi_list = np.loadtxt(files_list[f_index - 1], unpack=True)
    print len(multi_list)
    X = multi_list[0:len(multi_list) - 1]
    X = X.reshape(X.size, 1)
    Y = multi_list[len(multi_list) - 1:len(multi_list)]
    Y = Y.reshape(Y.size, 1)
    z_ones = np.ones((X.size, 1), dtype=np.int)
    Z = np.append(z_ones, X, axis=1)
    Z, theta = poly_function(X, Y, Z, degree_poly, test_size)
    print "MSE for train data", find_mse_train_data(Z, theta, Y, train_size)
    print "MSE for test data", find_mse_test_data(Z, theta, Y, test_size)
    plot_data(Z, theta, X, Y, test_size)


if __name__ == '__main__':
    main()
