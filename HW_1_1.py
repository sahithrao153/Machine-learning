import numpy as np
import matplotlib.pyplot as plt


# this function is to find the mean squared error for training data.
def find_mse_train_data(Z, theta, Y, train_size):
    Z = Z[0:int(train_size * len(Y))]
    Y = Y[0:int(train_size * len(Y))]
    ys = np.dot(theta.T, Z.T)
    diff = (ys.T - Y) ** 2
    msError = sum(diff) / len(Y)
    return msError


# this function is to find the mean squared error for testing data.
def find_mse_test_data(Z, theta, Y, test_size):
    Z = Z[int(len(Y) - (200 * test_size)):len(Y)]
    Y = Y[int(len(Y) - (200 * test_size)):len(Y)]
    ys = np.dot(theta.T, Z.T)
    diff = (ys.T - Y) ** 2
    msError = sum(diff) / len(Y)
    return msError


# this function is to plot the testing data.
def plot_data(Z, theta, X, Y, test_size):
    Z_test = Z[int(len(Y) - (200 * test_size)):len(Y)]
    X_test = X[int(len(Y) - (200 * test_size)):len(Y)]
    Y_test = np.dot(theta.T, Z_test.T)
    Y_test = Y_test.reshape(Y_test.size, 1)
    plt.scatter(X_test, Y[int(len(Y) - (200 * test_size)):len(Y)], color="blue")
    plt.plot(X_test, Y_test, 'go-', )
    plt.show()


# this is to return the theta and Z using the polynomial function.
def poly_function(X, Y, Z, degree_poly, test_size):
    for i in range(degree_poly - 1):
        X = X * X
        Z = np.append(Z, X, axis=1)
    Z_train = Z[0:int(len(Y) - (200 * test_size))]
    Y = Y[0:int(len(Y) - (200 * test_size))]
    theta = np.dot(np.dot(np.linalg.inv(np.dot(Z_train.T, Z_train)), Z_train.T), Y)

    return Z, theta


def split_data(X,Y,degree,test_size):
   X_sets =  np.split(X,10)
   Y_sets = np.split(Y,10)
   for i in range(len(X_sets)):
      X_test =np.vstack( X_sets[i])
      Y_test = np.vstack(Y_sets[i])
      if i<len(X_sets)-1:
         X_train = np.vstack(X_sets[i+1:])
         Y_train =np.vstack(Y_sets[i+1:])
      elif i==len(X_sets)-1 :
         X_train = np.vstack(X_sets[:i])
         Y_train = np.vstack(Y_sets[:i])
      while i>0:
          tempX = np.vstack(X_sets[i-1])
          X_train = np.append(tempX,X_train)
          tempY = np.vstack(Y_sets[i-1])
          Y_train = np.append(tempY,Y_train)
          i = i-1
      X_train = np.vstack(X_train)
      Y_train = np.vstack(Y_train)
      Z_train,theta,Z_test = poly_function(X_train,Y_train,degree,X_test)
      sum_mse = sum_mse + float(find_mse_test_data(Z_test,theta,Y_test,test_size ))


# main function, the program starts here.
def main():
    files_list = ['datasets/svar-set1.dat', 'datasets/svar-set2.dat', 'datasets/svar-set3.dat',
                  'datasets/svar-set4.dat']
    print 'files list is :\n', files_list
    f_index = int(raw_input('enter the filename index from the list from 1 to 4 '))
    degree_poly = int(raw_input('enter the degree of the polynomial from 1 to n '))
    test_size = float(raw_input('enter the test size from 0.0 to 1.0 '))
    train_size = 1.0 - test_size
    X, Y = np.loadtxt(files_list[f_index - 1], unpack=True, usecols=[0, 1])
    ''' plt.title('data')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(X, Y)
    plt.show()'''
    X = X.reshape(X.size, 1)
    Y = Y.reshape(Y.size, 1)
    Z_ones = np.ones((X.size, 1), dtype=np.int)
    Z = np.append(Z_ones, X, axis=1)
    Z, theta = poly_function(X, Y, Z, degree_poly, test_size)
    print "MSE for test data", find_mse_test_data(Z, theta, Y, test_size)
    print "MSE for train data", find_mse_train_data(Z, theta, Y, train_size)
    plot_data(Z, theta, X, Y, test_size)


if __name__ == '__main__':
    main()
