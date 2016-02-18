
import matplotlib.pyplot as plt
import numpy as np


def mse(Z,theta,Y):

    Yt =np.dot(theta.T,Z.T)
    difference = (Yt.T-Y)**2
    error = sum(difference)/len(Y)
    print error





def polynomial(X,Y,degree,X_test):

    z_test = np.ones((X_test.size,1), dtype=np.int)
    Z_test = np.append(z_test,X_test,axis =1)
    z_train = np.ones((X.size,1), dtype=np.int)
    Z_train = np.append(z_train,X,axis =1)
    for i in range(degree-1):
        X =X*X
        X_test = X_test*X_test
        Z_train =np.append(Z_train,X,axis=1)
        Z_test =np.append(Z_test,X_test,axis=1)

    theta = np.dot(np.dot(np.linalg.inv(np.dot(Z_train.T,Z_train)),Z_train.T),Y)
   # print theta

    return Z_train,theta,Z_test

#def plot_data(Z,theta,X,Y):
#
#     Z_test = Z[180:200]
#     X_test = X[180:200]
#     Y_test = np.dot(theta.T,Z_test.T)
#     Y_test = Y_test.reshape(Y_test.size,1)
#     plt.scatter(X,Y,color="red")
#     plt.scatter(X_test,Y_test)
#     plt.show()
def split_data(X,Y,degree):


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
          Z_train,theta,Z_test = polynomial(X_train,Y_train,degree,X_test)
          mse(Z_test,theta,Y_test)





def main():

    files  =['datasets/svar-set1.dat', 'datasets/svar-set2.dat', 'datasets/svar-set3.dat',
                  'datasets/svar-set4.dat']
    file_to_use = input("enter the file to be used:")
    X,Y = np.loadtxt(files[file_to_use-1],unpack =True,usecols=[0,1])
    degree = input("enter the degree of the polynomial:")

    X = X.reshape(X.size,1)

    Y = Y.reshape(Y.size,1)
    split_data(X,Y,degree)




if __name__ == "__main__":
    main()