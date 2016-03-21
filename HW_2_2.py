import numpy as np
import math as mth
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import Options


def accuracy(cfm):
    accuracy = float(cfm[0][0] + cfm[1][1]) / float(cfm[0][0] + cfm[1][0] + cfm[0][1] + cfm[1][1])
    print "accuracy:",accuracy
    return accuracy


def precision(cfm):
    precision_x = float(cfm[0][0]) / float(cfm[0][0] + cfm[0][1])
    precision_y = float(cfm[1][1]) / float(cfm[1][0] + cfm[1][1])
    print "precision for setosa:", precision_x
    print "precision for virginica", precision_y
    return precision_x, precision_y


def recall(cfm):
    recall_x = float(cfm[0][0]) / float(cfm[0][0] + cfm[1][0])
    recall_y = float(cfm[1][1]) / float(cfm[1][1] + cfm[0][1])
    print "recall for setosa:", recall_x
    print "recall for virginica:", recall_y
    return recall_x, recall_y


def fmeasure(precision_x, precision_y, recall_x, recall_y):
    Fmeasure_x = 2 * precision_x * recall_x / (precision_x + recall_x)
    Fmeasure_y = 2 * precision_y * recall_y / (precision_y + recall_y)
    print "F-measure for setosa", Fmeasure_x
    print "F-measure for virginica", Fmeasure_x
    return Fmeasure_x, Fmeasure_y


def GDA_N_D(X, M, cov1, cov2, classes,priors):
    dclass1 = []
    dclass2 = []
    cov1 = np.asmatrix(cov1, dtype='float')
    cov2 = np.asmatrix(cov2, dtype='float')
    X = np.asmatrix(X[:, 0:4], dtype='float')
    M = np.asmatrix(M, dtype='float')
    for i in range(0, len(X)):
        x = (X[i] - M[0])
        y = (X[i] - M[1])
        dclass1.append(-mth.log(np.linalg.det(cov1)) - 0.5 * (
            np.dot(np.dot(x, np.linalg.inv(cov1)), x.transpose())) + mth.log(priors[0]))
        dclass2.append(-mth.log(np.linalg.det(cov2)) - 0.5 * (
            np.dot(np.dot(y, np.linalg.inv(cov2)), y.transpose())) + mth.log(priors[1]))
    predict_class = []
    for i, j in zip(dclass1, dclass2):
        if i > j:
            predict_class.append(classes[0])
        else:
            predict_class.append(classes[1])
    return predict_class


def main():
    my_file = np.genfromtxt("data/Iris.txt", delimiter=',', dtype=('|S15'))
    a = my_file[:, 0:4]
    b = my_file[:, 4:5]
    classes = ["Iris-setosa", "Iris-virginica"]
    c = np.concatenate((a.astype(np.float), b), axis=1)
    subset1 = c[c[:, 4] == classes[0]]
    subset2 = c[c[:, 4] == classes[1]]
    two_D_data = np.concatenate((subset1, subset2), axis=0)
    two_D_train_data, two_D_test_data = train_test_split(two_D_data[:, 0:5], test_size=0.25, random_state=123)
    subset1_train = two_D_train_data[two_D_train_data[:, 4] == classes[0]]
    subset2_train = two_D_train_data[two_D_train_data[:, 4] == classes[1]]
    #########################################################################
    parameters = Options.Parameters
    Mean_setosa = parameters.Mean_(subset1_train[:, 0:4])
    Mean_virginica = parameters.Mean_(subset2_train[:, 0:4])
    #########################################################################
    COV_setosa = parameters.Covariance(subset1_train[:, 0:4])
    COV_virginica = parameters.Covariance(subset2_train[:, 0:4])
    #########################################################################
    prior_setosa=float(len(subset1_train))/float(len(two_D_train_data))
    prior_virginica=float(len(subset2_train))/float(len(two_D_train_data))
    #########################################################################
    Mean = [Mean_setosa,Mean_virginica]
    priors=[prior_setosa,prior_virginica]
    predicted_value = GDA_N_D(two_D_test_data, Mean, COV_setosa, COV_virginica, classes,priors)
    actual_value = two_D_test_data[:, 4:5]
    conf_mat = confusion_matrix(actual_value, predicted_value, labels=[classes[0], classes[1]])
    ############################################################################################################
    print "************************************************"
    print "Mean of Setosa:\n", Mean_setosa
    print "Mean of virginica:\n", Mean_virginica
    print "Covariance of Setosa:\n", COV_setosa
    print "Covariance of virginica:\n", COV_virginica
    print "Prior for setosa:",priors[0]
    print "Prior for virginica:",priors[1]
    print "Confusion Matrix is:\n", conf_mat
    print "************************************************"
    #############################################################################################################
    accuracy(conf_mat)
    pre_x, pre_y = precision(conf_mat)
    rec_x, rec_y = recall(conf_mat)
    fmeasure(pre_x, pre_y, rec_x, rec_y)
    print "************************************************"


if __name__ == '__main__':
    main()
