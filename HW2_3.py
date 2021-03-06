import numpy as np
import math as mth
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import Options


def accuracy(cfm):
    x = float(cfm[0][0] + cfm[1][1] + cfm[2][2])
    y = float(cfm[0][0] + cfm[1][0] + cfm[0][1] + cfm[1][1] + cfm[0][2] + cfm[1][2] + cfm[2][0] + cfm[2][1] + cfm[2][2])
    accuracy = x / y
    print "accuracy: ", accuracy
    return accuracy


def precision(cfm):
    precision_x = float(cfm[0][0]) / float((cfm[0][0] + cfm[0][1] + cfm[0][2]))
    precision_y = float(cfm[1][1]) / float((cfm[1][0] + cfm[1][1] + cfm[1][2]))
    precision_z = float(cfm[2][2]) / float(cfm[2][0] + cfm[2][1] + cfm[2][2])
    print "Precision for setosa:", precision_x
    print "Precision for virginica:", precision_y
    print "Precision for versicolor:", precision_z
    return precision_x, precision_y, precision_z


def recall(cfm):
    recall_x = float(cfm[0][0]) / float((cfm[0][0] + cfm[1][0] + cfm[2][0]))
    recall_y = float(cfm[1][1]) / float((cfm[1][1] + cfm[0][1] + cfm[2][1]))
    recall_z = float(cfm[2][2]) / float((cfm[2][2] + cfm[0][2] + cfm[1][2]))
    print "recall for setosa", recall_x
    print "recall for virginica", recall_y
    print "recall for versicolor", recall_z
    return recall_x, recall_y, recall_z


def fmeasure(precision_x, precision_y, precision_z, recall_x, recall_y, recall_z):
    Fmeasure_x = 2 * precision_x * recall_x / (precision_x + recall_x)
    Fmeasure_y = 2 * precision_y * recall_y / (precision_y + recall_y)
    Fmeasure_z = 2 * precision_z * recall_z / (precision_z + recall_z)
    print "Fmeasure for setosa:", Fmeasure_x
    print "Fmeasure for virginica:", Fmeasure_y
    print "Fmeasure for versicolor:", Fmeasure_z
    return Fmeasure_x, Fmeasure_y, Fmeasure_z


def GDA_N_D(X, M, cov1, cov2, cov3, classes,priors):
    dclass1 = []
    dclass2 = []
    dclass3 = []
    cov1 = np.asmatrix(cov1, dtype='float')
    cov2 = np.asmatrix(cov2, dtype='float')
    cov3 = np.asmatrix(cov3, dtype='float')
    X = np.asmatrix(X[:, 0:4], dtype='float')
    M = np.asmatrix(M, dtype='float')
    for i in range(0, len(X)):
        x = (X[i] - M[0])
        y = (X[i] - M[1])
        z = (X[i] - M[2])
        dclass1.append(-mth.log(np.linalg.det(cov1)) - 0.5 * (
            np.dot(np.dot(x, np.linalg.inv(cov1)), x.transpose())) + mth.log(priors[0]))
        dclass2.append(-mth.log(np.linalg.det(cov2)) - 0.5 * (
            np.dot(np.dot(y, np.linalg.inv(cov2)), y.transpose())) + mth.log(priors[1]))
        dclass3.append(-mth.log(np.linalg.det(cov3)) - 0.5 * (
            np.dot(np.dot(z, np.linalg.inv(cov3)), z.transpose())) + mth.log(priors[2]))
    predict_class = []
    for i, j, k in zip(dclass1, dclass2, dclass3):
        if i > j and i > k:
            predict_class.append(classes[0])
        elif j > i and j > k:
            predict_class.append(classes[1])
        else:
            predict_class.append(classes[2])
    return predict_class


def main():
    my_file = np.genfromtxt("data/Iris.txt", delimiter=',', dtype=('|S15'))
    a = my_file[:, 0:4]
    b = my_file[:, 4:5]
    classes = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
    c = np.concatenate((a.astype(np.float), b), axis=1)
    two_D_train_data, two_D_test_data = train_test_split(c[:, 0:5], test_size=0.25, random_state=123)
    subset1_train = two_D_train_data[two_D_train_data[:, 4] == classes[0]]
    subset2_train = two_D_train_data[two_D_train_data[:, 4] == classes[1]]
    subset3_train = two_D_train_data[two_D_train_data[:, 4] == classes[2]]
    ########################################################################
    parameters = Options.Parameters
    Mean_setosa = parameters.Mean_(subset1_train[:, 0:4])
    Mean_virginica = parameters.Mean_(subset2_train[:, 0:4])
    Mean_versicolor = parameters.Mean_(subset3_train[:, 0:4])
    ########################################################################
    COV_setosa = parameters.Covariance(subset1_train[:, 0:4])
    COV_virginica = parameters.Covariance(subset2_train[:, 0:4])
    COV_versicolor = parameters.Covariance(subset3_train[:, 0:4])
    ########################################################################
    prior_setosa=float(len(subset1_train))/float(len(two_D_train_data))
    prior_virginica=float(len(subset2_train))/float(len(two_D_train_data))
    prior_versicolor=float(len(subset3_train))/float(len(two_D_train_data))
    ########################################################################
    Mean = [Mean_setosa,Mean_virginica,Mean_versicolor]
    priors=[prior_setosa,prior_virginica,prior_versicolor]
    predicted_value = GDA_N_D(two_D_test_data, Mean, COV_setosa, COV_virginica, COV_versicolor, classes,priors)
    actual_value = two_D_test_data[:, 4:5].tolist()
    conf_mat = confusion_matrix(actual_value, predicted_value, labels=[classes[0], classes[1], classes[2]])
    ###################################################################################################################
    print "************************************************"
    print "Mean of Setosa:\n", Mean_setosa
    print "Mean of virginica:\n", Mean_virginica
    print "Mean of versicolor:\n", Mean_versicolor
    print "Covariance of Setosa:\n", COV_setosa
    print "Covariance of virginica:\n", COV_virginica
    print "Covariance of versicolor:\n", COV_versicolor
    print "Prior for setosa:",priors[0]
    print "Prior for virginica:",priors[1]
    print "Prior for versicolor:",priors[2]
    print "Confusion Matrix is:\n", conf_mat
    print "************************************************"
    ############################################################################################
    accuracy(conf_mat)
    pre_x, pre_y, pre_z = precision(conf_mat)
    rec_x, rec_y, rec_z = recall(conf_mat)
    fmeasure(pre_x, pre_y, pre_z, rec_x, rec_y, rec_z)
    print "************************************************"

if __name__ == '__main__':
    main()
