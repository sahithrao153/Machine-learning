import numpy as np
import math as mth
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import Options


def accuracy(cfm):
    accuracy = (cfm[0][0] + cfm[1][1]) / (cfm[0][0] + cfm[1][0] + cfm[0][1] + cfm[1][1])
    print "accuracy: %s" % accuracy
    return accuracy


def precision(cfm):
    precision = cfm[0][0] / (cfm[0][0] + cfm[0][1])
    print "precision: %s" % precision
    return precision


def recall(cfm):
    recall = cfm[0][0] / (cfm[0][0] + cfm[1][0])
    print "recall: %s" % recall
    return recall


def fmeasure(precision, recall):
    Fmeasure = 2 * precision * recall / (precision + recall)
    print "F-measure: %s" % Fmeasure
    return Fmeasure


def GDA_1_D_compare(param, param1, classes):
    if param > param1:
        return str(classes[0])
    return str(classes[1])


def GDA_1_D_formulae(X, M, SD, prior):
    return -mth.log(SD) - 0.5 * ((float(X) - float(M)) ** 2 / SD ** 2) + mth.log(prior)


def GDA_1_D(X, M, SD, classes,priors):
    predict_class = []
    for i in range(0, len(X)):
        predict_class.append(GDA_1_D_compare(GDA_1_D_formulae(X[i], M[0], SD[0], priors[0]),
                                             GDA_1_D_formulae(X[i], M[1], SD[1], priors[1]), classes))
    return predict_class


def main():
    my_file = np.genfromtxt("data/Iris.txt", delimiter=',', dtype=('|S15'))
    a = my_file[:, 3:4]
    b = my_file[:, 4:5]
    c = np.concatenate((a.astype(np.float), b), axis=1)
    classes = ["Iris-setosa", "Iris-virginica"]
    parameters = Options.Parameters
    subset1 = c[c[:, 1] == classes[0]]
    subset2 = c[c[:, 1] == classes[1]]
    one_D_data = np.concatenate((subset1, subset2), axis=0)
    one_D_train_data, one_D_test_data = train_test_split(one_D_data[:, 0:2], test_size=0.25, random_state=36)
    subset1_train = one_D_train_data[one_D_train_data[:, 1] == classes[0]]
    subset2_train = one_D_train_data[one_D_train_data[:, 1] == classes[1]]
    ##########################################################################
    Mean_setosa = parameters.Mean_(subset1_train[:, 0])
    Mean_virginica = parameters.Mean_(subset2_train[:, 0])
    ##########################################################################
    SD_setosa = parameters.Standard_deviation(subset1_train[:, 0])
    SD_virginica = parameters.Standard_deviation(subset2_train[:, 0])
    ##########################################################################
    prior_setosa=float(len(subset1_train))/float(len(one_D_train_data))
    prior_virginica=float(len(subset2_train))/float(len(one_D_train_data))
    ##########################################################################
    Mean = [Mean_setosa, Mean_virginica]
    SD = [SD_setosa, SD_virginica]
    priors=[prior_setosa,prior_virginica]
    predicted_value = GDA_1_D(one_D_test_data[:, 0], Mean, SD, classes,priors)
    actual_value = one_D_test_data[:, 1:2].tolist()
    conf_mat = confusion_matrix(actual_value, predicted_value, labels=[classes[0], classes[1]])
    ################################################################################################
    print "Mean of Setosa: ", Mean_setosa, "\nMean of Virginica: ", Mean_virginica
    print "Variance of Setosa: ", SD_setosa, "\nVariance of virginica: ", SD_virginica
    print "Prior for setosa:",priors[0]
    print "Prior for virginica",priors[1]
    print "Confusion Matrix is:\n", conf_mat
    ################################################################################################
    accuracy(conf_mat)
    pre = precision(conf_mat)
    rec = recall(conf_mat)
    fmeasure(pre, rec)


if __name__ == '__main__':
    main()
