import numpy as np
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score, confusion_matrix
import pylab as pl
from math import log
from collections import defaultdict
import operator as op


def doc_Name(lines):
    y = []
    x = []
    for line in lines:
        y += ['spam' if line.split()[0] == 'spam' else 'ham']
        x += [" ".join([word.lower() for word in line.split()[1:]])]
    return (x, y)


def calculateAlpha(docDict, y, total_spam_words, total_ham_words, smoothing=1.):
    spam_sum = sum([docDict[doc_id] for doc_id in docDict if y[doc_id] == 'spam'])
    ham_sum = sum([docDict[doc_id] for doc_id in docDict if y[doc_id] == 'ham'])
    return ((spam_sum + smoothing) / (total_spam_words + 2. * smoothing),
            (ham_sum + smoothing) / (total_ham_words + 2. * smoothing))


def naiveBayesClassifier(x, y):
    word_dict = defaultdict(lambda: defaultdict(lambda: 0))
    parameters = {}
    total_spam_words = 0
    total_ham_words = 0
    for doc_id in range(0, len(x)):
        if y[doc_id] == 'spam':
            total_spam_words += 1
        else:
            total_ham_words += 1
        for word in x[doc_id].split():
            word_dict[word][doc_id] = 1
    for word in word_dict:
        parameters[word] = calculateAlpha(word_dict[word], y, total_spam_words, total_ham_words)

    prior_spam = len([val for val in y if val == 'spam']) / float(len(y))
    prior_ham = len([val for val in y if val == 'ham']) / float(len(y))
    return (parameters, prior_spam, prior_ham)


def predict(x, parameters, prior_spam, prior_ham):
    y = []
    for doc in x:
        ham_meter = 0.
        spam_meter = 0.
        for word in list(set(doc.split())):
            if word in parameters:
                spam_meter += log(parameters[word][0])
                ham_meter += log(parameters[word][1])
        y += ['ham' if spam_meter * prior_spam <= ham_meter * prior_spam else 'spam']
    return y


def metrics(y_test, y_predict, showplot=False,flag=0):
    y_pred = []
    y = []
    for i in range(0, len(y_test)):
        y += [1 if y_test[i] == 'spam' else 0]
        y_pred += [1 if y_predict[i] == 'spam' else 0]

    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    if flag == 0:
        conf_mat = confusion_matrix(y, y_pred, labels=[0, 1])
        print "Confusion Matrix is: \n",conf_mat
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    if (showplot):
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False +ve Rate')
        pl.ylabel('True +ve Rate')
        pl.title('Receiver operating characteristic(ROC)')
        pl.show()
    return (accuracy, precision, recall, f1, roc_auc)


def display(predict):
    print 'Accuracy\t', predict[0]
    print 'Precision\t', predict[1]
    print 'Recall\t', predict[2]
    print 'F measure\t', predict[3]
    print "Area under the ROC curve\t", predict[4]



def crossValidation(x, y, folds=5):
    print "************************************************"
    print 'After',folds, 'fold cross validation'
    x = np.array(x)
    y = np.array(y)
    kf = KFold(len(y), n_folds=folds)
    t = (0, 0, 0, 0, 0)
    for train, test in kf:
        parameters, prior_spam, prior_ham = naiveBayesClassifier(x[train], y[train])
        y_predict = predict(x[test], parameters, prior_spam, prior_ham)
        m = metrics(y[test], y_predict,False,1)
        t = tuple(map(op.add, t, m))

    return tuple([e / float(folds) for e in t])


def main():
    with open('data/SMSSpamCollection.txt', 'r') as f:
        lines = f.readlines()
        print "************************************************"
        print 'total number of documents is: ', len(lines)
    x, y = doc_Name(lines)
    x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.25, random_state=123)
    parameters, prior_spam, prior_ham = naiveBayesClassifier(x_train, y_train)
    y_predict = predict(x_test, parameters, prior_spam, prior_ham)
    metr = metrics(y_test, y_predict, True,0)
    display(metr)
    avg_cross = crossValidation(x, y)
    display(avg_cross)
    print "************************************************"

if __name__ == '__main__':
    main()
