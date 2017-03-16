
import csv
import sys
import numpy as np

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegressionCV, LogisticRegression


# Inflation Analysis
def inflation(value, year):
    inflation = 0
    if year == 2011: inflation = 1.07;
    if year == 2012: inflation = 1.05;
    if year == 2013: inflation = 1.04;
    if year == 2014: inflation = 1.02;
    return inflation * value

def append_to_array(arr1, arr2):
    """
    Append two np arrays together
    if arr1 is empty, set arr1 to arr2
    :param arr1: array 1
    :param arr2: array 2
    :return:
    """
    if arr1.size == 0:
        arr1 = np.array([arr2])
        return arr1

    arr1 = np.vstack((arr1, np.array([arr2])))
    return arr1


X = np.array([])
Y = np.array([])

# Loading the Data
with open('Salaries.csv', 'rb') as csvfile:
    csv = csv.reader(csvfile, delimiter=',')

    Header_skipped = False
    count = 0
    for row in csv:
        # skip the first line in csv file. contains header.
        if not Header_skipped: Header_skipped = True; continue;

        job_title = row[2].lower()
        year = float(row[9])
        total_pay = inflation(float(row[7]), year)

        X = append_to_array(X, job_title)
        Y = append_to_array(Y, total_pay)

        count += 1
        #sys.stdout.write("\r" + str(float(count)/150000 * 100))
        #sys.stdout.flush()
        if count > 1000: break

labler_job = preprocessing.LabelEncoder()
X1 = labler_job.fit_transform(X[:, 0])
X1I = labler_job.inverse_transform(X1)

#X_train = np.array([X1[0], X2[0], X3[0]])#, np.float64)

X_train = np.array([])
for x_val in X1:
    X_train = append_to_array(X_train, x_val)

# Models used for prediction
methods = [LinearRegression(), BayesianRidge(), svm.SVR(), svm.LinearSVR()]

print; print; print;

for method in methods:
    searchCV = method
    searchCV.fit(X_train, Y)
    #kf = KFold(n_splits=2)
    #for train, test in kf.split(X_train):
    #    X_train_jr, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
    #    searchCV.fit(X_train, Y)

    print searchCV.predict(X_train[43])
    exit()

#print "\n" + str(method) + "\n" + str(cross_val_score(searchCV, X_train, Y))