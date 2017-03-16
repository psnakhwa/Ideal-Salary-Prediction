
import csv
import numpy as np
from Helper_Functions import *

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import svm

X_reg_list = []
Y_reg_list = []

job_input = raw_input("Job Title >> ").lower()
job_status= raw_input("Status >> ").lower()

print "Preprocessing..."
with open('Salaries.csv', 'rb') as csvfile:
    csv = csv.DictReader(csvfile, delimiter=',')

    Header_skipped = False
    count = 0

    for row in csv:
        count += 1

        # skip the first line in csv file. contains header.
        if not Header_skipped: Header_skipped = True; continue;

        # if the length of row is 0, skip
        if len(row) == 0:
            continue

        # if job title not provided, do not save
        if row['JobTitle'] == 'Not provided': continue

        # if job title is not same as output, skip
        if job_input in row['JobTitle'].lower(): continue

        #save Job Title and Status to X, save salary to Y
        X_reg_list.append([row['JobTitle'].lower(), row['Status'].lower()])
        Y_reg_list.append([float(row['Salary'])])


if len(X_reg_list) == 0:
    print "Job Not Found!"
    exit()


print "converting to np array..."
X = np.array(X_reg_list)
Y = np.array(Y_reg_list)


# Need to encode class data (Job_Title, Status)
labler_job = preprocessing.LabelEncoder()
labler_status = preprocessing.LabelEncoder()

X1 = labler_job.fit_transform(X[:, 0])
X1I = labler_job.inverse_transform(X1)

X2 = labler_status.fit_transform(X[:, 1])
X2I = labler_status.inverse_transform(X2)


# Seperate data into training and validation
split = int(len(X1)/4)

X_train = np.array(zip(X1[:split], X2[:split]))
Y_train = Y[:split]
X_val = np.array(zip(X1[split:], X2[split:]))
Y_val = Y[split:]


methods = [linear_model.LinearRegression(), linear_model.BayesianRidge(), linear_model.Lasso(), svm.SVR()]

pred_list = []
status_list = []

what_status_to_use = []
if job_status not in ['ft', 'pt']:
    what_status_to_use = ['ft', 'pt']
else:
    what_status_to_use = [job_status]

for stat in what_status_to_use:
    to_predict = [job_input, stat]
    encoded_to_predict = encode_prediction(to_predict, X1, X1I, X2, X2I)
    print "Info For:", " ".join(to_predict)

    for method, meth_name in zip(methods, ['LinReg', 'BayesRidge', 'Lasso', 'SVM(R)']):
        searchCV = method
        searchCV.fit(X_train, Y_train)

        prediction = searchCV.predict(encoded_to_predict)

        # sometimes output of prediction is list or list inside of list
        # code below just extracts inner content from list and save it as float
        val = 0
        try:
            val = prediction[0]
        except:
            pass
        try:
            val = prediction[0][0]
        except:
            pass
        pred_list.append(val)
        status_list.append(encoded_to_predict[1])

        print
        print "ALGORITHM:          ", meth_name
        print "PREDICTED SALARY:   ", val
        print "NEG MEAN SQ ERROR:  ", cross_val_score(searchCV, X_val, Y_val, scoring="neg_mean_squared_error")
        print "NEG MEAN ABS ERROR: ", cross_val_score(searchCV, X_val, Y_val, scoring="neg_mean_absolute_error")


plt.plot([y for x, y, z in zip(X1, X2, Y) if y==0], [z for x, y, z in zip(X1, X2, Y) if y==0], 'r.', label="FT Actual", markersize = 2)
plt.plot([y for x, y, z in zip(X1, X2, Y) if y==1], [z for x, y, z in zip(X1, X2, Y) if y==1], 'b.', label="PT Actual", markersize = 2)
plt.plot(status_list, pred_list, 'g*', label="Predictions", markersize = 10)

plt.xlabel("Status (0=FT, 1=PT)")
plt.ylabel("Salary")
plt.title(job_input.upper())
plt.legend()
plt.axis([-1, 2, 0, max(Y) + 100])
plt.show()