import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

#Reading files
X = pd.read_csv("trainingData.txt",sep='\t',header=None)
Y = pd.read_csv("trainingTruth.txt",sep='\t',header=None)
Y = np.array(Y).ravel()
X = X.fillna(0) ## imputing nan's as 0

#Create Model
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators = 2,random_state=25))
clf.fit(X,Y)
Y_predict = clf.predict_proba(X)


# Binarize the output
y_bin = label_binarize(Y, classes=[1, 2, 3,4])

#Calculate AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], Y_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc)

#Create submission
Xtest = pd.read_csv("testData.txt",sep="\t",header=None)
y_final_prob = clf.predict_proba(Xtest)
y_final_label = clf.predict(Xtest)

sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
sample.columns = ["prob1","prob2","prob3","prob4","label"]
sample.label = sample.label.astype(int)
#Submit this file to dropbox
sample.to_csv("LastName_FirstName.csv",sep="\t" ,index=False,header=None)