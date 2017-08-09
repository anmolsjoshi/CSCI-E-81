
# coding: utf-8

# In[1]:
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

    
import sys
import csv
import re
import time
import pandas


### Fills in missing data on the training set using KNN like algorithm

#--------  fill_missing_data
# A function for fitting missing data and replacing NaN's and NA's on
# the training set using KNN like algorithm
# Input: 
#      x_train (n x p pandas DataFrame containing all the training features data)
#      y_train (n x 1 pandas DataFrame containing all the training classes)
#      k (integer is the number of neighbors)
# Return: 
#      trainingData (n x p pandas DataFrame containing all the filled training features data)

def fill_missing_data(trainX, trainY, k):

    # Initializes empty array 
    empty = []

    # Uses a for and if statement to store all the rows in the training set with NA or NaN values
    for ii in range(0, len(trainX)):
        a = np.array(trainX.iloc[[ii],:])
        if np.sum(np.isnan(a)) != 0:
            empty = np.append(empty, ii)

    # Splits the training data in two dataframes - missing data and full data for both the training features and classes
    missing_data = trainX.iloc[empty, :]
    missing_data_class = trainY.iloc[empty, :]

    # Finds the full data index values by removing empty indices from all the indicies of the training data
    full_data_range = range(0, len(trainX))
    full_data_loc = [x for x in full_data_range if x not in empty]
    full_data = trainX.iloc[full_data_loc, :]
    full_data_class = trainY.iloc[full_data_loc, :]
    
    #Initializes required indicies for iterations and empty DataFrame to append filled data
    index_values = missing_data.index.values

    filled_data = pandas.DataFrame()
    
    #For loop to index missing data 
    for ii in index_values:
        
        row = missing_data.ix[[ii]]
        
        #Finds the class of row of missing_data
        class_ = int(missing_data_class.ix[ii])
        
        #Splits the full training classes to those equal to the class of the specific row of the full_data
        class_split = full_data_class[full_data_class[0] == class_]
        
        #Finds the index values of the full_data with that class and splices it from full_data to find nearest neighbour
        class_split_index = class_split.index.values
        sub_data = full_data.ix[class_split_index]
        
        #Finds Euclidean Distance by subbing NA with 0 and sorts distance from low to high
        diff = pandas.DataFrame((sub_data.values - row.values)**2, index = sub_data.index.values)
        diff = diff.fillna(0)
        diff = pandas.DataFrame(diff.sum(axis = 1), index = diff.index.values)
        diff = diff.sort(0, ascending = True)
        
        #Finds the index values of the k-nearest neighbors with lowest Euclidean distance
        diff_to_use = diff.index.values[:k]
        
        #Finds the location of the NaN in the row we are iterating
        nanloc = np.where(np.isnan(row))[1]
        
        #Replaces the NaN and NA with the mean of the k-nearest neighbors by taking an average of the 
        row.loc[ii, nanloc] = np.array(np.mean(sub_data.ix[diff_to_use, nanloc]))
        filled_data = filled_data.append(row)
    
    #Adds the index of the originial mising_data to the filled_data
    filled_data = filled_data.set_index(index_values)
    
    #Concatenates the missing_data and full_data and sorts index to now return the training set with all NA and NaN values filled. 
    trainingData = pandas.concat([filled_data, full_data])
    trainingData = trainingData.sort_index(axis = 0)
        
    return trainingData


### Binarizes class labels 

#--------  binarize
# A function to binarize the class labels for example for 4 classes, if Class = 1, output = [1 0 0 0]
# Input: 
#      trainYfile (str filename of the training classes data)
# Return: 
#      label_set (n x 4 arrary of 1's and 0's to binarize class)

def binarize(trainYfile):
    
    #Initializes values for use in the function
    label_set = []
    labels = []
    
    #Reads file containing training classes
    with open('data/' + trainY, 'r') as hw_label:
        hw_label = csv.reader(hw_label, delimiter="\t")
        for row in hw_label:
            y = float(row[0])
            labels.append((y))
    
    #Creates n x 4 array so that if class is equal the real class, then output = 1, otherwise output = 0
    for cls in range(4):
        z = np.array([int(i == (cls+1)) for i in labels])
        label_set.append(z)
    
    #Returns binarized class labels
    return label_set


### Binarizes class labels 

#--------  binarize
# A function to binarize the class labels for example for 4 classes, if Class = 1, output = [1 0 0 0]
# Input: 
#      trainYfile (str filename of the training classes data)
# Return: 
#      label_set (n x 4 arrary of 1's and 0's to binarize class)

def binarize(trainYfile):
    
    #Initializes values for use in the function
    label_set = []
    labels = []
    
    #Reads file containing training classes
    with open('data/' + trainY, 'r') as hw_label:
        hw_label = csv.reader(hw_label, delimiter="\t")
        for row in hw_label:
            y = float(row[0])
            labels.append((y))
    
    #Creates n x 4 array so that if class is equal the real class, then output = 1, otherwise output = 0
    for cls in range(4):
        z = np.array([int(i == (cls+1)) for i in labels])
        label_set.append(z)
    
    #Returns binarized class labels
    return label_set



def get_c_gamma_pll(X, y, kernel):
        # Train classifiers
        # For an initial search, a logarithmic grid with basis 10 is often helpful. 
        # Using a basis of 2, a finer tuning can be achieved but at a much higher cost.
        print(time.asctime( time.localtime(time.time()) ), '- Start .....')

        C_range = [0.5, 1.0, 5]
        gamma_range = [0.1, 0.5, 1.0, 5]

        print(time.asctime( time.localtime(time.time()) ), '- Before grid .....')
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedKFold(n_splits=2)
        print(time.asctime( time.localtime(time.time()) ), '- Before grid search .....')
        grid = GridSearchCV(svm.SVC(kernel=kernel, cache_size=25600), param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
        print(time.asctime( time.localtime(time.time()) ), '- Before grid fit .....')
        grid.fit(X, y)
        print(time.asctime( time.localtime(time.time()) ), '- After grid fit .....')

        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

        return grid

 

def do_classification(Xin, y, cl, XTin, kernel, C, gamma, do_plot):
    # discard featrues using tree-based estimators
    selector = ExtraTreesClassifier()
    selector = selector.fit(Xin, y)
    selector.feature_importances_ 
    smodel = SelectFromModel(selector, prefit=True)
    idxs = smodel.get_support(indices=True)
    X = np.array(Xin)[:, idxs]    # feature set for training data after variance reduction
    XT = np.array(XTin)[:, idxs]  # feature set for test data after variance reduction
    print("# of features removed: ", len(Xin[0])-len(X[0]))
             

    # Classification and ROC Analysis
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=3)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['seagreen', 'yellow', 'blue'])
    lw = 2

    if do_plot == 1:
       fig = plt.figure()

    i = 0
    best_auc = 0
    for (train, test), color in zip(cv.split(X, y), colors):
        print(time.asctime( time.localtime(time.time()) ), '- iteration: ', i)

        classifier = svm.SVC(cache_size=25600, kernel=kernel, probability=True, C=C, gamma=gamma).fit(X[train], y[train])
        fitted = classifier.predict_proba(X[test])
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if best_auc < roc_auc:
           best_auc = roc_auc
           best_classifier = classifier

        if do_plot == 1:
           plt.plot(fpr, tpr, lw=lw, color=color,
                    label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        print('ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    if do_plot == 1:
       plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
                label='Random Chance')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    if do_plot == 1:
       plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
                label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    print('Mean ROC (area = %0.2f)' % mean_auc)

    print(time.asctime( time.localtime(time.time()) ), '- best classifier probas')
    best_probas_ = best_classifier.predict_proba(XT)[:,1]
    

    if do_plot == 1:
       plt.xlim([-0.05, 1.05])
       plt.ylim([-0.05, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('ROC - ' + 'Class Label: ' + str(cl+1)  + '  Kernel: ' + kernel + '\n[C: ' + 
                 str(C) + '; gamma: ' + str(gamma) + '; # of Features Dropped: ' + str(len(Xin[0])-len(X[0])) + ']')
       plt.legend(loc="lower right")
       #plt.show()
       plt.savefig('plots/'+kernel+str(cls)+'.png')
    
    return best_probas_



#------
# MAIN
#------

# Read data from files
# set num_rows to pick subset for faster test runs, else set it to 0
num_rows = 0
if num_rows > 0:
   trainX = pandas.read_csv('data/trainingData.txt', sep = '\t', header = None, nrows=num_rows)
   trainY = pandas.read_csv('data/trainingTruth.txt', sep = '\t', header = None, nrows=num_rows)
   testX = pandas.read_csv('data/blindData.txt', sep = '\t', header = None, nrows=num_rows)
else:
   trainX = pandas.read_csv('data/trainingData.txt', sep = '\t', header = None)
   trainY = pandas.read_csv('data/trainingTruth.txt', sep = '\t', header = None)
   testX = pandas.read_csv('data/blindData.txt', sep = '\t', header = None)

# command line input:
#     0 - replace NA with 0
#     1 - replace NA using kNN 
knn_yes = int(sys.argv[1])

if knn_yes == 1:
    print('Replace/Fill NA /w values using kNN...')
    trainX = fill_missing_data(trainX, trainY, k = 10)
    fill_flag='fillKNN'
else:
    trainX = trainX.fillna(0)
    fill_flag='fillZEROS'

trainY = 'trainingTruth.txt'
input_y_all = binarize(trainY)

X = trainX.as_matrix()
XT = testX.as_matrix()

# verify nrows read 
print(len(X), len(XT))

n_samples, n_features = X.shape
# verify nrows for teat / blind data
print(len(XT))


all_probs = []
for cls in range(4):
    if num_rows > 0:
       y = input_y_all[cls][0:num_rows]
    else:
       y = input_y_all[cls]
    print(len(y))

    kernels = ['poly']

    for i,k in enumerate(kernels):
        print('Kernel: ', k, 'Class: ', cls)

        grid = get_c_gamma_pll(X,y, k)
        print('Class: ', cls)
        C = grid.best_params_['C']
        gamma = grid.best_params_['gamma']

        print('Gamma: ', gamma)
        print('    C: ', C)

        # set do_plot to 1 to do plots, else set it to 0 to skip
        do_plot = 0
        probs = do_classification(X, y, (cls+1), XT, k, gamma, C, do_plot)
        if k == 'poly':
           all_probs.append(probs)
 
# save all probabilities as csv file
pd = pandas.DataFrame(all_probs).T
pd['class'] = pd.idxmax(axis=1) + 1
pd.to_csv("data/HW3_Final_Using_blindData_" + fill_flag + ".csv", index=False, header=False, sep="\t")


# In[ ]:
