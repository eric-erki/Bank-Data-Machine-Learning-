from copy import deepcopy
import sklearn.linear_model as lm
import sklearn.neighbors as knn
import pandas as pd
import tflearn
import numpy as np
from data_encoding import one_hot, label_vec, binary_vectorizing
from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
import tensorflow.contrib.learn as learn
#import tflearn as learn


BANK_TRAINING = "bank-training_new.csv"
BANK_VALIDATION = "bank-crossvalidation_new.csv"
LOG_FILE = "log.txt"

COLUMNS = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration",
           "campaign","pdays","previous","poutcome","y"]
CATEGORICAL_COLUMNS = ["job","marital","education","default","housing","loan","contact","day","month","poutcome"]
CONTINUOUS_COLUMNS = ["age","balance","duration","campaign","pdays","previous"]
LABEL = "y"

min_max_scaler = preprocessing.MinMaxScaler()
standard_scaler = preprocessing.StandardScaler()
pca = PCA(n_components=5)
ipca = IncrementalPCA(n_components=5)

############################
training_set = pd.read_csv(BANK_TRAINING, names=COLUMNS, skipinitialspace=True, skiprows=1)
training_label_set = deepcopy(training_set[LABEL])
del training_set[LABEL]
#
training_set = one_hot(training_set, CATEGORICAL_COLUMNS)
training_label_set = binary_vectorizing(training_label_set, ['no', 'yes'])
#
training_set_mn_scaled = min_max_scaler.fit_transform(training_set)
training_set_mn_scaled = pd.DataFrame(training_set_mn_scaled)
#
training_set_sc_scaled = standard_scaler.fit_transform(training_set)
training_set_sc_scaled = pd.DataFrame(training_set_sc_scaled)
#
training_set_pca_sc_scaled = ipca.fit_transform(deepcopy(training_set_sc_scaled))

##########################
validation_set = pd.read_csv(BANK_VALIDATION, names=COLUMNS, skipinitialspace=True, skiprows=1)
validation_label_set = deepcopy(validation_set[LABEL])
del validation_set[LABEL]
#
validation_set = one_hot(validation_set, CATEGORICAL_COLUMNS)
validation_label_set = binary_vectorizing(validation_label_set, ['no', 'yes'])
#
validation_set_mn_scaled = min_max_scaler.fit_transform(validation_set)
validation_set_mn_scaled = pd.DataFrame(validation_set_mn_scaled)
#
validation_set_sc_scaled = standard_scaler.fit_transform(validation_set)
validation_set_sc_scaled = pd.DataFrame(validation_set_sc_scaled)
#
validation_set_pca_sc_scaled = ipca.fit_transform(deepcopy(validation_set_sc_scaled))

#####################################################################
#Array of Tuples of Model,Training Data, Validation Data, Description
pipelines = []

pipelines.append((lm.LogisticRegression(), training_set, validation_set, "Logistic Regression"))
pipelines.append((lm.LogisticRegression(), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->Logistic Regression"))
pipelines.append((lm.LogisticRegression(), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->Logistic Regression"))
pipelines.append((lm.LogisticRegression(), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->Logistic Regression"))

pipelines.append((knn.KNeighborsClassifier(), training_set, validation_set, "K-Nearest Neighbours"))
pipelines.append((knn.KNeighborsClassifier(), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->K-Nearest Neighbours"))
pipelines.append((knn.KNeighborsClassifier(), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->K-Nearest Neighbours"))
pipelines.append((knn.KNeighborsClassifier(), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->K-Nearest Neighbours"))

pipelines.append((svm.SVC(kernel='linear',  max_iter=1000000), training_set, validation_set, "Linear Support Vector Machine"))
pipelines.append((svm.SVC(kernel='linear',  max_iter=1000000), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->Linear Support Vector Machine"))
pipelines.append((svm.SVC(kernel='linear',  max_iter=1000000), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->Linear Support Vector Machine"))
pipelines.append((svm.SVC(kernel='linear',  max_iter=1000000), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->Linear Support Vector Machine"))

pipelines.append((svm.SVC(kernel='poly', degree=3,  max_iter=1000000), training_set, validation_set, "3Poly Support Vector Machine"))
pipelines.append((svm.SVC(kernel='poly', degree=3,  max_iter=1000000), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->3Poly Support Vector Machine"))
pipelines.append((svm.SVC(kernel='poly', degree=3,  max_iter=1000000), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->3Poly Support Vector Machine"))
pipelines.append((svm.SVC(kernel='poly', degree=3,  max_iter=1000000), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->3Poly Support Vector Machine"))

pipelines.append((svm.SVC(kernel='poly', degree=16,  max_iter=1000000), training_set, validation_set, "16Poly Support Vector Machine"))
pipelines.append((svm.SVC(kernel='poly', degree=16,  max_iter=1000000), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->16Poly Support Vector Machine"))
pipelines.append((svm.SVC(kernel='poly', degree=16,  max_iter=1000000), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->16Poly Support Vector Machine"))
pipelines.append((svm.SVC(kernel='poly', degree=16,  max_iter=1000000), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->16Poly Support Vector Machine"))

pipelines.append((svm.SVC(kernel='rbf', degree=16,  max_iter=1000000), training_set, validation_set, "16RBF Support Vector Machine"))
pipelines.append((svm.SVC(kernel='rbf', degree=16,  max_iter=1000000), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->16RBF Support Vector Machine"))
pipelines.append((svm.SVC(kernel='rbf', degree=16,  max_iter=1000000), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->16RBF Support Vector Machine"))
pipelines.append((svm.SVC(kernel='rbf', degree=16,  max_iter=1000000), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->16RBF Support Vector Machine"))

pipelines.append((tree.DecisionTreeClassifier(), training_set, validation_set, "Decision Tree"))
pipelines.append((tree.DecisionTreeClassifier(), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->Decision Tree"))
pipelines.append((tree.DecisionTreeClassifier(), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->Decision Tree"))
pipelines.append((tree.DecisionTreeClassifier(), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->Decision Tree"))

pipelines.append((RandomForestClassifier(), training_set, validation_set, "Random Forest"))
pipelines.append((RandomForestClassifier(), training_set_mn_scaled, validation_set_mn_scaled, "MinMax-Scaled->Random Forest"))
pipelines.append((RandomForestClassifier(), training_set_sc_scaled, validation_set_sc_scaled, "Standard-Scaled->Random Forest"))
pipelines.append((RandomForestClassifier(), training_set_pca_sc_scaled, validation_set_pca_sc_scaled, "Standard-Scaled->PCA->Random Forest"))


#####################################################################
f = open(LOG_FILE, 'w+')

for pipeline in pipelines:
    mdl, train, validate, description = pipeline
    mdl.fit(train, training_label_set)
    score = str( mdl.score(validate, validation_label_set) )
    describe = description +"  "+ score + "\n\n**********\n\n"
    f.write(describe)
    print(describe)

f.close()