from copy import deepcopy
import pandas as pd
import numpy as np
from data_encoding import one_hot, binary_vectorizing, ternary_vectorizing
from sklearn import preprocessing
import tensorflow.contrib.learn as learn
from tensorflow.contrib import layers


BANK_TRAINING = "bank-training_new.csv"
BANK_VALIDATION = "bank-crossvalidation_new.csv"
LOG_FILE = "log.txt"

COLUMNS = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration",
           "campaign","pdays","previous","poutcome","y"]
CATEGORICAL_COLUMNS = ["job","marital","education","default","housing","loan","contact","day","month","poutcome"]

CATEGORICAL_COLUMNS_2 = ["default","housing","loan","day","month","poutcome"]
CONTINUOUS_COLUMNS = ["age","balance","duration","campaign","pdays","previous"]
LABEL = "y"


standard_scaler = preprocessing.StandardScaler()
#imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
############################

training_set = pd.read_csv(BANK_TRAINING, names=COLUMNS, skipinitialspace=True, skiprows=1)
training_label_set = deepcopy(training_set[LABEL])
training_set = training_set.drop([LABEL, 'marital', 'job', 'contact'], axis=1)
training_set['education'] = ternary_vectorizing(training_set['education'], ['primary', 'secondary', 'tertiary'])
training_set['education'].replace('unknown',np.nan, inplace=True)
training_set.fillna(training_set.mean(), inplace=True)
#
training_set = one_hot(training_set, CATEGORICAL_COLUMNS_2)
training_label_set = binary_vectorizing(training_label_set, ['no', 'yes'])

##########################
validation_set = pd.read_csv(BANK_VALIDATION, names=COLUMNS, skipinitialspace=True, skiprows=1)
validation_label_set = deepcopy(validation_set[LABEL])

validation_set = validation_set.drop([LABEL, 'marital', 'job', 'contact'], axis=1)
validation_set['education'] = ternary_vectorizing(validation_set['education'], ['primary', 'secondary', 'tertiary'])
validation_set['education'].replace('unknown',np.nan, inplace=True)
validation_set.fillna(training_set.mean(), inplace=True)
#
validation_set = one_hot(validation_set, CATEGORICAL_COLUMNS_2)
validation_label_set = binary_vectorizing(validation_label_set, ['no', 'yes'])
#

fc = [layers.real_valued_column("", dimension=len(training_set.columns))]
classifier_lc = learn.LinearClassifier(feature_columns=fc, model_dir="./lc/", n_classes=2)
classifier_dlc = learn.DNNLinearCombinedClassifier(linear_feature_columns=fc, model_dir="./dlc/", n_classes=2)
classifier_dc = learn.DNNClassifier(feature_columns=fc, n_classes=2, hidden_units=[1000,300,200])

classifier_dc

classifier.fit(x=training_set , y=training_label_set , steps=10)


accuracy_score = classifier.evaluate(x=validation_set,
                                     y=validation_label_set)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

quit()