from copy import deepcopy
import sklearn.linear_model as lm
import sklearn.neighbors as knn
import pandas as pd
from data_encoding import one_hot, label_vec
from sklearn import svm

BANK_TRAINING = "bank-training_new.csv"
BANK_VALIDATION = "bank-crossvalidation_new.csv"
LOG_FILE = "log.txt"

COLUMNS = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration",
           "campaign","pdays","previous","poutcome","y"]
CATEGORICAL_COLUMNS = ["job","marital","education","default","housing","loan","contact","day","month","poutcome"]
CONTINUOUS_COLUMNS = ["age","balance","duration","campaign","pdays","previous"]
LABEL = "y"

training_set = pd.read_csv(BANK_TRAINING, names=COLUMNS, skipinitialspace=True, skiprows=1)
training_label_set = deepcopy(training_set[LABEL])
del training_set[LABEL]
training_set = one_hot(training_set, CATEGORICAL_COLUMNS)
training_label_set = label_vec(training_label_set)

validation_set = pd.read_csv(BANK_VALIDATION, names=COLUMNS, skipinitialspace=True, skiprows=1)
validation_label_set = deepcopy(validation_set[LABEL])
del validation_set[LABEL]
validation_set = one_hot(validation_set, CATEGORICAL_COLUMNS)
validation_label_set = label_vec(validation_label_set)

models = []
models.append(lm.LogisticRegression())
models.append(knn.KNeighborsClassifier())
models.append(svm.SVC(kernel='linear'))
models.append(svm.SVC(kernel='poly', degree='3'))
models.append(svm.SVC(kernel='poly', degree='16'))
models.append(svm.SVC(kernel='rbf', degree='16'))

f = open(LOG_FILE, 'w+')

for model in models:
    model.fit(training_set, training_label_set)
    score = str( model.score(validation_set, validation_label_set) )
    describe = str(model) + score + "\n\n**********\n\n"
    f.write(describe)
    print(describe)

f.close()