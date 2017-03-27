import tensorflow.contrib as tf
import tflearn
import pandas as pd

BANK_TRAINING = "bank-training_new.csv"
BANK_TESTING = "bank-testing_new.csv"

COLUMNS = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration",
           "campaign","pdays","previous","poutcome","y"]

training_set = pd.read_csv(BANK_TRAINING, names=COLUMNS, skipinitialspace=True, skiprows=1)

test_set = pd.read_csv(BANK_TESTING, names=COLUMNS, skipinitialspace=True, skiprows=1)

CATEGORICAL_COLUMNS = ["job","marital","education","default","housing","loan","contact","day","month","poutcome"]
CONTINUOUS_COLUMNS = ["age","balance","duration","campaign","pdays","previous"]
LABEL = "y"

optimizer = tflearn.optimizers.SGD (learning_rate=0.01)
estimator = tflearn.layers.regression(training_set, optimizer=optimizer, n_classes=2, learning_rate=0.01 )

model = tflearn.models.dnn.DNN(estimator, tensorboard_verbose=3)

