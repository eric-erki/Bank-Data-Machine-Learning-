import tensorflow.contrib as tf
import numpy as np
import pandas as pd

BANK_TRAINING = "bank.csv"
BANK_TEST = "bank_predict.csv"

COLUMNS = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration",
           "campaign","pdays","previous","poutcome","y"]

training_set = pd.read_csv(BANK_TRAINING, names=COLUMNS, skipinitialspace=True, skiprows=1)

test_set = pd.read_csv(BANK_TEST, names=COLUMNS, skipinitialspace=True, skiprows=1)

CATEGORICAL_COLUMNS = ["job","marital","education","default","housing","loan","contact","day","month","poutcome"]
CONTINUOUS_COLUMNS = ["age","balance","duration","campaign","pdays","previous"]
LABEL = "y"

