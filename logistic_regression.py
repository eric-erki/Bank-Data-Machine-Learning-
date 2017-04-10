from copy import deepcopy
import tflearn
import pandas as pd

BANK_TRAINING = "bank-training_new.csv"
BANK_VALIDATION = "bank-crossvalidation_new.csv"

COLUMNS = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration",
           "campaign","pdays","previous","poutcome","y"]

training_set = pd.read_csv(BANK_TRAINING, names=COLUMNS, skipinitialspace=True, skiprows=1)
training_label_set = deepcopy(training_set['y'])
del training_set['y']

validation_set = pd.read_csv(BANK_VALIDATION, names=COLUMNS, skipinitialspace=True, skiprows=1)
validation_label_set = deepcopy(validation_set['y'])
del validation_set['y']

CATEGORICAL_COLUMNS = ["job","marital","education","default","housing","loan","contact","day","month","poutcome"]
CONTINUOUS_COLUMNS = ["age","balance","duration","campaign","pdays","previous"]
LABEL = "y"

print(validation_set.shape)
print(validation_label_set.shape)
print(validation_set.head())
print("\n\n")
print(validation_label_set.head())

#quit()


optimizer = tflearn.optimizers.SGD (learning_rate=0.01)
shape = tflearn.layers.core.input_data (shape=training_set.shape)
estimator = tflearn.layers.regression(shape, optimizer=optimizer, n_classes=2, learning_rate=0.01 )

model = tflearn.models.dnn.DNN(estimator, tensorboard_verbose=3)

model.fit(training_set, training_label_set, n_epoch=10, batch_size=16, show_metric=True)

predicts = model.predict(validation_set)

total_right,total = 0,0
for p,l in predicts, validation_label_set:
    if p[1]==l :
        total_right+=1
    total +=1

print("Accuracy: ", total_right/total)