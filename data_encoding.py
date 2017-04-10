from copy import deepcopy
import pandas as pd

def one_hot(dtf, cols):
    df = deepcopy(dtf)
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(cols, axis=1)
    return df

def main():
    BANK_VALIDATION = "bank-crossvalidation_new.csv"

    COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month",
               "duration",
               "campaign", "pdays", "previous", "poutcome", "y"]
    CATEGORICAL_COLUMNS = ["job", "marital", "education", "default", "housing", "loan", "contact", "day", "month",
                           "poutcome", "y"]

    validation_set = pd.read_csv(BANK_VALIDATION, names=COLUMNS, skipinitialspace=True, skiprows=1)

    # Vectorize the categorical columns
    df = one_hot(validation_set, cols=CATEGORICAL_COLUMNS)

    print(df.head())


if __name__ == '__main__':
    main()