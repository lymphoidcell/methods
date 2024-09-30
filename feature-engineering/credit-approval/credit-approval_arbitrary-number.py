import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from feature_engine.imputation import ArbitraryNumberImputer

datapath = Path(__file__).parent / "dataset" / "credit_approval_dataset.csv"

data = pd.read_csv(datapath, header=0)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

# which columns have missing values?
print(X_train.dtypes)

missing_values = X_train.isnull().sum()

print(missing_values)

# maximum values
print(X_train[['A2','A3', 'A8', 'A11', 'A14']].max())

# NOTE: although A14 column has 7 missing values, it also has a huge gap of maximum value among other columns.
# that is why, we don't include the A14 column in the next steps.

# imputation using value that is bigger than the maximum values
X_train_rep = X_train.copy()
X_test_rep = X_test.copy()

# replace the missing values with 99
X_train_rep[["A2", "A3", "A8", "A11"]] = X_train_rep[[
    "A2", "A3", "A8", "A11"]].fillna(99)
X_test_rep[["A2", "A3", "A8", "A11"]] = X_test_rep[[
    "A2", "A3", "A8", "A11"]].fillna(99)

# use imputer to replace the missing values
imputer = SimpleImputer(strategy='constant', fill_value=99)

# fit
variables = ["A2", "A3", "A8", "A11"]
imputer.fit(X_train[variables])

# replace the missing values with 99
X_train_rep[variables] = imputer.transform(X_train[variables])
X_test_rep[variables] = imputer.transform(X_test[variables])

# check
print(X_test_rep[variables].isnull().sum())

# set up imputer again to replace missing values with 99 in the variables
imputer = ArbitraryNumberImputer(
    arbitrary_number=99,
    var=["A2", "A3", "A8", "A11"],
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)