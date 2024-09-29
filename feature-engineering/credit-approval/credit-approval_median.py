import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer

datapath = Path(__file__).parent / "dataset" / "credit_approval_dataset.csv"

data = pd.read_csv(datapath, header=0)

# NOTE: for numerical variables only, and the same ways can be applied if we want to use 'mean' instead of 'median'

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

# create a list with the numerical variables by excluding variables of type object
numeric_variables = X_train.select_dtypes(
    exclude="object").columns.to_list()

# see the variables' median values in a dictionary
median_values = X_train[
    numeric_variables].median().to_dict()

print(median_values)

# replace missing data with the median (testing)
X_train_rep = X_train.fillna(value=median_values)
X_test_rep = X_test.fillna(value=median_values)

print(X_train_rep[numeric_variables].isnull().sum())

# set up imputer to replace missing data with the median
imputer = SimpleImputer(strategy="median")

# restrict the imputer to the numerical variables only
ct = ColumnTransformer(
    [("imputer", imputer, numeric_variables)],
    remainder = "passthrough",
    force_int_remainder_cols=False,
).set_output(transform="pandas")

ct.fit(X_train)

# check
print(ct.named_transformers_.imputer.statistics_)

# replace missing values with the median
X_train_rep = ct.transform(X_train)
X_test_rep = ct.transform(X_test)

print(X_train_rep.head())

# imputer again
imputer = MeanMedianImputer(
    imputation_method="median",
    variables=numeric_variables,
)

# fit the imputer again to learn the median values
imputer.fit(X_train)

print(imputer.imputer_dict_)

# replace the missing values with the median
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)