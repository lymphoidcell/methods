import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engine.imputation import(AddMissingIndicator, CategoricalImputer, MeanMedianImputer)

datapath = Path(__file__).parent / "dataset" / "credit_approval_dataset.csv"

data = pd.read_csv(datapath, header=0)

# split
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

# list of variable names
variable_names = ["A1", "A3", "A4", "A5", "A6", "A7", "A8"]

# missing indicators, list them
indicators = [f"{var}_na" for var in variable_names]

print(indicators)

# copy df
X_train_rep = X_train.copy()
X_test_rep = X_test.copy()

# add missing indicators
X_train_rep[indicators] = X_train[
    variable_names].isna().astype(int)
X_test_rep[indicators] = X_test[
    variable_names].isna().astype(int)

# check
print(X_train_rep.head())

# add binary indicators to variables with missing data
imputer = AddMissingIndicator(
    variables=None, missing_only=True)

imputer.fit(X_train)

X_train_rep = imputer.transform(X_train)
X_test_rep = imputer.transform(X_test)

# pipeline, explanation in the markdown soon, TBA!
pipe = Pipeline([
    ("indicators",
        AddMissingIndicator(missing_only=True)),
    ("categorical", CategoricalImputer(
        imputation_method="frequent")),
    ("numerical", MeanMedianImputer()),
])

# add the indicators
X_train_rep = pipe.fit_transform(X_train)
X_test_rep = pipe.transform(X_test)

# check: make sure we no longer have missing data
print(X_train_rep.isnull().sum())
print(X_train_rep.head())

# list the variables
num = X_train.select_dtypes(
    exclude="object").columns.to_list()
cat = X_train.select_dtypes(
    include="object").columns.to_list()

# mean and mode
pipe = ColumnTransformer([
    ("num_imputer", SimpleImputer(
        strategy="mean",
        add_indicator=True),
    num),
    ("cat_imputer", SimpleImputer(
        strategy="most_frequent",
        add_indicator=True),
    cat),
]).set_output(transform="pandas")

# imputation
X_train_rep = pipe.fit_transform(X_train)
X_test_rep = pipe.transform(X_test)