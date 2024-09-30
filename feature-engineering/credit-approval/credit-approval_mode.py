import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import CategoricalImputer

datapath = Path(__file__).parent / "dataset" / "credit_approval_dataset.csv"

data = pd.read_csv(datapath, header=0)

#NOTE: for categorical variables

# split
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.3,
    random_state=0,
)

# categorical variables in a list
categorical_variables = X_train.select_dtypes(
    include="object").columns.to_list()

# stores the modes in a dictionary
modes = X_train[
    categorical_variables].mode().iloc[0].to_dict()

# replace missing values with modes
X_train_rep = X_train.fillna(value=modes)
X_test_rep = X_test.fillna(value=modes)

# replace data for a specific string
imputation_dictionary = {var: "no_data" for var in categorical_variables}

# set up imputer to find the mode per variable
imputer = SimpleImputer(strategy='most_frequent')

# restrict to categorical variables only
ct = ColumnTransformer(
    [("imputer", imputer, categorical_variables)],
    remainder="passthrough",
).set_output(transform="pandas")

# fit the imputer
ct.fit(X_train)

print(ct.named_transformers_.imputer.statistics_)

# replace missing values with the mode
X_train_rep = ct.transform(X_train)
X_test_rep = ct.transform(X_test)

# check
print(X_train_rep.head())

# replace the missing data with the mode
imputer = CategoricalImputer(
    imputation_method="frequent",
    variables=categorical_variables,
)

# fit the imputer
imputer.fit(X_train)

# check
print(imputer.imputer_dict_)

# replace missing values with frequent categories
X_train_rep = imputer.transform(X_train)
X_test_rep = imputer.transform(X_test)
