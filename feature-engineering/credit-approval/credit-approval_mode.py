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
X_test_rep = X_test.fillna(values=modes)

# replace data for a specific string
imputation_dictionary = {var: "no_data" for var in categorical_variables}
