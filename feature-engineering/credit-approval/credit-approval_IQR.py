import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from feature_engine.imputation import EndTailImputer

datapath = Path(__file__).parent / "dataset" / "credit_approval_dataset.csv"

data = pd.read_csv(datapath, header=0)

# numerical variables
numeric_variables = [
    var for var in data.select_dtypes(
        exclude="object").columns.to_list()
    if var !="target"
]

# split
X_train, X_test, y_train, y_test = train_test_split(
    data[numeric_variables],
    data["target"],
    test_size=0.3,
    random_state=0,
)

# IQR
IQR = X_train.quantile(0.75) - X_train.quantile(0.25)

print(IQR)

# create a dictionary
imputation_dict = (
    X_train.quantile(0.75) + 1.5 * IQR).to_dict()

# replace missing data
X_train_rep = X_train.fillna(value=imputation_dict)
X_test_rep = X_test.fillna(value=imputation_dict)


