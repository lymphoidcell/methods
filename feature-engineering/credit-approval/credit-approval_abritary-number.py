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

