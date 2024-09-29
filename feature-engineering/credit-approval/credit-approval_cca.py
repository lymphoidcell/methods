import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import DropMissingData

datapath = Path(__file__).parent / "dataset" / "credit_approval_dataset.csv"

data = pd.read_csv(datapath, header=0)
print(data.head())

# preparing the data to train machine learning models
# split the data into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.30,
    random_state=42
)

# bar plot to showcase the proportion of missing data per variable in the training and test sets
fig, axes = plt.subplots(
    2, 1, figsize=(15, 10), squeeze=False)

X_train.isnull().mean().plot(
    kind='bar', color='grey', ax=axes[0, 0], title="train")

X_test.isnull().mean().plot(
    kind='bar', color='black', ax=axes[1, 0], title="test")

axes[0, 0].set_ylabel('Fraction of NAN')
axes[1, 0].set_ylabel('Fraction of NAN')

plt.show()

# remove observations if these two have missing values in any variables
train_cca = X_train.dropna()
test_cca = X_test.dropna()

# print to see the comparison between the size of the original and complete case dataset
print(f"Total observations: {len(X_train)}")
print(f"Observations without NAN: {len(train_cca)}")

# align the target variables
y_train_cca = y_train.loc[train_cca.index]
y_test_cca = y_test.loc[test_cca.index]

# set up the imputer to automatically find the variables with missing data
cca = DropMissingData(variables=None, missing_only=True)

# fit the transformer so that it finds variables with missing data
cca.fit(X_train)

# return variables with missing data
print(cca.variables_)

# remove rows with missing data in the training and test sets
train_cca = cca.transform(X_train)
test_cca = cca.transform(X_test)

# adjust the target after removing missing data from the training set
train_c, y_train_c = cca.transform_x_y( X_train, y_train)
test_c, y_test_c = cca.transform_x_y(X_test, y_test)