import random
import numpy as np
import pandas as pd
from pathlib import Path

# load the dataset
datapath = Path(__file__).parent / "dataset" / "crx.data"
credit_approval = pd.read_csv(datapath, header=None)

# check the data
print(credit_approval.head())

# add names for each column according to the index description
variable_names = [f"A{s}" for s in range(1, 17)]

credit_approval.columns = variable_names

credit_approval = credit_approval.replace("?", np.nan)

credit_approval["A2"] = credit_approval["A2"].astype("float")
credit_approval["A14"] = credit_approval["A14"].astype("float")

credit_approval["A16"] = credit_approval["A16"].map({"+": 1, "-": 0})
credit_approval.rename(columns={"A16": "target"}, inplace=True)

print(credit_approval.head())

# check missing values in the dataset
print(credit_approval.isnull().sum())

# add missing values (random positions)
random.seed(9001)

chosen_variables = ["A3", "A8", "A9", "A10"]

# random position indexes:
for i in range(4):
    values = list(set([random.randint(i, len(credit_approval)) for p in range (0, 100)]))

    # add missing data:
    credit_approval.loc[values, chosen_variables[i]] = np.nan

# check the missing values again
print(credit_approval.isnull().sum())

# save the dataset
output_path = '../dataset/credit_approval_dataset.csv'

credit_approval.to_csv(output_path, index=False)

print(f"File saved successfully at {output_path}")

# check the proportion of missing data in the saved csv file
datapath_1 = Path(__file__).parent / "dataset" / "credit_approval_dataset.csv"

data = pd.read_csv(datapath_1, header=None)

# count total missing values
total_missing = data.isnull().sum().sum()

total_values = data.size

missing_proportion = total_missing / total_values

# print them
print(f"Total missing values: {total_missing}")
print(f"Total values in the dataset: {total_values}")
print(f"Proportion of missing values: {missing_proportion:.2%}")