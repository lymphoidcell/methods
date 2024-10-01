import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import (
    enable_iterative_imputer
)
from sklearn.impute import (
    IterativeImputer,
    SimpleImputer
)

# time series data
datapath = Path(__file__).parent / "dataset" / "air_passengers.csv"

data = pd.read_csv(datapath,header=0,
                   parse_dates=["ds"],
                   index_col=["ds"],
                   )

print(data.head())