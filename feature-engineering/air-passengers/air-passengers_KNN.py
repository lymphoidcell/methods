import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# time series data
datapath = Path(__file__).parent / "dataset" / "air_passengers.csv"

data = pd.read_csv(datapath,header=0,
                   parse_dates=["ds"],
                   index_col=["ds"],
                   )

print(data.head())