import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

datapath = Path(__file__).parent.parent / "dataset" / "example_air_passengers.csv"

data = pd.read_csv(datapath, header=0)

print(data.head())

data.plot()
plt.show()

# add missing data
data.loc[10:11, "y"] = np.nan
data.loc[25:28, "y"] = np.nan
data.loc[40:45, "y"] = np.nan
data.loc[70:94, "y"] = np.nan

data.iloc[10:11]

data.plot()
plt.show()

data.to_csv("air_passengers.csv", index=False)