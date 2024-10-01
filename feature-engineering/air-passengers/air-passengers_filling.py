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

# missing data
missing = data.isnull().mean() * 100
print(missing.astype(str) + '%')

# data gap
ax = data.plot(marker=".", figsize=[16, 8], legend=None)
ax.set_title("Air Passengers")
ax.set_ylabel("Number of Passengers")
ax.set_xlabel("Time")
plt.show()

# impute missing data using forward fill
imputed = data.ffill()

# check
print(imputed.isnull().sum())

# imputation result plot
ax = imputed.plot(
    linestyle="-", marker=".", figsize=[16, 8])
imputed[data.isnull()].plot(
    ax=ax, legend=None, marker=".", color="r")
ax.set_title("Air Passengers")
ax.set_ylabel("Number of Passengers")
ax.set_xlabel("Time")
plt.show()

# impute the missing data using backward fill
imputed_back = data.bfill()

# imputation result plot
ax = imputed_back.plot(
    linestyle="-", marker=".", figsize=[16, 8])
imputed_back[data.isnull()].plot(
    ax=ax, legend=None, marker=".", color="r")
ax.set_title("Air Passengers")
ax.set_ylabel("Number of Passengers")
ax.set_xlabel("Time")
plt.show()