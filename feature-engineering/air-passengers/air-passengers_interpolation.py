import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from Tools.demo.sortvisu import interpolate

# time series data
datapath = Path(__file__).parent / "dataset" / "air_passengers.csv"

data = pd.read_csv(datapath,header=0,
                   parse_dates=["ds"],
                   index_col=["ds"],
                   )

print(data.head())

# impute using linear interpolation
imputed_lin = data.interpolate(method='linear')

# check
print(imputed_lin.isnull().sum)

# plot
ax = imputed_lin.plot(
    linestyle="-", marker=".", figsize=[16, 8])
imputed_lin[data.isnull()].plot(
    ax=ax, legend=None, marker=".", color="r")
ax.set_title("Air Passengers")
ax.set_ylabel("Number of Passengers")
ax.set_xlabel("Time")
plt.show()

# impute using spline interpolation
imputed_sp = data.interpolate(method='spline', order=2)

# plot
ax = imputed_sp.plot(
    linestyle="-", marker=".", figsize=[16, 8])
imputed_sp[data.isnull()].plot(
    ax=ax, legend=None, marker=".", color="r")
ax.set_title("Air Passengers")
ax.set_ylabel("Number of Passengers")
ax.set_xlabel("Time")
plt.show()