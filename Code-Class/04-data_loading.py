# Let's load some data

import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv("./data/capm.csv")
data = pd.read_csv("capm.csv")

print(data.describe())
# print(data.info())
print(data.head())
# print(data.tail())

print("\n\n--Oracle--")
print(data["ORACLE"])
# print("\n\n--GE--")
# print(data["GE"])
# print("\n\n--GE and ORACLE--")
# print(data[["GE", "ORACLE"]])
# print(print("Oracle mean", data["ORACLE"].mean()))

# Graphing

data.set_index("Date", inplace=True)  # Set the 'Date' column as the index

data["ORACLE"].plot(ylabel="Price", title="ORACLE")  # Plot the 'Value' column
plt.show()

data["GE"].plot(ylabel="Price", title="GE")  # Plot the 'Value' column
plt.show()
