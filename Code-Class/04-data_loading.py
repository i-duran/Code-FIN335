# Let's load some data

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/capm.csv")
# data = pd.read_csv("capm.csv")

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


# To see both plots in the same "figure", we can use subplots like this

fig, axs = plt.subplots(2)  # Create a figure and two subplots

# Plot the 'ORACLE' column on the first subplot
data["ORACLE"].plot(ax=axs[0], ylabel="Price", title="ORACLE")

# Plot the 'GE' column on the second subplot
data["GE"].plot(ax=axs[1], ylabel="Price", title="GE")

plt.tight_layout()  # Adjust the layout to prevent overlaps
plt.show()  # Display the plots


# To see plots in the same "figure", in a 2x2 grid, we can use subplots this

fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)  # Create a figure and four subplots

# Plot the 'ORACLE' column on the first subplot
data["ORACLE"].plot(ax=axs[0, 0], ylabel="Price", title="ORACLE")

# Plot the 'GE' column on the second subplot
data["GE"].plot(ax=axs[0, 1], ylabel="Price", title="GE")

# Plot the 'MICROSOFT' column on the third subplot
data["MICROSOFT"].plot(ax=axs[1, 0], ylabel="Price", title="MICROSOFT")

# Plot the 'MICROSOFT' column on the fourth subplot
data["FORD"].plot(ax=axs[1, 1], ylabel="Price", title="FORD")

plt.tight_layout()  # Adjust the layout to prevent overlaps
plt.show()  # Display the plots
