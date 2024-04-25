# Linear or (Simple and Multiple) Regression

# On top of pandas, we will be using the statsmodels library for esimating
# linear regression models

import pandas as pd
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

# Load and summary of the data

df = pd.read_csv("./data/capm.csv", index_col=0)


df["FORD_r"] = df["FORD"].pct_change() * 100
df["FORD_r-Rf"] = df["FORD_r"] - df["USTB3M"]
df["SANDP_r"] = df["SANDP"].pct_change() * 100
df["SANDP_r-Rf"] = df["SANDP_r"] - df["USTB3M"]
print(df.head())
# print(df.describe())

df.dropna()

# ---Simple regression---

print("\n\n--- Simple Regression ---")

model = smf.ols("FORD_r ~ SANDP_r", data=df)
results = model.fit()
print(results.summary())

# Print just the estimated coefficients
print("\n\nCoeficients:")
print(results.params)

# Analyze Predicted values
ford_observed = df["FORD"]
# ford_pred = results.predict()
ford_pred = results.fittedvalues

prediction_df = pd.DataFrame(
    {
        "FORD_OBSERVED": ford_observed,
        "FORD_PREDICTED": ford_pred,
        "ERROR": ford_observed - ford_pred,
        "RESID": results.resid,
        "RESID-SQ": results.resid**2,
    }
)

print(prediction_df)
print("\nMean of the residuals:", results.resid.sum() / model.nobs)
print("Mean squared error of the residuals:", (results.resid**2).sum() / (model.nobs - 2))
print("Mean squared error of the residuals (statsmodels):", results.mse_resid)


# Scatter plot with "Fitted line"
# Create a scatter plot of the observed data
sns.scatterplot(x=df['SANDP'], y=df['FORD'])

# Generate and plot the fitted line
sns.lineplot(x=df['SANDP'], y=results.fittedvalues, color='red')

# Show the plot
plt.show()

# Trend Model
# A model that regress a variable against "time"

df["index"] = range(1, len(df) + 1)
trend_model = smf.ols("FORD ~ index", data=df)
results_trend = trend_model.fit()
print(results_trend.summary())
# Create a scatter plot of the observed data
sns.scatterplot(x=df['index'], y=df['FORD'])
# Generate and plot the fitted line
sns.lineplot(x=df['index'], y=results_trend.fittedvalues, color='red')
# Show the plot
plt.show()
