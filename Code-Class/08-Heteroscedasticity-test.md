# Testing for Heteroscedasticity

## Data set

We will be using the same modified data set of the previous section on
wages.

Remember that this data has the following variables:

- `wage`: average hourly earnings
- `educ`: years of education
- `exper`: years potential experience
- `tenure`: years with current employer
- `married`: =1 if married
- `numdep`: number of dependents
- `gender`: male or female
- `skin`: color of the skin (white or nonwhite)

## Base regression

``` python
import pandas as pd

data = pd.read_csv("./data/wage.csv")
data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | wage | educ | exper | tenure | married | numdep | gender | skin  |
|-----|------|------|-------|--------|---------|--------|--------|-------|
| 0   | 3.10 | 11   | 2     | 0      | 0       | 2      | female | white |
| 1   | 3.24 | 12   | 22    | 2      | 1       | 3      | female | white |
| 2   | 3.00 | 11   | 2     | 0      | 0       | 2      | male   | white |
| 3   | 6.00 | 8    | 44    | 28     | 1       | 0      | male   | white |
| 4   | 5.30 | 12   | 7     | 2      | 1       | 1      | male   | white |

</div>

As before, we first want to construct the dummy variables for `gender`
and `skin`:

``` python
data = pd.get_dummies(data, columns=["gender", "skin"], drop_first=False, dtype=int)
data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | wage | educ | exper | tenure | married | numdep | gender_female | gender_male | skin_nonwhite | skin_white |
|-----|------|------|-------|--------|---------|--------|---------------|-------------|---------------|------------|
| 0   | 3.10 | 11   | 2     | 0      | 0       | 2      | 1             | 0           | 0             | 1          |
| 1   | 3.24 | 12   | 22    | 2      | 1       | 3      | 1             | 0           | 0             | 1          |
| 2   | 3.00 | 11   | 2     | 0      | 0       | 2      | 0             | 1           | 0             | 1          |
| 3   | 6.00 | 8    | 44    | 28     | 1       | 0      | 0             | 1           | 0             | 1          |
| 4   | 5.30 | 12   | 7     | 2      | 1       | 1      | 0             | 1           | 0             | 1          |

</div>

Now, let’s run a regression:

``` python
import statsmodels.formula.api as smf

model = smf.ols("wage ~ educ + exper + tenure + married + numdep + gender_female + skin_nonwhite", data=data)
results = model.fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                   wage   R-squared:                       0.370
    Model:                            OLS   Adj. R-squared:                  0.362
    Method:                 Least Squares   F-statistic:                     43.54
    Date:                Sun, 05 May 2024   Prob (F-statistic):           2.53e-48
    Time:                        08:48:17   Log-Likelihood:                -1311.4
    No. Observations:                 526   AIC:                             2639.
    Df Residuals:                     518   BIC:                             2673.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept        -1.9799      0.783     -2.530      0.012      -3.517      -0.442
    educ              0.5746      0.052     11.034      0.000       0.472       0.677
    exper             0.0219      0.012      1.789      0.074      -0.002       0.046
    tenure            0.1383      0.021      6.541      0.000       0.097       0.180
    married           0.4597      0.295      1.556      0.120      -0.121       1.040
    numdep            0.1465      0.109      1.345      0.179      -0.067       0.360
    gender_female    -1.7594      0.267     -6.592      0.000      -2.284      -1.235
    skin_nonwhite    -0.1100      0.427     -0.257      0.797      -0.950       0.730
    ==============================================================================
    Omnibus:                      186.895   Durbin-Watson:                   1.788
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              711.513
    Skew:                           1.604   Prob(JB):                    3.14e-155
    Kurtosis:                       7.709   Cond. No.                         153.
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Performing the White test

Now, let’s perform the White test, which is also available in the
`statsmodels` library. Note the following:

- We use the same independent variables of the original model, obtained
  with `results.model.exog`.
- The estimated residual are obtained from the previous estimation with
  `results.resid`.
- `statsmodels` automatically include squares and interaction terms to
  run the auxiliary regression.

``` python
import statsmodels.stats.diagnostic as diag

white_test = diag.het_white(results.resid, results.model.exog)

# The output consists of the test statistic and the associated p-value
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Statistic p-value']
white_test_summary = dict(zip(labels, white_test))

for key, value in white_test_summary.items():
    print(f"{key}: {value}")
```

    Test Statistic: 76.30870031228504
    Test Statistic p-value: 1.7487062379433416e-05
    F-Statistic: 2.6143065587494148
    F-Statistic p-value: 6.246658985394247e-06

**Analyze the results!**

## Run the auxiliary regression ourselves

If we want to see which terms are significantly affecting the squared
error term, then we need to run the regression of the White test
ourselves. For example:

``` python
import numpy as np
import statsmodels.api as sm

# Calculate the squared residuals
data['resid_sq'] = results.resid ** 2

# We need to create the matrix of explanatory variables for the auxiliary regression:
# For simplicity, let's just include the original independent variables and their squares

X = results.model.exog  # original independent variables matrix
X_names = results.model.exog_names # names of the independent variables
X_squared = np.square(X) # calculate the squares of the independent variables
data_aux = pd.DataFrame(np.column_stack([X, X_squared]), columns=X_names + [name + '^2' for name in X_names])

# Fit the auxiliary regression
aux_model = sm.OLS(data['resid_sq'], sm.add_constant(data_aux))
aux_results = aux_model.fit()

# Display the auxiliary regression results
print("\nAuxiliary Regression Summary (for White's Test):\n")
print(aux_results.summary())
```


    Auxiliary Regression Summary (for White's Test):

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               resid_sq   R-squared:                       0.119
    Model:                            OLS   Adj. R-squared:                  0.101
    Method:                 Least Squares   F-statistic:                     6.337
    Date:                Sun, 05 May 2024   Prob (F-statistic):           7.20e-10
    Time:                        08:48:17   Log-Likelihood:                -2343.6
    No. Observations:                 526   AIC:                             4711.
    Df Residuals:                     514   BIC:                             4762.
    Df Model:                          11                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           8.4392      5.104      1.654      0.099      -1.587      18.466
    educ               -4.2397      1.598     -2.653      0.008      -7.380      -1.100
    exper               0.7940      0.289      2.749      0.006       0.227       1.361
    tenure              0.7152      0.362      1.975      0.049       0.004       1.427
    married            -1.3824      1.101     -1.256      0.210      -3.545       0.780
    numdep              2.0890      1.975      1.058      0.291      -1.790       5.968
    gender_female      -1.5583      0.966     -1.612      0.107      -3.457       0.340
    skin_nonwhite      -1.0385      1.533     -0.677      0.499      -4.051       1.974
    Intercept^2         8.4392      5.104      1.654      0.099      -1.587      18.466
    educ^2              0.2346      0.066      3.575      0.000       0.106       0.364
    exper^2            -0.0163      0.006     -2.605      0.009      -0.029      -0.004
    tenure^2           -0.0070      0.012     -0.563      0.574      -0.031       0.017
    married^2          -1.3824      1.101     -1.256      0.210      -3.545       0.780
    numdep^2           -0.5881      0.493     -1.193      0.233      -1.556       0.380
    gender_female^2    -1.5583      0.966     -1.612      0.107      -3.457       0.340
    skin_nonwhite^2    -1.0385      1.533     -0.677      0.499      -4.051       1.974
    ==============================================================================
    Omnibus:                      616.162   Durbin-Watson:                   1.879
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            38143.071
    Skew:                           5.683   Prob(JB):                         0.00
    Kurtosis:                      43.139   Cond. No.                     1.15e+19
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.49e-30. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.

**Analyze the results!**

## Run a WLS estimation

To try fixing the problem of heteroscedasticity, lets run a WLS
estimation:

``` python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the dataset and construct dummies
data = pd.read_csv("data/wage.csv")
data = pd.get_dummies(data, columns=["gender", "skin"], drop_first=False, dtype=int)

# Initial OLS model
model_ols = smf.ols('wage ~ educ + exper + tenure + married + numdep + gender_female + skin_nonwhite', data=data)
results_ols = model_ols.fit()

# Estimate residuals
residuals = results_ols.resid

# Assuming variance is proportional to the square of predictors (or another model of your choice)
# For example, using squares of fitted values
data['weights'] = results_ols.fittedvalues ** 2

# Transform weights to stabilize variance (Inverse or as per specific heteroscedasticity form)
weights = 1 / data['weights']

# Feasible GLS model using estimated weights
model_fglsg = smf.wls('wage ~ educ + exper + tenure + married + numdep + gender_female + skin_nonwhite', 
                      data=data, weights=weights)
results_fglsg = model_fglsg.fit()

# Output the results
print("\nFeasible GLS model summary:\n")
print(results_fglsg.summary())
```


    Feasible GLS model summary:

                                WLS Regression Results                            
    ==============================================================================
    Dep. Variable:                   wage   R-squared:                       0.308
    Model:                            WLS   Adj. R-squared:                  0.299
    Method:                 Least Squares   F-statistic:                     32.97
    Date:                Sun, 05 May 2024   Prob (F-statistic):           6.37e-38
    Time:                        08:48:17   Log-Likelihood:                -1241.3
    No. Observations:                 526   AIC:                             2499.
    Df Residuals:                     518   BIC:                             2533.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept         1.9993      0.476      4.202      0.000       1.065       2.934
    educ              0.2491      0.032      7.670      0.000       0.185       0.313
    exper             0.0048      0.008      0.594      0.553      -0.011       0.021
    tenure            0.1144      0.021      5.364      0.000       0.072       0.156
    married           0.6196      0.201      3.085      0.002       0.225       1.014
    numdep            0.0103      0.063      0.163      0.870      -0.113       0.134
    gender_female    -1.1811      0.210     -5.617      0.000      -1.594      -0.768
    skin_nonwhite     0.2219      0.253      0.876      0.382      -0.276       0.720
    ==============================================================================
    Omnibus:                      160.086   Durbin-Watson:                   1.717
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              417.268
    Skew:                           1.520   Prob(JB):                     2.46e-91
    Kurtosis:                       6.130   Cond. No.                         127.
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

``` python
import statsmodels.stats.diagnostic as diag

white_test = diag.het_white(results_fglsg.resid, results_fglsg.model.exog)

# The output consists of the test statistic and the associated p-value
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Statistic p-value']
white_test_summary = dict(zip(labels, white_test))

for key, value in white_test_summary.items():
    print(f"{key}: {value}")
```

    Test Statistic: 90.66730111962585
    Test Statistic p-value: 1.6207543134390787e-07
    F-Statistic: 3.208679502979574
    F-Statistic p-value: 2.5463611583492202e-08
