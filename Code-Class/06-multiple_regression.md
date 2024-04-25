# Multiple Regression Analysis

We will use as an example an extended CAPM model in which we add
additional macroeconomic variables, on top of the traditional variables
described in the model.

Let’s import the libraries first:

``` python
import pandas as pd
import statsmodels.formula.api as smf
```

Let’s load and summarize the data:

``` python
df = pd.read_csv("./data/macro.csv", index_col=0)
print(df.describe())
print(df.head())
```

            MICROSOFT        SANDP         CPI      INDPRO     M1SUPPLY  \
    count  385.000000   385.000000  385.000000  385.000000   385.000000   
    mean    23.301377  1066.036104  181.062234   86.629558  1514.690130   
    std     19.255689   602.397162   41.136587   15.887233   778.509244   
    min      0.100000   230.300000  108.600000   56.500000   624.300000   
    25%      2.890000   459.270000  147.200000   69.480000  1069.300000   
    50%     25.720000  1104.490000  178.800000   93.000000  1191.800000   
    75%     30.860000  1385.590000  218.180000  100.720000  1716.000000   
    max     95.010000  2823.810000  249.550000  106.660000  3684.700000   

               CCREDIT     BMINUSA      USTB3M     USTB10Y  
    count   385.000000  385.000000  385.000000  385.000000  
    mean   1897.814831    0.974623    3.296909    5.075403  
    std     949.565970    0.382047    2.589801    2.173512  
    min     606.800000    0.550000    0.010000    1.500000  
    25%     886.170000    0.720000    0.450000    3.330000  
    50%    1891.830000    0.900000    3.440000    4.910000  
    75%    2620.450000    1.130000    5.290000    6.740000  
    max    3843.420000    3.380000    9.140000    9.520000  
            MICROSOFT   SANDP    CPI  INDPRO  M1SUPPLY  CCREDIT  BMINUSA  USTB3M  \
    Date                                                                           
    Mar-86       0.10  238.90  108.8   56.54     624.3   606.80     1.50    6.76   
    Apr-86       0.11  235.52  108.6   56.57     647.0   614.37     1.40    6.24   
    May-86       0.12  247.35  108.9   56.69     645.7   621.92     1.20    6.33   
    Jun-86       0.11  250.84  109.5   56.50     662.8   627.89     1.21    6.40   
    Jul-86       0.10  236.12  109.5   56.81     673.4   633.61     1.28    6.00   

            USTB10Y  
    Date             
    Mar-86     7.78  
    Apr-86     7.30  
    May-86     7.71  
    Jun-86     7.80  
    Jul-86     7.30  

Calculate the return of Microsoft and S&P

``` python
df["MS_r"] = df["MICROSOFT"].pct_change() * 100
df["MS_r_premium"] = df["MS_r"] - df["USTB3M"]
df["SANDP_r"] = df["SANDP"].pct_change() * 100
df["SANDP_r_premium"] = df["SANDP_r"] - df["USTB3M"]
```

## Model to estimate

Let’s add the inflation rate as covarite or explanatory variable, and so
we will estimate the following equation:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + u$$

where $y$ is `MS_r_premium`, $x_1$ is the `SANDP_r_premium`, and $x_2$
is either `inflation` or `CPI`.

We have the CPI already, so let’s calculate the inflation rate:

``` python
df["inflation"] = df["CPI"].pct_change() * 100
```

Let’s drop the missing values from our data:

``` python
df = df.dropna()
print(df.head())
```

            MICROSOFT   SANDP    CPI  INDPRO  M1SUPPLY  CCREDIT  BMINUSA  USTB3M  \
    Date                                                                           
    Apr-86       0.11  235.52  108.6   56.57     647.0   614.37     1.40    6.24   
    May-86       0.12  247.35  108.9   56.69     645.7   621.92     1.20    6.33   
    Jun-86       0.11  250.84  109.5   56.50     662.8   627.89     1.21    6.40   
    Jul-86       0.10  236.12  109.5   56.81     673.4   633.61     1.28    6.00   
    Aug-86       0.10  252.93  109.7   56.73     678.4   640.51     1.46    5.69   

            USTB10Y       MS_r  MS_r_premium   SANDP_r  SANDP_r_premium  inflation  
    Date                                                                            
    Apr-86     7.30  10.000000      3.760000 -1.414818        -7.654818  -0.183824  
    May-86     7.71   9.090909      2.760909  5.022928        -1.307072   0.276243  
    Jun-86     7.80  -8.333333    -14.733333  1.410956        -4.989044   0.550964  
    Jul-86     7.30  -9.090909    -15.090909 -5.868283       -11.868283   0.000000  
    Aug-86     7.17   0.000000     -5.690000  7.119261         1.429261   0.182648  

Let’s define our `statsmodels` specifications:

``` python
model1 = smf.ols("MS_r_premium ~ SANDP_r_premium", data=df)
model2 = smf.ols("MS_r_premium ~ SANDP_r_premium + inflation", data=df)
model3 = smf.ols("MS_r_premium ~ SANDP_r_premium + CPI", data=df)
```

Let’s estimate the basic model first:

``` python
results1 = model1.fit()
print(results1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           MS_r_premium   R-squared:                       0.306
    Model:                            OLS   Adj. R-squared:                  0.304
    Method:                 Least Squares   F-statistic:                     168.2
    Date:                Thu, 25 Apr 2024   Prob (F-statistic):           3.99e-32
    Time:                        08:28:46   Log-Likelihood:                -1352.3
    No. Observations:                 384   AIC:                             2709.
    Df Residuals:                     382   BIC:                             2717.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           1.7922      0.472      3.795      0.000       0.864       2.721
    SANDP_r_premium     1.1015      0.085     12.969      0.000       0.935       1.268
    ==============================================================================
    Omnibus:                       50.069   Durbin-Watson:                   2.069
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              122.328
    Skew:                           0.651   Prob(JB):                     2.73e-27
    Kurtosis:                       5.439   Cond. No.                         6.31
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Now, let’s estimate the augmented model with `inflation`:

``` python
results2 = model2.fit()
print(results2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           MS_r_premium   R-squared:                       0.311
    Model:                            OLS   Adj. R-squared:                  0.308
    Method:                 Least Squares   F-statistic:                     86.10
    Date:                Thu, 25 Apr 2024   Prob (F-statistic):           1.40e-31
    Time:                        08:28:47   Log-Likelihood:                -1350.8
    No. Observations:                 384   AIC:                             2708.
    Df Residuals:                     381   BIC:                             2719.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           1.3377      0.537      2.490      0.013       0.281       2.394
    SANDP_r_premium     1.1194      0.085     13.121      0.000       0.952       1.287
    inflation           2.3068      1.312      1.759      0.079      -0.272       4.886
    ==============================================================================
    Omnibus:                       45.560   Durbin-Watson:                   2.050
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              105.516
    Skew:                           0.611   Prob(JB):                     1.22e-23
    Kurtosis:                       5.259   Cond. No.                         17.9
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

As a third specification, let’s estimate the augmented model with the
`CPI`, instead of inflation:

``` python
results3 = model3.fit()
print(results3.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           MS_r_premium   R-squared:                       0.332
    Model:                            OLS   Adj. R-squared:                  0.328
    Method:                 Least Squares   F-statistic:                     94.60
    Date:                Thu, 25 Apr 2024   Prob (F-statistic):           4.39e-34
    Time:                        08:28:47   Log-Likelihood:                -1344.9
    No. Observations:                 384   AIC:                             2696.
    Df Residuals:                     381   BIC:                             2708.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           9.8839      2.148      4.602      0.000       5.661      14.106
    SANDP_r_premium     1.2477      0.092     13.617      0.000       1.068       1.428
    CPI                -0.0426      0.011     -3.859      0.000      -0.064      -0.021
    ==============================================================================
    Omnibus:                       32.633   Durbin-Watson:                   2.115
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               86.723
    Skew:                           0.382   Prob(JB):                     1.47e-19
    Kurtosis:                       5.199   Cond. No.                         970.
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

**Analyze the results!**
