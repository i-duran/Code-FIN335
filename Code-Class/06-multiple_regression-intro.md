## Multiple Regression in Python

We want to be able to estimate the parameters $\beta_j$ in the following equation:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k + u $$

In order to accomplish this, we are going to use the `statsmodels` python library, in addition `pandas`, which we have been using already to load and analyze data.

```python
import pandas as pd
import statsmodels.formula.api as smf
```

From the `statsmodels` library, we want to use the `formula.api` module in particular, which allows us to specify the equation we want to estimate using simple notation.

For example, in order to estimate an equation with one single variable, we will use the following:

```python
model = smf.ols("y ~ x", data=df)
```

Where $y$ and $x$ would be replaced by the corresponding name of the variables in the data set.

Similarly, to estimate an equation with multiple variables, we will use the following specification:

```python
model = smf.ols("y ~ x1 + x2 + x3", data=df)
```

In this case, we are including three explanatory variables, and we are using an OLS estimations, which is what `smf.ols` mean.

## Useful variable transformations

**Percentage changes**

```python
data["FORD_return"] = data["FORD"].pct_change()
data["SANDP_return"] = data["SANDP"].pct_change() * 100
```

**Log transformation**
In this case, we need to use the `numpy` library as well

```python
import numpy as np
```

Then we can use it like this:

```python
data["FORD_log"] = data["FORD"].apply(np.log)
data["SANDP_log"] = data["SANDP"].apply(np.log)
```


**Operations between columns or variables**

```python
data["New_Variable"] = data["x1"] - data["x2"]
data["New_Variable"] = data["x1"] * data["x2"]
data["New_Variable"] = data["x1"] * 100
```
