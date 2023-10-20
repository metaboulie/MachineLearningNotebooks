[toc]



# Machine Learning for Time-Series with Python

## Chapter 1: Introduction to Time-Series with Python

### Characteristics of Time-Series: 

- Long-term movements of the values (**trend**)
- Seasonal variations (**seasonality**)
  1. Seasonality can occur on different time spans, such as daily, weekly, monthly, or yearly.
- Irregular or cyclic components
- **Stationarity** is the property of a time-series not to change its distribution over time as described by its summary statistics. If a time-series is stationary, it means that it has no trend and no deterministic seasonal variability.

## Chaper 2: Time-Series Analysis with Python

### Crucial steps for working with time-series:

- Importing the dataset (**parsing**)
- Data cleaning
- Understanding variables
- Uncovering relationships between variables
- Identifying trend and seasonality
- Preprocessing (including feature engineering)
- Training a machine learning model

### Working with Time-Series in Python

Input:

```python
import pandas as pd

# Import the pandas library
df = pd.DataFrame({"year": [2021, 2022], "month": [3, 4], "day": [24, 25]})

# Create a DataFrame with columns "year", "month", and "day" and corresponding values
ts1 = pd.to_datetime(df)

# Convert the DataFrame to a datetime Series using the pd.to_datetime() function
ts2 = pd.to_datetime("20210324", format="%Y%m%d")

# Convert the string "20210324" to a datetime object using the specified format

s = pd.Series([1, 2, 3, 4, 5]).rolling(3).sum()

# Create a Series with values [1, 2, 3, 4, 5]
# Apply a rolling window of size 3 to calculate the sum
# The rolling() function creates a rolling window object and sum() computes the sum within the window

display(ts1, ts2, s)
# Display the datetime Series ts1, ts2, and the rolling sum Series s
```

Output:

```py
# ts1
0   2021-03-24
1   2022-04-25
dtype: datetime64[ns]
 
# ts2
Timestamp('2021-03-24 00:00:00')

# s
0     NaN
1     NaN
2     6.0
3     9.0
4    12.0
dtype: float64
```

Input:

```python
import numpy as np
import pandas as pd

# Generate a range of dates from "2021-03-24" to "2021-09-01" with a daily frequency
rng = pd.date_range("2021-03-24", "2021-09-01", freq="D")

# Create a Series with random values and the generated dates as the index
ts = pd.Series(np.random.randn(len(rng)), index=rng)

# Display the first few rows of the Series
display(ts.head())

# Display the index (dates) of the first two rows of the Series
display(ts[:2].index)

# Display the subset of the Series that falls within the date range from "2021-03-28" to "2021-03-30"
display(ts["2021-03-28":"2021-03-30"])

# Shift the values of the Series by one position and display the first five rows
display(ts.shift(1)[:5])

# Resample the Series to a monthly frequency and display the result
display(ts.asfreq("M"))
```

Output:

```py
2021-03-24   -0.779174
2021-03-25    1.533478
2021-03-26   -0.270554
2021-03-27    0.135214
2021-03-28   -0.913979
Freq: D, dtype: float64
DatetimeIndex(['2021-03-24', '2021-03-25'], dtype='datetime64[ns]', freq='D')
2021-03-28   -0.913979
2021-03-29    0.397967
2021-03-30    1.191266
Freq: D, dtype: float64
2021-03-24         NaN
2021-03-25   -0.779174
2021-03-26    1.533478
2021-03-27   -0.270554
2021-03-28    0.135214
Freq: D, dtype: float64
2021-03-31   -0.709683
2021-04-30   -2.350385
2021-05-31   -0.275853
2021-06-30    0.678065
2021-07-31   -1.703184
2021-08-31   -1.724842
Freq: M, dtype: float64
```