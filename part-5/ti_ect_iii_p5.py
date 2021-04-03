# -*- coding: utf-8 -*-
"""
@author: Walter Verwer
@git: https://github.com/walterverwer

"""

# Imports:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

# Plot style:
plt.style.use('ggplot')

#%% Data:

df = pd.read_csv('data_assign_p5.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.reindex(index=df.index[::-1]) # Reverse df
df = df.set_index(df['Date'], drop=True)

# Plot new cases + 7d MA:
plt.plot(df['New Cases'], label='New Cases')
plt.plot(df['7-Day Moving Avg'], label='7-Day Moving Average')
plt.xlabel('Time')
plt.title('United States Covid-19 Cases')
plt.legend()
plt.show()

# Clean data:
# Data has a lot of zeros: drop first number of rows. We start from 2020-02-26:
STARTING_DATE = '2020-02-26' # Change this date to change start of dataset
df = df.loc[STARTING_DATE:]

#%% Analyze time series properties:

# Summary statistics:
print(df.describe().to_latex(float_format="%.2f"))

# Unit-root test:
def adf_test_all_vars():
    output = pd.DataFrame() # store results in a df
    for i in df:
        result = adfuller(df[i], regression='c',autolag='bic', store=False)
        output[i] = pd.Series(result[0:4], 
                           index=['Stat',
                                  'P-value',
                                  'Lags',
                                  'N'])
    return output

# Obtain adf output:
output_adf=adf_test_all_vars()
print(output_adf.to_latex(float_format='%.2f'))

# Cointegration test:
coint_result = coint(df['New Cases'], df['7-Day Moving Avg'], trend = 'nc')
coint_output = pd.Series(coint_result[0:2], 
                           index=['Cointegration t-stat',
                                  'P-value',])
# Obtain coint output:
print(coint_output.to_latex(float_format="%.2f"))

#%% Estimation:


























