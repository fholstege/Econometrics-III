# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:45:36 2021

@author: flori
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf
import scipy.stats as st
s

# for augmented dickey fueller
from statsmodels.tsa.stattools import adfuller


## Helper function to create lags
def lag(x, n):
    if n == 0:
        return x
    if isinstance(x, pd.Series):
        return x.shift(n)
    else:
        x = pd.Series(x)
        return x.shift(n)
    x = x.copy()
    x[n:] = x[0:-n]
    x[:n] = np.nan
    return x


df = pd.read_csv("data_assign_p4.csv")
df['obs'] = pd.to_datetime(df['obs'])

########## Question 1


plt.plot(df['obs'],df['CONS'], label = 'Consumption', color = 'red')
plt.plot(df['obs'],df['INC'], label = "Income", color = 'blue')
plt.ylabel("Aggregate Euros Per Quarter")
plt.legend(loc="upper left")



# ACF plot - consumption
sm.graphics.tsa.plot_acf(df['CONS'], lags=12, 
                         title='Aggregate Quarterly Consumption ACF plot')


# ACF plot - income
sm.graphics.tsa.plot_acf(df['INC'], lags=12, 
                         title='Aggregate Quarterly Income ACF plot')


########## Question 2

adf_cons = adfuller(df['CONS'],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)
adf_inc = adfuller(df['INC'],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)


########## Question 3

lagged_cons = lag(df['CONS'],1)
lagged_inc = lag(df['INC'],1)

cons_fd = df['CONS'] - lagged_cons
inc_fd = df['INC'] - lagged_inc

adf_cons_fd = adfuller(cons_fd[~np.isnan(cons_fd)],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)
adf_inc_fd = adfuller(inc_fd[~np.isnan(inc_fd)],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)


########## Question 4

form = 'CONS ~ INC'

smf.ols(formula = full_form, data = df)