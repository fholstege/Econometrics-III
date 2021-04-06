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
from stargazer.stargazer import Stargazer

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



# PACF plot - consumption
sm.graphics.tsa.plot_pacf(df['CONS'], lags=12, 
                         title='Aggregate Quarterly Consumption PACF plot')



# ACF plot - income
sm.graphics.tsa.plot_acf(df['INC'], lags=12, 
                         title='Aggregate Quarterly Income ACF plot')


# PACF plot - income
sm.graphics.tsa.plot_pacf(df['INC'], lags=12, 
                         title='Aggregate Quarterly Income PACF plot')




########## Question 2

adf_cons = adfuller(df['CONS'],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)
adf_inc = adfuller(df['INC'],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)

adf_cons
adf_inc

########## Question 3

lagged_cons = lag(df['CONS'],1)
lagged_inc = lag(df['INC'],1)

cons_fd = df['CONS'] - lagged_cons
inc_fd = df['INC'] - lagged_inc

adf_cons_fd = adfuller(cons_fd[~np.isnan(cons_fd)],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)
adf_inc_fd = adfuller(inc_fd[~np.isnan(inc_fd)],maxlag=12 ,regression  ='c',autolag = 'bic', store = True)

adf_cons_fd
adf_inc_fd



########## Question 4

form = 'CONS ~ INC'

# run model with consumption and income
static_model = smf.ols(formula = form, data = df)
static_fitted = static_model.fit(use_t = True)

Stargazer([static_fitted]).render_latex()


est_delta = static_fitted.params[0]
est_lambda = static_fitted.params[1]

Z_est = df['CONS'] - est_delta - est_lambda*df['INC']
Z_est_lagged = lag(Z_est,1)

df_Z = pd.DataFrame({'Z_est':Z_est, 'Z_est_lagged':Z_est_lagged})

# perform augmented dicky fueller on the residuals
residuals_static_model = static_fitted.resid
adfuller(residuals_static_model, maxlag = 12, regression = 'c', autolag='bic', store=True)



########## Question 5

Y_t_fd = df['CONS'] - lag(df['CONS'],1)
X_t_fd = df['INC'] - lag(df['INC'], 1)


def create_df_lags_fd(df,p, q, Y_t_fd, X_t_fd):
    
    df_lags = df.copy(deep = True)
    
    for i in range(1, p+1):
        df_lags['X_t_fd_{0}'.format(i)] = lag(X_t_fd,i)
    
    for i in range(1, q+1):
        df_lags['Y_t_fd_{0}'.format(i)] = lag(Y_t_fd, i)
        
    return df_lags

p = 4 
q = 4

df_ecm = create_df_lags_fd(df_Z, p, q, Y_t_fd, X_t_fd)      
df_ecm['X_t_fd'] = X_t_fd
df_ecm['Y_t_fd'] = Y_t_fd

df_ecm.to_csv("df_ecm.csv")


