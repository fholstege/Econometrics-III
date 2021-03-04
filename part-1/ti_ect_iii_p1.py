# Econometrics III Part I

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR


# Data
df = pd.read_csv('data_assign_p1.csv')
df['obs'] = pd.to_datetime(df['obs'])
df = df.set_index('obs')

## Question 1:
# General plot:
plt.plot(df['GDP_QGR'])
plt.xlabel('Time')
plt.ylabel('GDP growth rate')
plt.show()

# ACF plot:
sm.graphics.tsa.plot_acf(df['GDP_QGR'], lags=12)

# PACF plot:
sm.graphics.tsa.plot_pacf(df['GDP_QGR'], lags=12)

## Question 2:
# AR(p):
# model = AR(df['GDP_QGR'])
# lags = model.select_order(maxlag=15, ic='aic')
# lags = for loop use G2S!!!!
# model_fitted = model.fit(lags)
# params = model_fitted.params
# print('Parameter information: \n', params)
# print('Lag length =', lags)


