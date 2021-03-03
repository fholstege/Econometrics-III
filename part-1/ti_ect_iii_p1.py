# Econometrics III Part I

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AR
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import math

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





