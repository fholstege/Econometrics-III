# Econometrics III Part I

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import AutoReg


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
sm.graphics.tsa.plot_acf(df['GDP_QGR'], lags=12, 
                         title='GDP Quarterly Growth rate ACF plot')

# PACF plot:
sm.graphics.tsa.plot_pacf(df['GDP_QGR'], lags=12,
                          title='GDP Quarterly Growth rate PACF plot')

## Question 2:
# AR(p):
def ar_g2s(maxlags, alpha):
# Estimate AR(p) model for different lags, select based on g2p
# maxlags = starting lag value
# alpha = confidence level for g2p method
    
    for i in reversed(range(1,maxlags+1)):
        model = AutoReg(df['GDP_QGR'], lags=i, trend='c', old_names=False)
        model_fitted = model.fit(use_t=True)
        
        if model_fitted.pvalues[-1]<=0.05:
            result = model_fitted
            break
        else:
            print('Last coefficient is not significant at 95% level...')

    return result

model_q2 = ar_g2s(4, 0.05)
model_q2_params = model_q2.params
print('Parameters of the AR model: \n', model_q2_params)

## Question 3:
# ACF of AR(1) residuals
model_q2_resids = model_q2.resid
sm.graphics.tsa.plot_acf(model_q2_resids, lags=12,
                         title='AR(1) residuals ACF plot')




