# Econometrics III Part I

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
plt.show()

# PACF plot:
sm.graphics.tsa.plot_pacf(df['GDP_QGR'], lags=12,
                          title='GDP Quarterly Growth rate PACF plot')
plt.show()

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
plt.show()

## Question 4 and 5:
# Forecast
q4_forecast = model_q2.get_prediction(start='2009-04-01', end='2011-01-01')
forecast_table = pd.merge(q4_forecast.predicted_mean, q4_forecast.conf_int(alpha=0.05),
                          right_index=True, left_index=True)
print(forecast_table.to_latex())

## Question 6:
# JB test:
jb = sm.stats.stattools.jarque_bera(model_q2_resids, axis=0)
print('The JB test-statistic = ',jb[0],
      '\nP-value = ', jb[1])

# ARCH-LM test for hetereoskedasticity:
hetreosk_test = model_q2.test_heteroskedasticity(lags=12)
print(hetreosk_test.to_latex())


## Question 7:
# Merge forecast with actual data:
data_q7 = pd.Series(data=[-1.63, 0.28, 0.33, 0.66, 1.51, 0.51, 0.71, 0.81],
                    index=forecast_table.index, name='actual')
forecast_table = pd.merge(forecast_table, data_q7, right_index=True, left_index=True)

# Plot of forecast vs actual:
plt.plot(forecast_table['predicted_mean'], label='Forecast', linestyle='--')
plt.plot(forecast_table['lower'], color='royalblue', label='95% Confidence bound')
plt.plot(forecast_table['upper'], color='royalblue')
plt.plot(forecast_table['actual'], label='Actual',color='red')
plt.fill_between(forecast_table.index, 
                 forecast_table['lower'], forecast_table['upper'],
                 color='lightsteelblue', alpha=0.2)
plt.ylabel('GDP growth rate')
plt.xlabel('Time')
plt.title('Forecast compared to actual GDP growth rate')
plt.legend()
plt.show()

# MAE:
forecast_table['error'] = forecast_table['predicted_mean'] - forecast_table['actual']
mae_q7 = np.abs(forecast_table['error'].mean())
print('The mean absolute error =', mae_q7)