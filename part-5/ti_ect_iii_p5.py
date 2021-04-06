# Econometrics III - Part 5

# Imports Python:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset

# Plot style:
plt.style.use('ggplot')


#%% Data

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


#%% Properties

# Summary statistics:
print(df.describe().to_latex(float_format="%.2f"))

# Construct first difference of both variables:
df['D1 New Cases'] = (df['New Cases'] - df['New Cases'].shift(1)).fillna(0)
df['D1 7-Day Moving Avg'] = (df['7-Day Moving Avg']
                                 - df['7-Day Moving Avg'].shift(1)).fillna(0)

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


# New Cases ACF plot:
sm.graphics.tsa.plot_acf(df['New Cases'], lags=20, 
                         title='First diff. new cases ACF plot')
plt.show()

# New Cases PACF plot:
sm.graphics.tsa.plot_pacf(df['New Cases'], lags=20,
                          title='First diff. new cases PACF plot')
plt.show()


#%% Estimation

# Model selector:
def arma_model_selector(maxlag):
    grid = pd.DataFrame(index=[0])
    for p in range(1,maxlag+1):
        for q in range (1,maxlag+1):
            try: # It is possible for a model to be non-stationary
                model = ARIMA(df['New Cases'],
                              order=(p,1,q),
                              trend='n').fit(method='innovations_mle')
                grid[p,q] = model.bic
                
            except ValueError:
                print('Non-stationary! Model specification: (',p,q,')')
                grid[p,q] = np.nan
    
    grid_min_bic = grid.min(axis=1)
    model_specification = grid.idxmin(axis=1)
    
    return grid, grid_min_bic, model_specification

# MAXLAGS=20
# grid, grid_min_bic, model_specification = arma_model_selector(MAXLAGS)
### arma_model_selector() -> RESULT: ARMA(4,3)   "BASED ON MAXLAGS=20"

model = ARIMA(df['New Cases'],
                          order=(4,1,3),
                          freq=DateOffset(days=1),
                          trend='n').fit(method='innovations_mle')


#%% Model validation

residuals = model.resid

# Residuals ACF plot:
sm.graphics.tsa.plot_acf(residuals, lags=10, 
                         title='ARMA(4,3) residuals ACF plot')
plt.show()

# Residuals PACF plot:
sm.graphics.tsa.plot_pacf(residuals, lags=10,
                          title='ARMA(4,3) residuals PACF plot')
plt.show()


#%% Model forecast

forecast_results = model.get_forecast(steps=14) # 2 weeks ahead forecast
df_forecast = forecast_results.conf_int(alpha=0.05)
df_forecast['Point Estimate'] = forecast_results.predicted_mean

plt.plot(df_forecast['Point Estimate'], color='k', label='Forecast')
plt.plot(df_forecast['upper New Cases'], color='r',
         label='95% Confidence Bounds', linestyle='--')
plt.plot(df_forecast['lower New Cases'], color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('New Cases')
plt.title('Two week ahead forecast of US daily new Covid-19 cases')
plt.legend()
plt.show()



