# Econometrics III - Part 3:

# Imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from stargazer.stargazer import Stargazer, LineLocation
from IPython.core.display import HTML

# Data:
df = pd.read_csv('data_assign_p3.csv')
df.index = pd.to_datetime(df['DATE'], dayfirst=True)
df = df.drop('DATE',axis=1)


## Question 1:
# Chosen series: Apple & Exxon Mobil

# Apple Plots:
plt.plot(df['APPLE'])
plt.title('Apple Stock Price')
plt.show()

# Apple ACF plot:
sm.graphics.tsa.plot_acf(df['APPLE'], lags=12, 
                         title='Apple ACF plot')
plt.show()

# Apple PACF plot:
sm.graphics.tsa.plot_pacf(df['APPLE'], lags=12,
                          title='Apple PACF plot')
plt.show()


# Exxon Mobil plots:
plt.title('Exxon Mobil Stock Price')
plt.plot(df['EXXON_MOBIL'])
plt.show()

# Exxon Mobil ACF plot:
sm.graphics.tsa.plot_acf(df['EXXON_MOBIL'], lags=12, 
                         title='Exxon Mobil ACF plot')
plt.show()

# Exxon Mobil PACF plot:
sm.graphics.tsa.plot_pacf(df['EXXON_MOBIL'], lags=12,
                          title='Exxon Mobil PACF plot')
plt.show()


## Question 2:
# Perform an ADF test
def adf_test_all_vars():
    output = pd.DataFrame()
    for i in df:
        result = adfuller(df[i], regression='c',autolag='bic', store=False)
        output[i] = pd.Series(result[0:4], 
                           index=['Stat',
                                  'P-value',
                                  'Lags',
                                  'N'])
    return output

output_adf=adf_test_all_vars()
print(output_adf.to_latex(float_format='%.2f'))


## Question 3:
# Random walk forecast, fc = forecast:
fc_length = 5

# Create df of point forecast (simply fc) and variance (var_fc_STOCK):
fc_apple = np.full(fc_length, df['APPLE'].iloc[-1])
fc_exxon = np.full(fc_length, df['EXXON_MOBIL'].iloc[-1])

df['APPLE_L1'] = df['APPLE'].shift(1)
df['EXXON_MOBIL_L1'] = df['EXXON_MOBIL'].shift(1)

model_apple = smf.ols(formula = 'APPLE ~ APPLE_L1', data = df).fit()
model_exxon = smf.ols(formula = 'EXXON_MOBIL ~ EXXON_MOBIL_L1', data = df).fit()

var_fc_apple =  np.multiply(np.full(fc_length, model_apple.resid.var()),
                            np.array([*range(1,fc_length+1)]))
var_fc_exxon = np.multiply(np.full(fc_length, model_exxon.resid.var()),
                           np.array([*range(1,fc_length+1)]))

# Plot Apple forecast + variance:
fig, axs = plt.subplots(2)
axs[0].plot(fc_apple, color='k')
axs[0].plot(fc_apple+1.96*np.sqrt(var_fc_apple), color='k', linestyle='--')
axs[0].plot(fc_apple-1.96*np.sqrt(var_fc_apple), color='k', linestyle='--')
axs[0].set_title('Apple random walk forecast')

# Plot Exxon Mobil forecast + variance:
axs[1].plot(fc_exxon, color='k')
axs[1].plot(fc_exxon+1.96*np.sqrt(var_fc_exxon), color='k', linestyle='--')
axs[1].plot(fc_exxon-1.96*np.sqrt(var_fc_exxon), color='k', linestyle='--')
axs[1].set_title('Exxon Mobil random walk forecast')

plt.xlabel('Time')
plt.show()

## Question 4:
# Reg Mic ~ Exxon, spurious regression:
model_mic_exxon = smf.ols('MICROSOFT ~ EXXON_MOBIL', data=df).fit(use_t = True)
stargazer = Stargazer([model_mic_exxon])
stargazer.custom_columns('MICROSOFT')
stargazer.show_model_numbers(False)
stargazer.show_degrees_of_freedom(False)
print(stargazer.render_latex())





