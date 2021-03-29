# Econometrics III - Part 3:

# Imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

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
















