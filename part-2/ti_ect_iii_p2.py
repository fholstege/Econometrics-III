# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import acf
import itertools
from operator import itemgetter 
from scipy.stats import norm


# get rid of scientific notation
pd.options.display.float_format = '{:.2f}'.format

# Data
df = pd.read_csv('data_assign_p2.csv')
df['obs'] = pd.to_datetime(df['obs'])



## Question 1:
# Plot of GDP and

figure, axis1 = plt.subplots()

# first, plot the GDP growth rate
color1 = 'tab:red'
axis1.set_xlabel('time (s)')
axis1.set_ylabel('GDP Growth Rate (%)', color=color1)
axis1.plot(df['obs'], df['GDP_QGR'], color=color1)
axis1.tick_params(axis='y', labelcolor=color1)

axis2 = axis1.twinx()  # instantiate a second axis that shares the same x-axis

# second, plot the UN rate on the second axis
color2 = 'tab:blue'
axis2.set_ylabel('UE Rate (%)', color=color2)  
axis2.plot(df['obs'], df['UN_RATE'], color=color2)
axis2.tick_params(axis='y', labelcolor=color2)

# show together in the same plot
figure.tight_layout()  
plt.show()


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

def ADL_model(df, sDependent, lIndependent, lLags, exo_adjust = True):

    # define empty string with exogenous var to be filled if exo_ajdust = True
    ex_var_t = ''

    if exo_adjust:
        # save here the exogeneous variables at t
        exogenous_var = [x for x in lIndependent if x != sDependent]

        # add to formula
        for ex_var in exogenous_var:
            ex_var_t = ex_var+' +'
  
    # save here the lags for all other variables
    independent_form = ''

    # index to count
    i = 0

    # add lags for the independent var
    for index_var in range(0,len(lIndependent)):
        lags_var = lLags[index_var]
        var = lIndependent[index_var]


        for n_lags in lags_var:
            str_independent = 'lag('+var+', '+str(n_lags)+') + '
            independent_form += str_independent
        i += 1

    # create string for formula
    dependent_form = sDependent + " ~ 1 + " + ex_var_t
    full_form = dependent_form + independent_form[:-2]
    
    # create model and fit
    model_est = smf.ols(formula = full_form, data = df)
    model_fitted = model_est.fit(use_t = True)

    return(model_fitted)


def ADL_General2Specific(df,iMax_lags, fSignificance_level, sDependent,lIndependent, nlags_acf = 12):

    # get number of variables
    n_vars = len(lIndependent)

    # get which vars are exgogenous
    exogenous_var = [x for x in lIndependent if x != sDependent]
    index_exogenous_var = np.where(np.isin(exogenous_var,lIndependent))

    #  boolean: while true, continue to try out lags
    one_var_insignificant = True
    
    # start at both at max lag
    lag_structure = list(range(1,iMax_lags+1))
    current_lags = []

    for j in range(0, n_vars):
        
        if j in [index_exogenous_var]:
            exo_lags = [0] + lag_structure
            current_lags.append(exo_lags)
        else:
            current_lags.append(lag_structure)



    # while one insignificant, check models
    while one_var_insignificant:

        # run adl model for current iteration
        current_result = ADL_model(df = df, sDependent = sDependent, lIndependent=lIndependent,lLags=current_lags, exo_adjust=False)

        # get the residuals and acf
        current_result_residuals =  current_result.resid
        acf_result = acf(current_result_residuals, adjusted = True, nlags = nlags_acf, qstat = True, fft = False)
        qstat_pval = acf_result[2]

        if min(qstat_pval) <= fSignificance_level:
            one_var_insignificant = False
     
        # get the p-values
        result_pvalues = list(current_result.pvalues[1:])

        # get largest p-value
        largest_pvalue = max(result_pvalues)

        # get largest p-value and check if statistically insignificant
        if largest_pvalue <= fSignificance_level:
            one_var_insignificant = False
        # if not, check index of the largest value
        else:
            index_largest_value = result_pvalues.index(largest_pvalue)
        
            # flatten the list of lags
            flat_current_lags = list(itertools.chain(*current_lags))

            # remove insignificant var
            del flat_current_lags[index_largest_value]


            # turn the list back to one that can be used in adl_model
            previous_lag = flat_current_lags[0]
            lags_var = [previous_lag]
            current_lags = []

            for lag in flat_current_lags[1:]:
 
                if lag > previous_lag:
                    lags_var.append(lag)
                else:
                    current_lags.append(lags_var)
                    lags_var = []
                    lags_var.append(lag)

                previous_lag = lag
            
            current_lags.append(lags_var)
    return current_result
        


result_adl_UNRATE = ADL_General2Specific(df, 4, 0.05, 'UN_RATE', ['UN_RATE', 'GDP_QGR'])
print("ADL Unemployment Model: parameters and p-values")
print(result_adl_UNRATE.params)
print(result_adl_UNRATE.pvalues)

result_ar_GDP = ADL_General2Specific(df, 4, 0.05, 'GDP_QGR', ['GDP_QGR'])
print("AR GDP Model: parameters and p-values")
print(result_ar_GDP.params)
print(result_ar_GDP.pvalues)


# Question 2:
## short-run: there is none, = 0
## long-run: use derivation from lecture slides, sum the coefficients
## two-step: use derivation from lecture slides, estimate y_t and x_t 

X_hat = np.mean(df['GDP_QGR'])
b_0 = result_adl_UNRATE.params[1]
b_1 = result_adl_UNRATE.params[2]
b_2 = result_adl_UNRATE.params[3]
sum_b0_b1= np.sum(b_0,b_1)
alpha = result_adl_UNRATE.params[0]

long_term_UN_RATE = ((X_hat * b_2) + alpha)/(1-sum_phi1_phi3)
print("Long-term UN Rate est.")
print(long_term_UN_RATE)


two_step_multiplier = b_0 * b_2 + b_2 * result_ar_GDP.params[1]
print("two-step multiplier:")
print(two_step_multiplier)

# Question 3:
## Suppose that the innovations are iid Gaussian. What is the probability of the unemployment rate rising above 7.8% in the second quarter of 2014? What is the probability that
## it drops below 7.8%? Do you trust the iid Gaussian assumption?

# prediction for quarter 2 of 2014

df_prediction_forQ22014= df.tail(4)
print(df_prediction_forQ22014)

prediction_Q22014  = result_adl_UNRATE.predict(df_prediction_forQ22014).iloc[-1]
print("Prediction for Q2 of 2014")
print(prediction_Q22014)
print(type(prediction_Q22014))

residuals_ADL = result_adl_UNRATE.resid
plt.hist(residuals_ADL)
plt.show()

mean_residuals = np.mean(residuals_ADL)
var_residuals = np.var(residuals_ADL)

print("Mean residuals of ADL: ",mean_residuals )
print("Variance residuals of ADL: ",var_residuals )

benchmark = 7.8
diff_prediction_benchmark = benchmark - prediction_Q22014
print(benchmark)
print(prediction_Q22014)
chance_observing_higher = norm.cdf(diff_prediction_benchmark, loc = 0, scale =np.sqrt(var_residuals ))
print("Chance of observing difference of ", diff_prediction_benchmark)
print(chance_observing_higher)




