# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.formula.api as smf
from statsmodels.tsa.arima_model import ARIMA
import itertools

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


def AR_General2Specific(iMax_lags, fSignificance_level, df, lVariables):
    '''
    AR_General2Specific: applies the general to specific approach for finding the appropriate number of lags 
    '''

    # go through total lags to check
    for number_lags in range(iMax_lags, 0, -1):
        AR_Model = AutoReg(df[lVariables], lags=number_lags, trend='c', old_names=False)
        AR_Model_fitted = AR_Model.fit(use_t=True)
                
        # check if significance level is matched by last lag
        if AR_Model_fitted.pvalues[-1]<=fSignificance_level:
            result = AR_Model_fitted
            break
        else:
            print('At ' + str(number_lags) +' lags, the last coefficient is not significant at the ' + str(1-fSignificance_level) + ' level')

    return result


GDP_AR = AR_General2Specific(4, 0.05, df, ['GDP_QGR'])
GDP_AR_PARAM = GDP_AR.params
print('Parameters of the AR model: \n', GDP_AR_PARAM)


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


def ADL_General2Specific(df,iMax_lags, fSignificance_level, sDependent,lIndependent):

    # get number of variables
    n_vars = len(lIndependent)

    # get which vars are exgogenous
    exogenous_var = [x for x in lIndependent if x != sDependent]
    index_exogenous_var = lIndependent.index(exogenous_var[0])

    #  boolean: while true, continue to try out lags
    one_var_insignificant = True
    
    # start at both at max lag
    lag_structure = list(range(1,iMax_lags+1))
    current_lags = []

    for j in range(0, n_vars):
        
        if j == index_exogenous_var:
            print("ADDING EXOGENOUS")
            exo_lags = [0] + lag_structure
            current_lags.append(exo_lags)
        else:
            current_lags.append(lag_structure)

    print(current_lags)
    print("888")



    # while one insignificant, check models
    while one_var_insignificant:

        # run adl model for current iteration
        current_result = ADL_model(df = df, sDependent = sDependent, lIndependent=lIndependent,lLags=current_lags, exo_adjust=False)

        # get the p-values
        result_pvalues = list(current_result.pvalues[1:])

        # get largest p-value
        largest_pvalue = max(result_pvalues)

        print(result_pvalues)

        # get largest p-value and check if statistically insignificant
        if largest_pvalue <= fSignificance_level:
            one_var_insignificant = False
        # if not, check index of the largest value
        else:
            index_largest_value = result_pvalues.index(largest_pvalue)
            print(index_largest_value)


            
        
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
            print(current_lags)
    print("---")
    print(current_lags)
    return current_result
        



#UNRATE_ADL = ADL_model(df, 'UN_RATE', ['GDP_QGR', "UN_RATE"], [[1,2,3,4], [1,2,3,4]])
#UNRATE_ADL_PARAM = UNRATE_ADL.params
#print('Parameters of the ADL model: \n', UNRATE_ADL_PARAM)


UNRATE_ADL_G2S = ADL_General2Specific(df,4, 0.05, 'UN_RATE', ['GDP_QGR', "UN_RATE"])
print(UNRATE_ADL_G2S.params)