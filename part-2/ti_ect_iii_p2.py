# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.formula.api as smf

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

def ADL_model(df, sDependent, lIndependent, lN_lags):

    # save here the exogeneous variables at t
    ex_var_t = ''
    exogenous_var = [x for x in lIndependent if x != sDependent]

    # save here the lags for all other variables
    independent_form = ''

    # index to count
    i = 0

    # these two for loops create a formula for ADL model
    for ex_var in exogenous_var:
        ex_var_t = ex_var+' +'

    for var in lIndependent:
        for n_lags in range(1, lN_lags[i] + 1):
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

    #  boolean: while true, continue to try out lags
    one_var_insignificant = True
    
    # start at both at max lag
    current_lags = np.array([iMax_lags]*n_vars)

    # create list of indices for p-values to check
    multiplier_pval = np.array(range(1,len(current_lags)+1))
    check_pval_indeces = (current_lags * multiplier_pval)-1

    # while one insignificant, check models
    while one_var_insignificant:

        # run adl model for current iteration
        current_result = ADL_model(df = df, sDependent = sDependent, lIndependent=lIndependent,lN_lags=current_lags )

        # get p-values for current iteration
        result_pvalues = np.array(current_result.pvalues[1:])


        # get p-values for largest lags (e.g. if largest lag is 4, get 4th lag)
        pval_to_check = result_pvalues[check_pval_indeces]

        # count how many are not significant
        n_not_significant = 0

        # count how many var have been removed in process
        n_removed = 0

        # go through all p-values 
        for i in range(0,len(pval_to_check)):
            
            # current p-value to check
            pval_var = pval_to_check[i]

            # change indices based on how many already removed
            check_pval_indeces[i - n_removed] = check_pval_indeces[i - n_removed]- n_not_significant

            # if statistically insignificant...
            if pval_var > fSignificance_level:
                
                # reduce the lag - e.g. if 4th lag statistically insignificant, remove that one
                current_lags[i] = current_lags[i]-1

                # change indices of p-values to check
                check_pval_indeces[i] = check_pval_indeces[i]-1 

                # continue counting how many significant for while loop
                n_not_significant += 1
            
            # if one of the variables removed (because all statistically insignificant)
            if current_lags[i-n_removed] <=0:
                current_lags = np.delete(current_lags, [i])
                check_pval_indeces = np.delete(check_pval_indeces, [i])
                del lIndependent[i]
                n_removed =+ 1
                
            
        # if none statistically insignificant, then stop while        
        if n_not_significant == 0:
            one_var_insignificant = False
    # return latest model
    return(current_result)    



UNRATE_ADL = ADL_General2Specific(df,4 ,0.1,'UN_RATE', ['GDP_QGR', "UN_RATE"])
UNRATE_ADL_params = UNRATE_ADL.params
print('Parameters of the ADL model: \n', UNRATE_ADL_params)

# question 2
ADL_UNRATE_G2S = ADL_model(df, 'UN_RATE', ['UN_RATE'], [3])

# short run
print(ADL_UNRATE_G2S.params)

# long run
mean_lag1_unrate = np.mean(lag(df['UN_RATE'],1))
mean_lag2_unrate = np.mean(lag(df['UN_RATE'],2))
mean_lag3_unrate = np.mean(lag(df['UN_RATE'],3))

df_LR = pd.DataFrame()
df_LR['mean_lag1_UN_RATE'] = mean_lag1_unrate
df_LR['mean_lag2_UN_RATE'] = mean_lag2_unrate
df_LR['mean_lag3_UN_RATE'] = mean_lag3_unrate


