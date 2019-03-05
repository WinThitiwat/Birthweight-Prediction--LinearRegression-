"""
Created on Sat Mar  2 11:37:51 2019

@author: Machine Learning Cohort 1 Group 3
"""
# Loading Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf  # regression modeling with stats info
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  # train/test split
from sklearn import metrics
import numpy as np

# loading data
file = "birthweight_feature_set.xlsx"
birthweight = pd.read_excel(file)

# print shape of the dataset
print(birthweight.shape)

# get statistical info
print(birthweight.describe())

# print number of missing value in each columns
print(birthweight.isnull().sum())


# function: fill missing value based on the median of the `s` series
def fill_mising_with_median(s):
    m = s.median()
    s.fillna(m, inplace=True)
    return s

# get a list of columns containing missing values
m_vars_list = list(birthweight.columns[birthweight.isnull().any()])

# fill in the missing values on columns from `m_vars_list`
# using the `fill_mising_with_median` function
birthweight[m_vars_list] = birthweight[m_vars_list]\
                            .apply(fill_mising_with_median)


######################################
#  Data Visualization ##
######################################

# make a copy in case some implementation is incorrect
bw_explored = birthweight.copy()

# filter mom and father ethinicity
mom_white_father_white = (
    (bw_explored.mwhte == 1) &
    (bw_explored.fwhte == 1)
    )
mom_black_father_black = (
    (bw_explored.mblck == 1) &
    (bw_explored.fblck == 1)
    )
mom_other_father_other = (
    (bw_explored.moth == 1) &
    (bw_explored.foth == 1)
    )

# target variable to be compared with ethinicity
target_var = "bwght"

# all ethinicity list in a dictionary
target_plot = {
        "mwfw": bw_explored[mom_white_father_white][target_var],
        "mbfb": bw_explored[mom_black_father_black][target_var],
        "mofo": bw_explored[mom_other_father_other][target_var]
               }

# display seaborn distplot to visualize each distribution of ethinicity
for col, plot_value in target_plot.items():
    sns.distplot(plot_value, label=col)
    plt.legend()


##################################################################
# before we decide what the outlier values are, we used boxplot to
# do visual inspection to finalize the outlier values
##################################################################


######################################
#  Features Engineering  ##
######################################

# Outlier flags
mage_hi = 63

monpre_hi = 4

npvis_hi = 15

npvis_lo = 7.5

fage_hi = 56

feduc_lo = 7

drink_hi = 12


# function: flag outlier based on comparing with `val_compare`
# and assign with `flag_val`
def flag_outlier(
    df,
    col,
    new_col,
    val_compare,
    flag_val,
    compare_greater=True
):
    try:
        if len(df[new_col]):
            pass
    except:
        df[new_col] = 0
    if compare_greater:
        for col_val, idx in enumerate(df.loc[:, col]):
            if col_val >= val_compare:
                df.loc[idx, new_col] = flag_val
    else:
        for col_val, idx in enumerate(df.loc[:, col]):
            if col_val <= val_compare:
                df.loc[idx, new_col] = flag_val

flag_outlier(birthweight, "mage", "out_mage", mage_hi, 1)
flag_outlier(birthweight, "monpre", "out_monpre", monpre_hi, 1)
flag_outlier(
    birthweight,
    "npvis",
    "out_npvis",
    npvis_lo,
    -1,
    compare_greater=False
    )
flag_outlier(birthweight, "npvis", "out_npvis", npvis_hi, 1)
flag_outlier(birthweight, "fage", "out_fage", fage_hi, 1)
flag_outlier(
    birthweight,
    "feduc",
    "out_feduc",
    feduc_lo,
    -1,
    compare_greater=False
    )
flag_outlier(birthweight, "drink", "out_drink", drink_hi, 1)

# inspect some new outlier flag columns
print(birthweight["out_monpre"].head())
print(birthweight["out_fage"].head())


######################################
#  Data Analysis ##
######################################
# make a copy in case some implementation is incorrect
bw_data = birthweight.copy()

# Preparing a DataFrame based the the analysis abov
bw_independents = bw_data.loc[:, [
                                'mage',
                                'cigs',
                                'drink',
                                'out_mage',
                                'out_drink'
                                ]]
# Preparing the target variable
bw_target = bw_data.loc[:, "bwght"]

# Train test split with random_state at 508 and test size 10%
X_train, X_test, y_train, y_test = train_test_split(
                                                    bw_independents,
                                                    bw_target,
                                                    test_size=0.1,
                                                    random_state=508
                                                    )

# Prepping the Model
lr = LinearRegression(fit_intercept=True)

# Fitting the model
lr_fit = lr.fit(X_train, y_train)

# Predictions
lr_pred = lr_fit.predict(X_test)

# evaluating the accuracy using Root Mean Squared Error


print(f"""
        Test set predictions:
        {lr_pred.round(2)}
        """)

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

# Let's compare the testing score to the training score.
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Printing model results
print(f"""
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")

# Get statistic summary of the model
allcol = '+'.join(['mage', 'cigs', 'drink', 'out_mage', 'out_drink'])
my_formula = "bwght~"+allcol
lm_significant = smf.ols(formula=my_formula,
                         data=bw_data)
results = lm_significant.fit()
print(results.summary())

# evaluate R-Square (same as the score above)
lr_rsq = metrics.r2_score(y_test, lr_pred)
print(lr_rsq)

# evaluate Mean Squared Error
lr_mse = metrics.mean_squared_error(y_test, lr_pred)
print(lr_mse)

# evaluate Root Mean Squared Error (how far off are we on each observation?)
lr_rmse = np.sqrt(lr_mse)
print(lr_rmse)

# export prediction values to Excel
df_lr_pred = pd.DataFrame(
                        lr_pred.round(2),
                        columns=["predicted_weight"])
df_lr_pred.to_excel("predicted_weight.xlsx", index=False)
