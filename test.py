#################################################################################################
# %% EUR CROP PRODUCTION MODEL
# version: 0.0.2
# last reviewed: 2024-Dec-30, daniel.pereira@lseg.com

# this script estimates yield-area based on weather variables suggested by regression fits
# Geographies: EU-28 countries
# Crops: Wheat durum, Wheat , Corn, Rapeseed

# TODO (PARTIAL COMPLETE) get rid of dependency on mlxtend
# TODO Create individual files for each Country X Crop
# TODO #def main():
################################################################################################
# %% IMPORT AND CONFIG 
import polars as pl
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# polars config
pl.Config.set_float_precision(2)
pl.Config.set_tbl_formatting("NOTHING")
pl.Config.set_tbl_cols(22)

# %% DATA LOADING
## reading from a spreadsheet allows for data reviewing. In this dataset missing observations is 
## commonplace. 
dat = pl.read_excel(r"C:\\Users\\U6084679\\OneDrive - London Stock Exchange Group\\Ags Research\\EU\\MODEL\\EUR_CROPS\\EU_CROPS.xlsx", sheet_name="_dt")

# %% MUNGING-01

countriesForAnalysis = ["AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA",
                        "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "NLD", "POL", "PRT",
                        "ROU", "SVK", "SVN", "SWE", "GBR"]

dat = dat.drop(["ID", "FD", "VD", "key"])

## only countries that will be analyzed
dat = dat.filter(pl.col("geo").is_in(countriesForAnalysis))
dat = dat.rename({"V": "value"})

# %% USER INPUT 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# DEFINE CROP FOR ANALYSIS #########################################################################
list(set(list(dat['crop'])))
list(set(list(dat['geo'])))
dat = dat.filter(pl.col("crop") == "Common wheat & spelt"); dat
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<x<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# %% TREND Definition
# what is expected for next year? There are several ways to approach this. Detrending the series is 
# recommended. I will use a simple naive median on version 0.0.1 and improve it on subsequent versions
dat = dat.sort(["geo", "crop", "year"])
dat = dat.filter(pl.col("var") == "Yield")

medians = dat.filter(pl.col("year") > 2015).filter(pl.col("crop") == "Durum wheat").group_by(pl.col("geo")).agg(pl.median("value"))
medians = medians.with_columns(year = pl.lit(2025))

dat = dat.join(medians, on=["geo"], how="left")
dat = dat.drop("year_right")
dat = dat.rename({"value_right": "medianAdopted"})
dat = dat.rename({"value": "yield"})
dat = pl.DataFrame.to_pandas(dat)
dat


# %% chart yield series and trend values
sns.set_theme(style="white")
datTmp = dat
g = sns.FacetGrid(data=datTmp, col="geo", col_wrap=6)
g = g.map(sns.lineplot, "year", "yield")
# FIX NEEDED g = g.xticklabels(range(min(datTmp['year']), max(datTmp['year']) + 1))
# g =g.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.show()

# %% read weather
#### pl.read_csv(r"C:\\Users\\U6084679\\OneDrive - London Stock Exchange Group\\Ags Research\\EU\\MODEL\\EUR_CROPS\\_dt_weather\\AgWea.csv")
#### quick visualization - April precip is an important variable

t = pd.read_csv(filepath_or_buffer=r"C:\\Users\\U6084679\\OneDrive - London Stock Exchange Group\\Ags Research\\EU\\MODEL\\EUR_CROPS\\_dt_weather\\AgWea.csv")
t = t.melt(id_vars=["geounit", "prev_year", "year"])

t["country"] = t["geounit"].str[:3]

t = t[t.variable.str.contains("prcp")]
t = t.groupby(["country", "year", "variable"]).value.aggregate(["sum", "min", "max", "mean"]).reset_index()

g = sns.FacetGrid(col="country", height=4, aspect=1/.65, col_wrap=6, data=t,
                   margin_titles=True, despine=False, palette="Set1",
                   sharey=False, sharex=False)
g = g.map_dataframe(sns.histplot, x='mean', cumulative=False, common_norm=True, element="bars", fill=True, kde=True)
g = g.map_dataframe(sns.rugplot, x="mean", data=t, color="darkorange")
g = g.map_dataframe(lambda data, **kws: plt.axvline(30, color="darkorange", linewidth=7))
g
g.savefig("tmpBarbara.pdf", dpi=10, transparent=True, facecolor="white", pad_inches=0.25,edgecolor="pink")


# %% join and merge datasets  --- weather + ayp
t = pd.read_csv(filepath_or_buffer=r"C:\\Users\\U6084679\\OneDrive - London Stock Exchange Group\\Ags Research\\EU\\MODEL\\EUR_CROPS\\_dt_weather\\AgWea.csv")
t["country"] = t["geounit"].str[:3]
t = t.drop("geounit",  axis=1)
t = t.groupby(["country", "year", "prev_year"]).agg(lambda x: np.median(x))                   ### How about mean here?
t = t.reset_index() # gotta like indexes huh. I don't

dat = dat.drop(["medianAdopted", "var", "crop"], axis=1)
dat = dat.rename(columns={'geo': 'country'})

dt = pd.merge(left=dat,right=t,on=["year", "country"])
dt
# %% 
dt = dt.query("country == 'DEU'")
dtX = dt.drop(["prev_year", "yield", "country"], axis=1)
dtY = dt["yield"]

# %% using sklearn and exhaustive selection on mlxtend
## exhaustive takes a long time to run. 
## https://scikit-learn.org/stable/modules/feature_selection.html

''' 
There are issues with this approach:

more info here: https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn/24447#24447
and here: https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-a-forward-selection-stepwise-regression-algorithm

Because of that, scikit-learn deliberately avoids inferential approach to model learning.

In the next version of this implementation 

'''

from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

X, y = dtX, dtY
lr = LinearRegression()
efs = EFS(lr, 
          min_features=2,
          max_features=2,
          scoring='neg_mean_squared_error',
          cv=2, 
          print)

# time consuming step
efs.fit(X, y)

## print results 
print('Best MSE score: %.2f' % efs.best_score_ * (-1))
print('Best subset:', efs.best_idx_)


import statsmodels as sm
import statsmodels.api as sm
model = sm.OLS(y, X) #X is a matrix with the explanatory variables listed in 4.3


results = model.fit()
print(results.summary())




# %% 
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import statsmodels.api as sm

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X, y = dtX, dtY

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, #0.01 original
                       threshold_out = 0.05, #0.05 original 
                       verbose=True):
    
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included



# !!! Barbara start from here
# %% the practical stuff :

import statsmodels as sm
import statsmodels.api as sm

result = stepwise_selection(X, y, threshold_out = 0.04, threshold_in = 0.01,
                            verbose=True, 
                            initial_list=[
'year',
 #'pprcp_jan',
 'pprcp_feb',
 'pprcp_mar',
 'pprcp_apr',
 'pprcp_may',
 'pprcp_jun',
 'pprcp_jul',
 'pprcp_aug',
 'pprcp_sep',
 #'pprcp_oct',
 #'pprcp_nov',
 #'pprcp_dec',

 'prcp_apr',
 'prcp_aug',
 'prcp_dec',
 'prcp_feb',
 'prcp_jan',
 'prcp_jul',
 'prcp_jun',
 'prcp_mar',
 'prcp_may',
 #'prcp_nov',
 #'prcp_oct',
 #'prcp_sep',
 
 'psum_edd_apr',
 'psum_edd_aug',
 'psum_edd_dec',
 'psum_edd_feb',
 'psum_edd_jan',
 'psum_edd_jul',
 'psum_edd_jun',
 'psum_edd_mar',
 'psum_edd_may',
 'psum_edd_nov',
 'psum_edd_oct',
 'psum_edd_sep',

 #'psum_gdd_corn_apr',
 #'psum_gdd_corn_aug',
 #'psum_gdd_corn_dec',
 #'psum_gdd_corn_feb',
 #'psum_gdd_corn_jan',
 #'psum_gdd_corn_jul',
 #'psum_gdd_corn_jun',
 #'psum_gdd_corn_mar',
 #'psum_gdd_corn_may',
 #'psum_gdd_corn_nov',
 #'psum_gdd_corn_oct',
 #'psum_gdd_corn_sep',
 #'psum_gdd_soy_apr',
 #'psum_gdd_soy_aug',
 #'psum_gdd_soy_dec',
 #'psum_gdd_soy_feb',
 #'psum_gdd_soy_jan',
 #'psum_gdd_soy_jul',
 #'psum_gdd_soy_jun',
 #'psum_gdd_soy_mar',
 #'psum_gdd_soy_may',
 #'psum_gdd_soy_nov',
 #'psum_gdd_soy_oct',
 #'psum_gdd_soy_sep',

 #'psum_gdd_wheat_jan',
 #'psum_gdd_wheat_feb',
 #'psum_gdd_wheat_mar',
 #'psum_gdd_wheat_apr',
 'psum_gdd_wheat_may',
 'psum_gdd_wheat_jun',
 'psum_gdd_wheat_jul',
 'psum_gdd_wheat_aug',
 #'psum_gdd_wheat_sep',
 #'psum_gdd_wheat_oct',
 #'psum_gdd_wheat_nov',
 #'psum_gdd_wheat_dec',
 
 'ptavg_apr',
 'ptavg_aug',
 'ptavg_dec',
 'ptavg_feb',
 'ptavg_jan',
 'ptavg_jul',
 'ptavg_jun',
 'ptavg_mar',
 'ptavg_may',
 'ptavg_nov',
 'ptavg_oct',
 'ptavg_sep',

 'ptmax_apr',
 'ptmax_aug',
 'ptmax_dec',
 'ptmax_feb',
 'ptmax_jan',
 'ptmax_jul',
 'ptmax_jun',
 'ptmax_mar',
 'ptmax_may',
 'ptmax_nov',
 'ptmax_oct',
 'ptmax_sep',
 
 'ptmin_apr',
 'ptmin_aug',
 'ptmin_dec',
 'ptmin_feb',
 'ptmin_jan',
 'ptmin_jul',
 'ptmin_jun',
 'ptmin_mar',
 'ptmin_may',
 'ptmin_nov',
 'ptmin_oct',
 'ptmin_sep',
 
 'sum_edd_apr',
 'sum_edd_aug',
 'sum_edd_dec',
 'sum_edd_feb',
 'sum_edd_jan',
 'sum_edd_jul',
 'sum_edd_jun',
 'sum_edd_mar',
 'sum_edd_may',
 'sum_edd_nov',
 'sum_edd_oct',
 'sum_edd_sep',
 
 #'sum_gdd_corn_apr',
 #'sum_gdd_corn_aug',
 #'sum_gdd_corn_dec',
 #'sum_gdd_corn_feb',
 #'sum_gdd_corn_jan',
 #'sum_gdd_corn_jul',
 #'sum_gdd_corn_jun',
 #'sum_gdd_corn_mar',
 #'sum_gdd_corn_may',
 #'sum_gdd_corn_nov',
 #'sum_gdd_corn_oct',
 #'sum_gdd_corn_sep',
 #'sum_gdd_soy_apr',
 #'sum_gdd_soy_aug',
 #'sum_gdd_soy_dec',
 #'sum_gdd_soy_feb',
 #'sum_gdd_soy_jan',
 #'sum_gdd_soy_jul',
 #'sum_gdd_soy_jun',
 #'sum_gdd_soy_mar',
 #'sum_gdd_soy_may',
 #'sum_gdd_soy_nov',
 #'sum_gdd_soy_oct',
 #'sum_gdd_soy_sep',
 
 #'sum_gdd_wheat_jan',
 #'sum_gdd_wheat_feb',
 'sum_gdd_wheat_mar',
 'sum_gdd_wheat_apr',
 'sum_gdd_wheat_may',
 'sum_gdd_wheat_jun',
 'sum_gdd_wheat_jul',
 'sum_gdd_wheat_aug',
 #'sum_gdd_wheat_sep',
 #'sum_gdd_wheat_oct',
 #'sum_gdd_wheat_nov',
 #'sum_gdd_wheat_dec',
 
 'tavg_apr',
 'tavg_aug',
 'tavg_dec',
 'tavg_feb',
 'tavg_jan',
 'tavg_jul',
 'tavg_jun',
 'tavg_mar',
 'tavg_may',
 'tavg_nov',
 'tavg_oct',
 'tavg_sep',
 
 'tmax_apr',
 'tmax_aug',
 'tmax_dec',
 'tmax_feb',
 'tmax_jan',
 'tmax_jul',
 'tmax_jun',
 'tmax_mar',
 'tmax_may',
 'tmax_nov',
 'tmax_oct',
 'tmax_sep',
 
 'tmin_apr',
 'tmin_aug',
 'tmin_dec',
 'tmin_feb',
 'tmin_jan',
 'tmin_jul',
 'tmin_jun',
 'tmin_mar',
 'tmin_may',
 'tmin_nov',
 'tmin_oct',
 'tmin_sep'])


print('resulting features:')
print(result)


# %%
# TODO Populate Excel automagically
