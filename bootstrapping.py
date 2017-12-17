import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import DistanceMetric, BallTree


# load full data 
full_data = pd.read_csv("ts_corn_47yr_df.csv")
full_data.fips_code = full_data.fips_code.astype(str).apply(lambda x: x.zfill(5))
full_data.set_index('fips_code', inplace=True)
full_data.columns = full_data.columns.astype(int)
# load county centroids (inner join with the data)
centroids_df = pd.read_csv("county_centroids.csv", index_col=0)
centroids_df.fips_code = centroids_df.fips_code.astype(str).apply(lambda x: x.zfill(5))
centroids_df.set_index('fips_code', inplace=True)
full_data_centroids = full_data.join(centroids_df) 
full_spatial = full_data_centroids[['latitude', 'longitude']]
# create haversine distance df (only needs to be done once)
hav_dist_df = np.radians(full_spatial)

# load predictions from FULL linear regression models 
# will be used for indexing later
full_LR = pd.read_csv("lin_reg_preds.csv", index_col="fips_code")
full_LR.columns = full_LR.columns.astype(int)

full_HAV_LR = pd.read_csv("lin_reg_haversine_norm_preds.csv", index_col="fips_code")
full_HAV_LR.columns = full_HAV_LR.columns.astype(int)

full_HAV_EXP2_LR = pd.read_csv("lin_reg_hav_norm_exp2_df.csv", index_col="fips_code")
full_HAV_EXP2_LR.columns = full_HAV_EXP2_LR.columns.astype(int)

# load nearest neighbors 
############################



def mape(y_pred,y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


pop_list = np.random.randint(1970,2017, full_data.shape[0])
true = [full_data.iloc[i][yr] for i, yr in enumerate(pop_list)]

full_test = full_data.copy() # make a copy before removing data
for i, yr in enumerate(pop_list):
    full_test.iloc[i][yr] = np.nan 
    
# linear regression preds using only individual county data
y_pred = []

for i in range(full_data.shape[0]):
    X = full_data.columns.tolist()
    X.remove(pop_list[i])
    X_miss = np.array(X).reshape(-1,1)
    y = full_test.iloc[i].dropna().values
    lm = LinearRegression()
    lm.fit(X_miss, y)
    
    X_full = np.array(full_data.columns).reshape(-1,1)
    preds = lm.predict(X_full)
    y_pred.append(preds[pop_list[i]-1970])


ffill_df = full_test.copy()
ffill_df.fillna(method='ffill', axis=1, inplace=True)
# unless it is the first year in the series, then fill in value from the second year
ffill_df.fillna(method='bfill', axis=1, inplace=True)

ffill_preds = [ffill_df.iloc[i][pop_list[i]] for i in range(full_test.shape[0])]

# fill in missing values (1) with the mean of all years for that specific county
yr_mean_df = full_test.copy()
yr_mean_df = yr_mean_df.apply(lambda x: x.fillna(x.mean()),axis=1)

yr_mean_preds = [yr_mean_df.iloc[i][pop_list[i]] for i in range(full_test.shape[0])]

# fill in missing values (random number per year) with the mean of each year 
county_mean_df = full_test.copy()
county_mean_df = county_mean_df.apply(lambda x: x.fillna(x.mean()),axis=0)

county_mean_preds = [county_mean_df.iloc[i][pop_list[i]] for i in range(full_test.shape[0])]

#save baseline and individual county linear regression predicitions 
predictions = pd.DataFrame({"years":pop_list}, index=full_data.index)
predictions["y_true"] = true
predictions["lr_preds"] = y_pred
predictions["ffill_preds"] = ffill_preds
predictions["yr_mean_preds"] = yr_mean_preds
predictions["county_mean_preds"] = county_mean_preds


# exponential smoothing (haversine distance)
for i in [.25, .5, 1, 2, 4, 8, 16]:
    hav_exp_smooth = hav_dist_df.copy()
    hav_exp_smooth = hav_exp_smooth.applymap(lambda x: x**-i if x != 0 else 0)
    hav_exp_smooth = hav_exp_smooth.apply(lambda x: x/sum(x), axis=1)
 
    hav_exp_smooth_vals = np.array(hav_exp_smooth).dot(np.array(full_data))

    predictions["exp_smooth_preds{}".format(i)]  = [hav_exp_smooth_vals[j][pop_list[j]-1970] for j in range(full_test.shape[0])]

# Looking up the missing values in the FULL linear regression tables 
FULL_LR_preds = [full_LR.iloc[i][pop_list[i]] for i in range(full_test.shape[0])]
FULL_HAV_LR_preds = [full_HAV_LR.iloc[i][pop_list[i]] for i in range(full_test.shape[0])]
FULL_HAV_LR_EXP2_preds = [full_HAV_EXP2_LR.iloc[i][pop_list[i]] for i in range(full_test.shape[0])]

predictions["FULL_LR_pred"] = FULL_LR_preds
predictions["FULL_HAV_LR_pred"] = FULL_HAV_LR_preds
predictions["FULL_HAV_LR_EXP2_pred"] = FULL_HAV_LR_EXP2_preds
























