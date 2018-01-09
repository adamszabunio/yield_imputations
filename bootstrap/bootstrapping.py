#!/usr/bin/env python3
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import DistanceMetric, BallTree


def k_neighbors(k, pop_list, full_data, neigh_table, neigh_dist, s=1): 
    '''
    Return two values:
    mean of the k-nearest neighbors, and inverse weighted mean.
    (e.g. crop yield measured in bushels/acre)
    INPUTS: 
    "k"= int, range(1-10), number of neighbors to use for computing missing values.
    "pop_list"= random discrete uniform distribution array of years. used for indexing missing values.
    "full_data"= complete data. used for looking up true values of "k" nearest points.
    "neigh_table"= array of "k" nearest points for "full_data".
    "neigh_dist"= distances array of "neigh_table".
    "s"= IDW (Inverse Distance Weighting) exponent, default=1/(dist**1), where dist=distance to neighbor. 
        (e.g. 1/(dist**"s"))
    OUTPUTS: 
    "kN_mean_preds"= list of "k" nearest neighbor predictions (continuous value for each pred). 
    "kN_IDW_preds"= IDW average using hyperparameter "s" (continuous value for each pred).
    '''
    counties = full_data.shape[0]
    kN_mean_preds, kN_IDW_preds = [], []
    for i in range(counties):
        neighbors = []
        for j in neigh_table.iloc[i][:k]:
            n = full_data.iloc[j][pop_list[i]]
            neighbors.append(n)
        kN_mean_preds.append(np.mean(neighbors))
        w = list(map(lambda x: x**-s, neigh_dist.iloc[i][:k]))
        kN_IDW_preds.append(np.average(neighbors, weights=w))
        
    return kN_mean_preds, kN_weighted_preds

def mape(y_pred, y_true):
    '''Returns the Mean Absolute Precentage Error'''
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def bootstrap(full_data, hav_dist_df, neigh_table, neigh_dist, full_LR, full_HAV_LR, full_HAV_EXP2_LR):
    '''
    Return two dataframes: 
    "predictions" of all missing values per techinque.
    "metrics" of each technique (MAPE, MAE, RMSE, RSQUARED).
    INPUTS: 
    "full_data"= complete data. used for looking up true values.
    "hav_dist_df"= pairwise dataframe of haversine distances.
    "neigh_table"= array of "k" nearest points for "full_data".
    "neigh_dist"= distances array of "neigh_table".
    "full_LR"= dataframe of 'complete data' linear regression predicitons
    "full_HAV_LR"= dataframe of 'complete data' Weighted linear regression predicitons \
        using IDW (Inverse Distance Weighting) with an exponent of 1 sum(1/(dist**1)), \
        where dist=distance to neighbors. 
    "full_HAV_EXP2_LR"= same as above, with an IDW exponent of 2
    OUTPUTS: 
    "predictions"= dataframe of prediciton for all missing values per techinque.
        all techniques explained in the README.
    "metrics"= dataframe metric results of each technique (MAPE, MAE, RMSE, RSQUARED).
    '''
    # number of counties in the complete dataset
    counties = full_data.shape[0]
    # create random test set
    pop_list = np.random.randint(1970, 2017, counties)
    true = [full_data.iloc[i][yr] for i, yr in enumerate(pop_list)]

    full_test = full_data.copy() # make a copy before removing data
    for i, yr in enumerate(pop_list):
        full_test.iloc[i][yr] = np.nan 

    # linear regression preds using only individual county data
    y_pred = []

    for i in range(counties):
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
    # fill in missing value from the previous year
    ffill_df.fillna(method='ffill', axis=1, inplace=True)
    # unless it is the first year in the series, then fill in value from the second year
    ffill_df.fillna(method='bfill', axis=1, inplace=True)
    ffill_preds = [ffill_df.iloc[i][pop_list[i]] for i in range(counties)]

    # fill in missing values (1) with the mean of all years for that specific county
    #yr_mean_df = full_test.copy()
    #yr_mean_df = yr_mean_df.apply(lambda x: x.fillna(x.mean()),axis=1)
    #yr_mean_preds = [yr_mean_df.iloc[i][pop_list[i]] for i in range(counties)]

    # fill in missing values (random num per year) with the mean of associated year 
    county_mean_df = full_test.copy()
    county_mean_df = county_mean_df.apply(lambda x: x.fillna(x.mean()),axis=0)
    county_mean_preds = [county_mean_df.iloc[i][pop_list[i]] for i in range(counties)]

    #save baseline and individual county linear regression predicitions 
    predictions = pd.DataFrame({"years":pop_list}, index=full_data.index)
    predictions["y_true"] = true
    predictions["lr_pred"] = y_pred
    predictions["ffill_pred"] = ffill_preds
    #predictions["yr_mean_pred"] = yr_mean_preds
    predictions["county_mean_pred"] = county_mean_preds

    # inverse distance weighting (haversine distance)
    for i in [.25, .5, 1, 2, 4, 8, 16]:
        hav_IDW = hav_dist_df.copy()
        hav_IDW = hav_IDW.applymap(lambda x: x**-i if x != 0 else 0)
        hav_IDW = hav_IDW.apply(lambda x: x/sum(x), axis=1)

        hav_IDW_vals = np.array(hav_IDW).dot(np.array(full_data))

        predictions["exp{}_pred".format(i)] = \
        [hav_IDW_vals[j][pop_list[j]-1970] for j in range(counties)]

    # k Nearest Neighbor preds
    kN1_preds = []
    for i in range(counties):
        n = full_data.iloc[neigh_table.iloc[i][0]][pop_list[i]]
        kN1_preds.append(n)

    kN2_mean_preds, kN2_weighted_preds = \
        k_neighbors(2, pop_list, full_data, neigh_table, neigh_dist)
    kN3_mean_preds, kN3_weighted_preds = \
        k_neighbors(3, pop_list, full_data, neigh_table, neigh_dist)
    kN4_mean_preds, kN4_weighted_preds = \
        k_neighbors(4, pop_list, full_data, neigh_table, neigh_dist)
    kN5_mean_preds, kN5_weighted_preds = \
        k_neighbors(5, pop_list, full_data, neigh_table, neigh_dist)
    kN10_mean_preds, kN10_weighted_preds = \
        k_neighbors(10, pop_list, full_data, neigh_table, neigh_dist)
    _, kN5_exp2weighted_preds = \
        k_neighbors(5, pop_list, full_data, neigh_table, neigh_dist, s=2)
    _, kN10_exp2weighted_preds = \
        k_neighbors(10, pop_list, full_data, neigh_table, neigh_dist, s=2)

    predictions["k1_pred"] = kN1_preds
    predictions["k2_mean_pred"] = kN2_mean_preds
    predictions["k2_weighted_pred"] = kN2_weighted_preds
    predictions["k3_mean_pred"] = kN3_mean_preds
    predictions["k3_weighted_pred"] = kN3_weighted_preds
    predictions["k4_mean_pred"] = kN4_mean_preds
    predictions["k4_weighted_pred"] = kN4_weighted_preds
    predictions["k5_mean_pred"] = kN5_mean_preds
    predictions["k5_weighted_pred"] = kN5_weighted_preds
    predictions["k10_mean_pred"] = kN10_mean_preds
    predictions["k10_weighted_pred"] = kN10_weighted_preds
    predictions["k5_exp2weighted_pred"] = kN5_exp2weighted_preds
    predictions["k10_exp2weighted_pred"] = kN10_exp2weighted_preds
    
    # Look up the missing values in the complete data linear regression tables 
    FULL_LR_preds = [full_LR.iloc[i][pop_list[i]] for i in range(counties)]
    FULL_HAV_LR_preds = [full_HAV_LR.iloc[i][pop_list[i]] for i in range(counties)]
    FULL_HAV_LR_EXP2_preds = [full_HAV_EXP2_LR.iloc[i][pop_list[i]] for i in range(counties)]

    predictions["FULL_LR_pred"] = FULL_LR_preds
    predictions["FULL_HAV_LR_pred"] = FULL_HAV_LR_preds
    predictions["FULL_HAV_LR_EXP2_pred"] = FULL_HAV_LR_EXP2_preds

    pred_cols = [col for col in predictions.columns if "pred" in col]
    MAPE, MAE, R2, RMSE = [], [], [], []

    for col in pred_cols:
        MAPE.append(mape(predictions["y_true"], predictions[col]))
        MAE.append(median_absolute_error(predictions["y_true"], predictions[col]))
        R2.append(r2_score(predictions["y_true"], predictions[col]))
        RMSE.append(np.sqrt(mean_squared_error(predictions["y_true"], predictions[col])))

        
    metrics = pd.DataFrame({"MAPE": MAPE, "MAE":MAE, "RMSE":RMSE, "RSQUARED":R2}, 
                                 index=[col.upper() for col in pred_cols])
        
    return predictions, metrics


# load full data 
full_data = pd.read_csv("../data/ts_corn_47yr_df.csv")
full_data.fips_code = full_data.fips_code.astype(str).apply(lambda x: x.zfill(5))
full_data.set_index('fips_code', inplace=True)
full_data.columns = full_data.columns.astype(int)
# load county centroids (inner join with the data)
centroids_df = pd.read_csv("../data/county_centroids.csv", index_col=0)
centroids_df.fips_code = centroids_df.fips_code.astype(str).apply(lambda x: x.zfill(5))
centroids_df.set_index('fips_code', inplace=True)
full_data_centroids = full_data.join(centroids_df) 
full_spatial = full_data_centroids[['latitude', 'longitude']]
# create haversine distance df (only needs to be done once)
full_hav_df = np.radians(full_spatial)
dist = DistanceMetric.get_metric("haversine")
hav_dist_df = pd.DataFrame(dist.pairwise(full_hav_df))

# load predictions from FULL linear regression models 
# will be used for indexing later
full_LR = pd.read_csv("../data/lin_reg_preds.csv", index_col="fips_code")
full_LR.columns = full_LR.columns.astype(int)

full_HAV_LR = pd.read_csv("../data/lin_reg_haversine_norm_preds.csv", index_col="fips_code")
full_HAV_LR.columns = full_HAV_LR.columns.astype(int)

full_HAV_EXP2_LR = pd.read_csv("../data/lin_reg_hav_norm_exp2_df.csv", index_col="fips_code")
full_HAV_EXP2_LR.columns = full_HAV_EXP2_LR.columns.astype(int)

# load nearest neighbors and distances
neigh_table = pd.read_csv("../data/k10neighbors.csv", index_col=0)
neigh_dist = pd.read_csv("../data/k10_distances.csv", index_col=0)

iterations = 1000

print("Starting {} Bootstrap samples...".format(iterations))
all_predictions, all_metrics = bootstrap(full_data, hav_dist_df, neigh_table, neigh_dist, full_LR, full_HAV_LR, full_HAV_EXP2_LR)

for bs in range(1, iterations):
    if (bs)%10 == 0:
        print("Starting Iteration {}".format(bs))
    predictions, metrics = bootstrap(full_data, hav_dist_df, neigh_table, neigh_dist, full_LR, full_HAV_LR, full_HAV_EXP2_LR)
    all_predictions = all_predictions.join(predictions, rsuffix=bs)  
    all_metrics = all_metrics.join(metrics, rsuffix=bs)
    
print("Bootstrapping complete.")

# save all predictions and metrics 
all_predictions.to_csv("../data/bootstrap_corn_predicitions.csv")
all_metrics.to_csv("../data/bootstrap_corn_metrics.csv")

















