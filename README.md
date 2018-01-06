# Yield Imputations

### Introduction:
-----------------
- The project explores a variety of techniques to impute missing values. 
- These experiments were conducted during an Internship for [Granular](www.granular.ag) with the goal of increasing the company's predictive modeling accuracy for datasets with missing data.
- The techniques used include a variety of spatial and temporal methods. (Described below under the "Goal" heading.

### The Data:
-------------
- Data is queried from [NASS quick stats](https://quickstats.nass.usda.gov/) for two crops, Corn and Soybeans.
- Using a [python script](eda/get_data.py) historical yield data is queried at the county level from 1970 - 2016. 
- [Cleaning and EDA (Exploratory Data Analysis)](eda/) is conducted for both [CORN](eda/corn_EDA.ipynb) and [SOYBEANS](eda/soybeans.ipynb)

## Interactive Map of Data Completeness [can be found HERE](https://s3.amazonaws.com/yieldimputations/counties_chloropleth_layers_40.html)

![](images/year_counts.png?raw=true)

- As can be seen from the map and barchart above, complete data is spatialy correlated, and comprises less than a third of the dataset. 

- In this case, it is hard to imagine a scenario where you would need only 47 consecutive years of data and would settle for nothing less. However, if we generalize the problem to a dataset with 47 independent variables (columns) and 2768 samples (rows), a [[2768 X 47]] matrix, then we can approach it as any dataset that we would like to model. 
    - A majority of machine learning models/algorithms depend on complete data.
    - What methods can you think of off the top of your head to replace these values?
----
Heatmaps are a useful tool used for visualizing a range of values within datasets (the color bar on the right of the below heat maps denotes crop yield values). 

Additionally, heatmaps help to visualize where we have missing values.

Taking a closer look at Iowa, we notice there are only 4 missing values (the 4 white pixels for the year of 2015):
![](images/IA_heatmap.png?raw=true)
* fips (Federal Information Processing Standards) codes uniquely identify counties in the United States. Similar to zipcodes. 

Alabama tells another story:
![](images/AL_heatmap.png?raw=true)

And California yet another:
![](images/CA_heatmap.png?raw=true)

At first glance the enitre dataset, doesn't look all that bad. Roughly a quarter of the data is missing. 
![](images/full_heatmap.png?raw=true)
- However, this low estimate is quite deceiving. If our goal is to only include "full" data, i.e. samples with all 47 features (years in this case), then the true figure is much lower than 24%. 
- Considering only complete observations leads to an immense loss of information, ≈70% of this dataset. 


### The Goal: 
-------------
Below we see the additional gain for each year of data imputed. For instance, if we were able to impute only one year of data, we have nearly 10% additional complete data. This increases our 30% of complete data, from 814 counties to 1071 counties, an increase of over 24%!
![](images/cumulative_sum.png?raw=true)

Clearly, it would be beneficial for predictive modeling if we were able to impute these missing values. In this repository, imputation procedures are explored and analyzed.

Baseline and Temporal Methods:
- Mean imputations
- Forward (and Backward) Filling
- Linear Regression (Individual county predictions)

Spatial Methods:
- Weighted Distance Metrics (Inverse Haversine distance)
- Exponentially Smoothed Kernels
- k nearest neighbor imputation
- k averaged neighbors 
- k distance-weighted-average neighbors  

Spatial and Temporal Methods: 
- Linear Regression (Full data predictions)
- Linear Regression (Inverse Haversine distance)
- Exponentially Weighted Linear Regression (Haversine distance)

### Results: 
------------
Work in progress...

- To evaluate the accuracy of the spatial and more advanced temporal methods, the first step was to create Baseline methods.
- The second step was to choose accuracy metrics. 

For baseline methods, I chose three standard methods for interpolating missing data:
+ Forward-filling (and back-filling, when the missing data point was the first (yr 1970)
+ Mean imputation 
    + per county mean (across years; 1970-2016)
    + per year mean (across counties)
+ Vanilla linear regression to interpolate the missing point.

I hypothesized all three of these methods would return poor results. If it doesn't immeadiately stick out to you, this image will help to show  why the per county mean and forward fill methods are bad decisions:

IMAGE DEPICTING LINEAR TREND OF DATA


![](images/all_bootstrap_preds_mae.png?raw=true)

![](images/all_bootstrap_preds_mape.png?raw=true)

![](images/all_bootstrap_preds_rmse.png?raw=true)

![](images/all_bootstrap_preds_r2.png?raw=true)


### Next Steps:
----------------
