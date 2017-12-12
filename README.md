# Yield Imputations

### Introduction:
-----------------
- The project explores a variety of techniques to impute missing values. 
- These experiments were conducted during an Internship for [Granular](www.granular.ag) with the goal of increasing the company's model predictions by filling in missing data.


### The Data:
-------------
- Data is queried from [NASS quick stats](https://quickstats.nass.usda.gov/) for two crops, Corn and Soybeans.
- Using a [python script](get_data.py) historical yield data is queried at the county level from 1970 - 2016. 
- Cleaning and EDA (Exploratory Data Analysis) is conducted for both [CORN](corn_EDA.ipynb) and [SOYBEANS](soybeans.ipynb)

[Map of Data Completeness](https://s3.amazonaws.com/yieldimputations/counties_chloropleth_layers_40.html)

![](images/year_counts.png?raw=true)

- As can be seen from the map and barchart above, complete data is spatialy correlated, and comprises less than a third of the dataset. 

- In this case, it is hard to imagine a scenario where you would need only 47 consecutive years of data and would settle for nothing less. However, if we generalize the problem to a dataset with 47 independent variables (columns) and 2768 samples (rows), a $2768\times47$ matrix, then we can approach it as any dataset that we would like to model. 
    - A majority of machine learning models/algorithms depend on complete data.
    - What methods can you think of off the top of your head to replace these values?

Heatmaps are a useful tool to visualize not only the range of values within datasets, but also to see where we have missing values.

Taking a closer look at Iowa, we notice their are only 4 missing values:
![](images/IA_heatmap.png?raw=true)

Alabama tells another story:
![](images/AL_heatmap.png?raw=true)

And California yet another:
![](images/CA_heatmap.png?raw=true)

At first glance the enitre dataset, doesn't look all that bad. Roughly a quarter of the data is missing. 
![](images/full_heatmap.png?raw=true)
However, this low estimate is quite deceiving. If our goal is to only include "full" data, i.e. samples with all 47 features (years in this case), then the true figure is a shocking. Considering only complete observations leads to an immense loss of information, ≈70% of this dataset.


### The Goal: 
-------------

Clearly, methods are required to impute these missing values. In this repository, imputation procedures are explored and analyzed. 

Methods include:
- Mean imputations
- Forward (and Backward) Filling 
- Linear Regression
- Spatial approaches
    - Weighted Distance Metrics
    - Exponentially Smoothed Kernels
    - k nearest neighbors algorithms
        - k averaged neighbors
        - radius 
