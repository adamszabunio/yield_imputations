# Yield Imputations

### Introduction:
-----------------
- The project explores a variety of techniques to impute missing values. 
- These experiments were conducted during an Internship for [Granular](www.granular.ag) with the goal of increasing the company's model predictions by filling in missing data.


### The Data:
-------------
- Data is queried from [NASS quick stats](https://quickstats.nass.usda.gov/) for two crops, Corn and Soybeans.
- Using a [python script](get_data.py) historical yield data at the county level is queried from 1970 - 2016. 
- Cleaning and EDA (Exploratory Data Analysis) is conducted for both [CORN](corn.ipynb) and [SOYBEANS](soybeans.ipynb)

[Map of Data Completeness](https://s3.amazonaws.com/yieldimputations/counties_chloropleth.html)
