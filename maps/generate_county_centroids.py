from pyspatial.vector import read_layer

url3 = "http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_county_500k.zip"
vl3, vldf3 = read_layer(url3, index="GEOID")

centroids = vl3.centroids(format='DataFrame').reset_index()
centroids.rename(columns={'index':'fips_code', 'x':'longitude', 'y':'latitude'}, inplace=True)

centroids.to_csv("county_centroids.csv")