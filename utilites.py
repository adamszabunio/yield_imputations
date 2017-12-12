def get_haversine_dist(spatial_df, constant=3958.75, normalize=False, smooth=False, smooth_exp=None):
    '''
    - returns a haversine pairwise distance matrix --> hav_dist_df
    - four options for hav_dist_df, as is, multiplied by constant, smoothed and normalize, or normed 
    - earth's radius = 3958.75 (in miles), 
    inputs: 
        spatial_df: pd.DataFrame of shape m*n, m = num counties, n = 2 (lat, long)
        dist: None if returning the haversine distance pairwise matrix, normalizing and/or smoothing
        smooth: whether to exponentially smooth distance matrix x**-smooth_exp
        smooth_exp: if smooth: by which exponential factor to smooth 
        normalize: whether to return a normalized copy of the matrix 
    '''
    assert ["latitude", "longitude"] == spatial_df.columns.lower()
    hav_df = spatial_df.copy()
    hav_df.latitude = hav_df.latitude.apply(lambda x : x*np.pi/180)
    hav_df.longitude = hav_df.longitude.apply(lambda x : x*np.pi/180)
    dist_met = DistanceMetric.get_metric("haversine")
    hav_dist_df = pd.DataFrame(dist.pairwise(hav_df))
    
    if constant:
        hav_dist_df *= constant
        
    if smooth:
        hav_dist_df = hav_dist_df.applymap(lambda x: x**-smooth_exp if x != 0 else 0)
    
    if normalize: 
        # returns a matrix for dot product multiplication (with county_yields_df)
        hav_dist_df = hav_dist_df.apply(lambda x: x/sum(x), axis=1) # normalize along the rows 
        # assert we have normalized correctly 
        ones = np.ones(hav_dist_df.shape[0])
        norms = [sum(hav_dist_df.iloc[i]) for i in hav_dist_df.columns.values]
        assert(np.allclose(ones, norms))
    
    return hav_dist_df


def impute(county_yields_df, dist_df, radius=100, k=5):
    
    
    
    
    
    