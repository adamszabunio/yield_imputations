from numpy import radians
from sklearn.neighbors import BallTree

class K_Spatial_Neighbors:
    '''
    Using latitude and longitude, returns either the k nearest neighbors
    or the neighbors within a specified radius
    inputs:
        coordinates: 
        an array/DataFrame of shape [n, L=2], 
        where L must be passed as latitude, longitude (in that order).
        if passing a DataFrame, use df[['latitude', 'longitude']].values
        constant: 
        default accepts radius queries based on distance in miles
        if choosing to query with kilometers, set constant=6367
    '''
    def __init__(self, coordinates, constant=3958.75):
        self.coordinates = radians(coordinates) # convert to radians for haversine
        self.constant = constant
        self.ball_tree_index = BallTree(self.coordinates, metric='haversine')
        self.srt_dist = []
        
        
    def query_radius(self, query, radius, return_distance=True, sort_results=True):
        '''
        query: 
        an array or Series of shape (2,)
        radius: 
        distance in miles (defualt)
        returns the neighbors (including itself) within a specified radius
        if distances=True and sorting=True, it increases search time.
        However, this will help with identifying which index is the same 
        as the query (indices[0][0])
        NOTE: You cannot set return_distance=False, sort_results=True
        '''
        radius_radian = radius / self.constant 
        indices = self.ball_tree_index.query_radius(X=[query], r=radius_radian,\
                                                    return_distance=return_distance,\
                                                    sort_results=sort_results)     
        return indices[0]
    
    
    def query_k_neighbors(self, query, k, dualtree=True, return_distances=False): 
        '''
        query: 
        an array or Series of shape (2,) 
        returns: 
        indices --> indices[0] are the sorted distances 
        (including distance to self, indices[0][0])
        indices[1] are the sorted k nearest neighbors (including itself)
        NOTE: the first index is the same as the query (and indices[1][0]) 
        from sklearn documentation: 
        if dualtree=True, use the dual tree formalism for the query: a tree is
        built for the query points, and the pair of trees is used to
        efficiently search this space.  This can lead to better
        performance as the number of points grows large.
        '''
        indices = self.ball_tree_index.query(X=[query], k=k+1, dualtree=dualtree)
        if return_distances:
            self.srt_dist.append(indices[0][0][1:] *self.constant)
            
        return indices # indices[0] --> sorted distances, indices[1] --> neighbors
    
    
    def build_neighbors_table(self, k=5, ret_dist=False):
        '''
        builds a table of `k` nearest neighbors for each coordinate pair
        if ret_dist: appends distances to self.srt_dist
        '''
        assert len(self.coordinates) > k 
        neighbors_table = []
        
        for i in self.coordinates:
            q = self.query_k_neighbors(query=i, k=k, return_distances=ret_dist)[1]
            neighbors_table.append(q[0][1:]) # append the k-nearest
         
        return neighbors_table

    
    def build_radius_table(self, radius=100):
        '''
        builds a table of n nearest neighbors for each coordinate pair
        within a specified radius (default = miles)
        '''
        radius_table = []

        for i in self.coordinates:
            q = self.query_radius(query=i, radius=radius)
            radius_table.append(q[0][1:])
        
        return radius_table