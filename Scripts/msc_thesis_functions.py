import matplotlib.pyplot as plt
from haversine import haversine, Unit
import numpy as np
from scipy.signal import find_peaks
import cartopy.crs as ccrs
from shapely.geometry import Polygon, Point, MultiPoint, LineString
from cartopy.geodesic import Geodesic
import shapely
import geopandas as gpd
import pandas as pd
import gstools as gs
from pykrige.ok import OrdinaryKriging
from tqdm.notebook import trange, tqdm
from pydlc import dense_lines
from datetime import datetime, timedelta
from thefuzz import process, fuzz
import h5py
from scipy.spatial.distance import cdist
import xarray as xr
import matplotlib.dates as mdates
import datetime as dt
from sklearn.neighbors import BallTree
from geopy import distance
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from pyproj import Transformer
from scipy.constants import speed_of_light


def roll_MP_snow_depth(df, var, window_size):
    rolled = []
    for i in range(len(df)):
        start = max(0, i - window_size // 2)
        end = min(len(df), i + window_size // 2 + 1)
        window = df[var][start:end]
        rolled.append([item for sublist in window for item in sublist])
        
    return pd.Series(rolled, index=df.index)


def get_footprintsize(H, c=speed_of_light, center_frequency=3.5e9, B=6e9, kt=1.5 ,T=0, v=140, n=16, PRF=1953.125):
    '''
    Equation 4, 5 and 7 from IRSNO1B documentation
    
    H: altitude
    c: speed of light in vacuum
    center_frequency: center frequency (5 GHz)
    B: Bandwidth in radians (?), using GHz right now
    kt: 1.5 (side-lobe reduction something)
    T: depth in ice
    
    '''
    
    lambdac = speed_of_light / center_frequency
    # L = np.sqrt(H * lambdac / 2) #SAR aperature length
    L = (n * v) / (PRF) # from Arttu thesis equation 4.6
    
    along_track_resolution = H * np.tan(np.arcsin(lambdac/ (2 * L)))
    along_track_resolution = along_track_resolution / 5 # according to Appendix A2 of Fredensborg-Hansen (2024)
    
    across_track_resolution = 2 * np.sqrt((c * kt)/B * (H + (T)/(np.sqrt(3.15))))
    # across_track_resolution = np.array([200] * len(across_track_resolution))
    
    
    # FRESNEL ZONE, IF SMOOTH (QUASI SPECULAR TARGET) AS E.G. INTERNAL LAYERS (?)
    # across_track_resolution = np.sqrt(2 * lambdac * (H + (T)/(np.sqrt(3.15))))
    
    return across_track_resolution/2, along_track_resolution/2


def perpendicular_line_with_buffer(linestring, p1, across_track_radius, along_track_radius):
    """
    Constructs a rectangle centered at a given point along a LineString.
    
    Args:
        linestring (LineString): The input LineString in meters.
        p1 (tuple): The point (latitude, longitude) where the rectangle is centered.
        across_track_radius (float): The width of the rectangle (in meters).
        along_track_radius (float): The length of the rectangle (in meters).
    
    Returns:
        Polygon: The rectangle as a Polygon.
    """
    # Find the nearest point on the projected LineString to p1
    point_on_line = linestring.interpolate(linestring.project(Point(p1)))

    # Identify the tangent vector of the nearest segment
    coords = np.array(linestring.coords)
    min_distance = float('inf')
    closest_segment = None
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        distance = segment.distance(point_on_line)
        if distance < min_distance:
            min_distance = distance
            closest_segment = coords[i], coords[i + 1]

    # Calculate the tangent vector
    (x1, y1), (x2, y2) = closest_segment
    tangent_vector = np.array([x2 - x1, y2 - y1])
    tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)  # Normalize it

    # Compute the perpendicular vector by rotating the tangent vector by 90 degrees
    perp_vector = np.array([-tangent_vector[1], tangent_vector[0]])
    perp_vector = perp_vector / np.linalg.norm(perp_vector)  # Normalize it

    # Scale the perpendicular vector to half the desired width
    half_width_vector = perp_vector * (across_track_radius)
    half_length_vector = tangent_vector * (along_track_radius)

    # Create the four corners of the rectangle
    mid_point = np.array([point_on_line.x, point_on_line.y])
    corner1 = mid_point + half_width_vector + half_length_vector
    corner2 = mid_point + half_width_vector - half_length_vector
    corner3 = mid_point - half_width_vector - half_length_vector
    corner4 = mid_point - half_width_vector + half_length_vector

    # Create the rectangle as a Polygon
    rectangle = Polygon([corner1, corner2, corner3, corner4, corner1])

    return rectangle


def construct_footprints_theoretical(df, ds, mode='pulse_limited_unfocused'):
    """
    Constructs footprints at x,y of input DataFrame with given along-track and across-track radius 
    
    Args:
        df (pandas DataFrame): The input LineString in meters.
        along_track_radius (float): Radius [m] of the footprint in along-track direction 
        across_track_radius (float):  Radius [m] of the footprint across along-track direction 
    
    Returns:
        The DataFrame indexed from 1:-1, since we use adjacent coordinates to construct the footprint
        The footprints (shapely.Polygon)
    """
    
    if mode == 'pulse_limited_unfocused':
        across_track_radius, along_track_radius = get_footprintsize(H=df['altitude'] - ds.attrs['level_ice_elevation'],
                                                                    B=ds.attrs['bandwidth'],
                                                                    # center_frequency=ds.attrs['center_frequency'], #center frequency is weird. It should be (8 - 2) / 2, I guess
                                                                    v=ds.attrs['velocity'],
                                                                    n=ds.attrs['number_averages'],
                                                                    PRF=ds.attrs['PRF']
                                                                    )
    if mode == 'beam_limited_unfocused':
        across_track_radius, along_track_radius = get_footprintsize(H=df['altitude'] - ds.attrs['level_ice_elevation'],
                                                                    B=ds.attrs['bandwidth'],
                                                                    # center_frequency=ds.attrs['center_frequency'], #center frequency is weird. It should be (8 - 2) / 2, I guess
                                                                    v=ds.attrs['velocity'],
                                                                    n=ds.attrs['number_averages'],
                                                                    PRF=ds.attrs['PRF']
                                                                    )
        across_track_radius = (2 * (df['altitude'] - ds.attrs['level_ice_elevation']) * np.deg2rad(45)/2) / 2
        
    elif mode == 'GSFC':
        across_track_radius = np.array([5] * len(df))
        along_track_radius = np.array([20] * len(df))
        
    
        
    df['across_track_radius'] = across_track_radius
    df['along_track_radius'] = along_track_radius
    
    footprints = []   
    for i in df.index[1:-1]:
        p0 = (df['x'].loc[i-1], df['y'].loc[i-1])
        p1 = (df['x'].loc[i], df['y'].loc[i])
        p2 = (df['x'].loc[i+1], df['y'].loc[i+1])

        rectangle = perpendicular_line_with_buffer(LineString([p0,p2]), p1, across_track_radius[i], along_track_radius[i])
        footprints.append(rectangle)
        
    df = df.loc[df.index[1:-1]]
    df.reset_index(inplace=True, drop=True)
    return df, footprints



def direct_gridding(points, grid_lon, grid_lat):
    """
    Fills a grid using direct gridding by averaging point measurements that fall into each grid cell.
    
    Parameters:
    - points (pd.DataFrame): DataFrame with 'longitude', 'latitude', and 'data' columns.
    - grid_lon (np.ndarray): 2D array of grid longitudes (meshgrid format).
    - grid_lat (np.ndarray): 2D array of grid latitudes (meshgrid format).
    
    Returns:
    - gridded_data (np.ndarray): 2D array of gridded data with averaged values.
    """
    # Calculate the cell size (assumes uniform spacing)
    cell_width = np.abs(grid_lon[0, 1] - grid_lon[0, 0])  # Longitude step
    cell_height = np.abs(grid_lat[1, 0] - grid_lat[0, 0])  # Latitude step
    
    # Initialize grid for data
    gridded_data = np.full(grid_lon.shape, np.nan)
    
    # Loop over each grid cell
    for i in range(grid_lon.shape[0]):
        for j in range(grid_lon.shape[1]):
            # Define cell boundaries
            lon_min = grid_lon[i, j] - cell_width / 2
            lon_max = grid_lon[i, j] + cell_width / 2
            lat_min = grid_lat[i, j] - cell_height / 2
            lat_max = grid_lat[i, j] + cell_height / 2
            
            # Find points within the cell
            in_cell = (
                (points['lon'] >= lon_min) & (points['lon'] < lon_max) &
                (points['lat'] >= lat_min) & (points['lat'] < lat_max)
            )
            
            # Average data if points are found
            if in_cell.any():
                gridded_data[i, j] = points.loc[in_cell, 'elev'].mean()
    
    return gridded_data


def grid_data(points, stepsize, return_grid=False, method='linear'):
    """
    Combines direct gridding and restricted interpolation using convex hull to limit extrapolation.
    
    Parameters:
    - points (pd.DataFrame): DataFrame with 'longitude', 'latitude', and 'data' columns.
    - method (str): Interpolation method ('linear', 'nearest', 'cubic').
    
    Returns:
    - gridded_data (np.ndarray): 2D array of gridded data with interpolated values within the convex hull.
    """
    
    #Step 0: Construct grid
    extent = get_extent_latlon(points['lon'],points['lat'])
    grid_lon, grid_lat = grid_in_extent(extent, stepsize)
    
    # Step 1: Direct gridding
    gridded_data = direct_gridding(points, grid_lon, grid_lat)
    
    # Step 2: Convex Hull of measurement points
    convex_hull = measurement_bounds_with_dist(3, points['lon'],points['lat'])
    
    # Step 3: Interpolation within convex hull
    nan_mask = np.isnan(gridded_data)

    if np.any(nan_mask):
        # Flatten grid for interpolation
        grid_coords = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))
        grid_coords_list = list(zip(grid_lon.ravel()[nan_mask.ravel()].tolist(), grid_lat.ravel()[nan_mask.ravel()].tolist()))
        
        # Filter known points within convex hull
        valid_points = points_in_single_poly(convex_hull,grid_coords_list)[0]
        print('valid_points, check')
        
        # Interpolate only within the convex hull
        interpolated_values = griddata(
            points=grid_coords[~nan_mask.ravel()],  # Coordinates of known data
            values=gridded_data.ravel()[~nan_mask.ravel()], # Values at those points
            xi=grid_coords[nan_mask.ravel()][valid_points], # Coordinates to interpolate
            method='linear'        # Interpolation method
        )
        
        # Assign interpolated values to NaN cells within convex hull
        gridded_data_tmp = gridded_data.copy().ravel()
        idx = np.arange(len(gridded_data_tmp))[nan_mask.ravel()][valid_points]
        gridded_data_tmp[idx] = interpolated_values
        gridded_data_tmp = gridded_data_tmp.reshape(np.shape(gridded_data))
    if return_grid == True:
        return gridded_data_tmp, grid_lon, grid_lat
    else:
        return gridded_data_tmp


def points_in_single_poly(polygon, points):
    
    '''
    Function to find the indices of points located within each polygon from a list of polygons.

    Inputs:
    polygons: List of shapely.Polygon objects where points shall be checked.
    points: List of (x, y) tuples representing the points to check.
    min_points: Minimum number of points required within a polygon to include it in the output (default is 1).

    Returns:
    poly_points_indices: Dictionary where each key is the index of the polygon in the input list,
                         and the value is a list of indices of the points located within that polygon.
                         Only includes entries with at least min_points points.
    '''
    
    # Convert polygons to a GeoDataFrame
    gdf_poly = gpd.GeoDataFrame({'geometry': polygon if type(polygon)==list else [polygon]}, geometry='geometry')

    # Convert points to a GeoDataFrame
    df_points = pd.DataFrame()
    df_points['points'] = points
    df_points['points'] = df_points['points'].apply(Point)
    gdf_points = gpd.GeoDataFrame(df_points, geometry='points')

    # Create a dictionary to store indices of points within each polygon
    poly_points_indices = {}

    # Loop through each polygon and find indices of points within it
    for i, poly in gdf_poly.iterrows():
        # Spatial join between the current polygon and all points
        sjoin = gpd.tools.sjoin(gdf_points, gpd.GeoDataFrame([poly], geometry='geometry'), predicate="within", how='inner')
        
        # Store the indices of points that are within this polygon
        poly_points_indices[i] = sjoin.index.tolist()

    # Remove entries with fewer than min_points points
    poly_points_indices = {k: v for k, v in poly_points_indices.items()}

    return poly_points_indices



def points_in_poly_list(polygons, points, min_points=1):
    '''
    Function to find the indices of points located within each polygon from a list of polygons.

    Inputs:
    polygons: List of shapely.Polygon objects where points shall be checked.
    points: List of (x, y) tuples representing the points to check.
    min_points: Minimum number of points required within a polygon to include it in the output (default is 1).

    Returns:
    poly_points_indices: Dictionary where each key is the index of the polygon in the input list,
                         and the value is a list of indices of the points located within that polygon.
                         Only includes entries with at least min_points points.
    '''
    
    # Convert polygons to a GeoDataFrame
    gdf_poly = gpd.GeoDataFrame({'geometry': polygons}, geometry='geometry')

    # Convert points to a GeoDataFrame
    df_points = pd.DataFrame()
    df_points['points'] = points
    df_points['points'] = df_points['points'].apply(Point)
    gdf_points = gpd.GeoDataFrame(df_points, geometry='points')

    # Create a dictionary to store indices of points within each polygon
    poly_points_indices = {}

    # Loop through each polygon and find indices of points within it
    for i, poly in gdf_poly.iterrows():
        # Spatial join between the current polygon and all points
        sjoin = gpd.tools.sjoin(gdf_points, gpd.GeoDataFrame([poly], geometry='geometry'), predicate="within", how='inner')
        
        # Store the indices of points that are within this polygon
        poly_points_indices[i] = sjoin.index.tolist()

    # Remove entries with fewer than min_points points
    poly_points_indices = {k: v for k, v in poly_points_indices.items() if len(v) >= min_points}

    return poly_points_indices


#FIND CLOSEST POINT IN DF2 FROM LONLAT IN DF1
def generate_balltree(df):
    '''
        Generate Balltree using customize distance (i.e. Geodesic distance)
    '''
    # return  BallTree(df[['lat', 'lon']].values, metric=lambda u, v: distance.distance(u, v).meters) #geodesic distance
    return  BallTree(df[['lat', 'lon']].values) #euclidian distance

def find_matches(tree, df):
    '''
        Find closest matches in df to items in tree
    '''
    distances, indices = tree.query(df[['lat', 'lon']].values, k = 1)
    df['min_dist'] = distances
    df['min_loc'] = indices
    return df


def pClosest(points, point, K):
    points  = np.array(points)
    # points.sort(key = lambda K: (K[0]-point[0])**2 + (K[1]-point[1])**2)   
    squared_distances = (points[:, 0] - point[0])**2 + (points[:, 1] - point[1])**2
    sorted_indices = np.argsort(squared_distances)

    return sorted_indices[:K]

def flatten(xss):
    return [x for xs in xss for x in xs]


def latlon_polygon_area(geom, radius = 6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    
    from https://stackoverflow.com/a/61184491/6615512
    """
    if geom.geom_type not in ['MultiPolygon','Polygon']:
        return np.nan

    # For MultiPolygon do each separately
    if geom.geom_type=='MultiPolygon':
        return np.sum([latlon_polygon_area(p) for p in geom.geoms])

    # parse out interior rings when present. These are "holes" in polygons.
    if len(geom.interiors)>0:
        interior_area = np.sum([latlon_polygon_area(Polygon(g)) for g in geom.interiors])
        geom = Polygon(geom.exterior)
    else:
        interior_area = 0
        
    # Convert shapely polygon to a 2 column numpy array of lat/lon coordinates.
    geom = np.array(geom.boundary.coords)

    lats = np.deg2rad(geom[:,1])
    lons = np.deg2rad(geom[:,0])

    # Line integral based on Green's Theorem, assumes spherical Earth

    #close polygon
    if lats[0]!=lats[-1]:
        lats = np.append(lats, lats[0])
        lons = np.append(lons, lons[0])

    #colatitudes relative to (0,0)
    a = np.sin(lats/2)**2 + np.cos(lats)* np.sin(lons/2)**2
    colat = 2*np.arctan2( np.sqrt(a), np.sqrt(1-a) )

    #azimuths relative to (0,0)
    az = np.arctan2(np.cos(lats) * np.sin(lons), np.sin(lats)) % (2*np.pi)

    # Calculate diffs
    # daz = np.diff(az) % (2*pi)
    daz = np.diff(az)
    daz = (daz + np.pi) % (2 * np.pi) - np.pi

    deltas=np.diff(colat)/2
    colat=colat[0:-1]+deltas

    # Perform integral
    integrands = (1-np.cos(colat)) * daz

    # Integrate 
    area = abs(sum(integrands))/(4*np.pi)

    area = min(area,1-area)
    if radius is not None: #return in units of radius
        return (area * 4*np.pi*radius**2) - interior_area
    else: #return in ratio of sphere total area 
        return area - interior_area



def coarsen_data(LON, LAT, var, mask, factor):
    z_reshaped = np.zeros(np.shape(LON))
    z_reshaped[:] = np.nan
    z_reshaped[~mask] = var
    xarr = xr.DataArray(z_reshaped.T, [('lon', LON[0,:]), ('lat', LAT[:,0])])

    xarr2 = xarr.coarsen(lon=factor,lat=factor,boundary='trim').mean()
    lon = xarr2['lon'].data.ravel()
    lat = xarr2['lat'].data.ravel()

    LAT2, LON2 = np.meshgrid(lat,lon)
    var = xarr2.data
    mask = ~np.isnan(var)

    lon = LON2[mask].ravel()
    lat = LAT2[mask].ravel()
    var = xarr2.data[mask].ravel()
    return lon, lat, var



def distance_between_datapoints(lon, lat):
    dists = []
    for i in range(len(lon)-1):
        dists.append(haversine((lat[i],lon[i]), (lat[i+1],lon[i+1]), unit=Unit.METERS, normalize=True))

    dists.append(np.mean(dists))
    return dists


def ordering_latlon(df):
    bounds = get_extent_latlon(df['lon'],df['lat'])
    idx = df.loc[(df['lat'] == bounds[1])].index[0]

    reind = np.arange(0,len(df))
    reind = np.delete(reind, idx)
    reind = np.insert(reind, 0, idx)
    df = df.reindex(reind)
    df.reset_index(inplace=True, drop=True)

    dists = cdist(list(zip(df['lon'],df['lat'])),list(zip(df['lon'],df['lat'])))
    mask = np.zeros(len(df['lon'])).astype(bool)
    tst = []
    i = 0
    for j in range(len(mask)-1):
        mask[i] = True
        dist_zero = dists[0,:]
        dist_zero[mask] = np.nan
        row = dists[:,i].copy() + dist_zero
        i = np.nanargmin(row)
        tst.append(i)
    df = df.loc[tst]
    df.reset_index(inplace=True, drop=True)
    return df

def strtime_to_datetime(df):
    try:
        df['time'] = df['time'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M') )
        
    except:
        df['time'] = clean_time_col(df['time'])
        df['time'] = df['time'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M') )

    return df 


def clean_time_col(time): 
    weird_indices = time.apply(lambda x: len(x) > 16) #length of weird timestamps is higher
    time.loc[weird_indices] = np.nan
    time = time.ffill()
    return time

def open_data(path, filetype, mode, instrument):
    if mode == 'dict':
        df_dict = {}
        if filetype == 'csv' or filetype=='txt':
            if instrument == 'MP':
                print(path)
                lon_col = detect_cols(path, keywords=['lon','Longitude'])
                lat_col = detect_cols(path, keywords=['lat', 'Latitude'])
                time_col = detect_cols(path, keywords=['timestamp', 'datetime'])
                var_col = detect_cols(path, keywords=['snow_depth', 'MagnaProbe','depth'])
                site_col = detect_cols(path, keywords=['site','site_id'])
                type_col = detect_cols(path, keywords=['type','ice_type'])

                cols = []
                cols.append(lon_col) if lon_col != None else None
                cols.append(lat_col) if lat_col != None else None
                cols.append(time_col) if time_col != None else None
                cols.append(var_col) if var_col != None else None
                cols.append(site_col) if site_col != None else None
                cols.append(type_col) if type_col != None else None

                names = []
                names.append('lon') if lon_col != None else None
                names.append('lat') if lat_col != None else None
                names.append('time') if time_col != None else None
                names.append('snow_depth') if var_col != None else None
                names.append("site_id") if site_col != None else None
                names.append("ice_type") if type_col != None else None

                df_import = pd.DataFrame({'names':names}, index=cols)

                df_import.sort_index(inplace=True)
                df = pd.read_csv(path, skiprows=1,  usecols=list(df_import.index), names=df_import['names'])
                if np.mean(df['snow_depth'] > 5): #convert [cm] to [m] in snowdepth
                    df['snow_depth'] /= 100
                df.loc[df['lon'] < 0, 'lon'] += 360
                
                df = strtime_to_datetime(df)
                
                if site_col != None:
                    df_dict = split_data(df, df_dict)
                    
                else:
                    df_dict['1'] = df

        elif filetype == 'h5':
            #This could (should) be more flexible to detect data fields etc.
            #Since we only (atm) use 1 h5 file, it is tailored for this file
            f = h5py.File(path, 'r')
            group = f['eureka_data']
            data = group['magnaprobe']
            df = pd.DataFrame({'lat':data['latitude'][()], 'lon':data['longitude'][()], 'snow_depth':data['snow_depth'][()], 'site_id':data['site_id'][()]})
            df.loc[df['lon'] < 0, 'lon'] += 360
            df_dict = split_data(df, df_dict)

        return df_dict

    elif mode == 'df':
        
        if filetype == 'csv' or filetype=='txt':
            if instrument == 'OIB':
                df = pd.read_csv(path, usecols=[0,1,2,7,8,15,16])
                df['date'] = df['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
                df['datetime'] = df['date'] + df['elapsed'].apply(lambda x: timedelta(seconds=x))
                df = df[df['snow_depth'] != -99999]
                del df['date'], df['elapsed']
                
            elif instrument == 'ATM':
                df = pd.read_csv(path, index_col=0)
                
        return df


def split_data(df, df_dict):
    for uniq in df['site_id'].unique():
        df_tmp = df[df['site_id'] == uniq]
        if len(df_tmp) > 10000:
            df_tmp = ordering_latlon(df_tmp)
            df_tmp.dropna(inplace=True)

            num_steps = int(np.ceil(len(df_tmp) / 3000))
            steps = np.linspace(0,len(df_tmp), num_steps).astype(int)
            for i in range(len(steps)-1):
                df_tmp_part = df_tmp.loc[steps[i]:steps[i+1]]
                df_dict[f'{uniq}.{i}'] = df_tmp_part

        else:
            df_dict[str(uniq)] = df_tmp

    return df_dict


def detect_cols(path, keywords):

    df_test = pd.read_csv(path)
    cols = list(df_test.columns)
    scores = np.zeros(len(cols),dtype='int')
    df_scores = pd.DataFrame({'scores':scores},index=cols)
    dt = np.dtype(np.str_,np.int_)

    for key in keywords:
        fuzzy = process.extract(key, cols, limit=len(cols)) 
        inds = np.array(fuzzy,dtype=dt)[:,0]
        scrs = pd.Series(np.array(fuzzy,dtype=dt)[:,1].astype(int), index=inds)
        df_scores['scores'] += scrs

        if True in (scrs > 80).unique():
            tst = df_scores.reset_index()
            col_ind = tst[tst['index'] == (scrs > 80).index[0]].index[0]
            return col_ind

    col_ind = df_scores['scores'].argmax() if df_scores['scores'].max() > len(keywords) * 65 else None
    return col_ind


def random_points_in_bounds(polygon, number):   
    '''
    Sample random locations within a polygons bounds.
    '''
    minx, miny, maxx, maxy = polygon.bounds
    x = np.random.uniform( minx, maxx, number )
    y = np.random.uniform( miny, maxy, number )
    return x, y



def points_in_poly(poly, points, joint_style='left',reset_index=True):
    '''
    Function to find the points that are located within either one or multiple polygons.

    Inputs:
    poly: [shapely.Polygon object] Polygon in which points shall be checked
    points: Points as list(zip(x,y)) which shall be checked
    joint_style: ['inner' or 'left' (default)]

    Returns:
    pnts_in_poly [if joint_style == 'left]: GeoDataFrame with the rows representing the points located
                                             within the polygon
    sjoin [if joint_style == 'inner]: 
    indices [if joint_style == 'inner]:

    '''
    
    gdf_poly = gpd.GeoDataFrame({'geometry': poly if type(poly)==list else [poly]}, geometry='geometry')

    df = pd.DataFrame()
    df['points'] = points
    df['points'] = df['points'].apply(Point)
    gdf_points = gpd.GeoDataFrame(df, geometry='points')

    if joint_style == 'left':
        sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')
        pnts_in_poly = gdf_points[sjoin['index_right']==0.0]
        if reset_index:
            pnts_in_poly.reset_index(inplace=True, drop=True)
        return pnts_in_poly

    elif joint_style == 'inner':
        sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='inner')
        indices = np.sort(pd.unique(sjoin['index_right']))
        return indices, sjoin




def translate_poly_lon(poly):
    '''
    '''

    x,y = poly.exterior.xy
    x = np.array(list(x))
    x[x < 0] += 360
    poly = Polygon(zip(x,y))
    return poly




def construct_oib_footprints_v2(df, along_track_radius, across_track_radius):

    footprints = []   
    for i in df.index[1:-1]:
        p0 = (df['lon'].loc[i-1], df['lat'].loc[i-1])
        p1 = (df['lon'].loc[i], df['lat'].loc[i])
        p2 = (df['lon'].loc[i+1], df['lat'].loc[i+1])

        line = LineString([p0,p2])
        perp, perp_buffer = perpendicular_line_with_buffer(line, p1, across_track_radius, along_track_radius)
        perp_buffer = translate_poly_lon(perp_buffer)
        footprints.append(perp_buffer)
        
    df = df.loc[df.index[1:-1]]
    df.reset_index(inplace=True, drop=True)
    
    return df, footprints


def krigging_vario_estimates_grid(lon, lat, var, model, measurement_boundaries, footprint_areas, reps):
    '''
    This needs documentation!
    '''
    
    models_dict = {}
    z_dict = {}
    ss_dict = {}
    lon_dict = {}
    lat_dict = {}

    model_rescaled = gs.Exponential(dim=2, len_scale=model.len_scale, var=model.var, nugget=model.nugget, rescale=gs.KM_SCALE*1000, latlon=True)

    OK = OrdinaryKriging(
                lon,lat, var,
                variogram_model=model_rescaled,
                coordinates_type='geographic',
                exact_values=True
    )
    
    num_points = []

    for fp in footprint_areas:
        stepsize = np.sqrt(fp)
        print(f'Footprint area: {fp:.2f}')

        LON, LAT, gridlon, gridlat, krig_mask = grid_in_extent(extent_latlon=get_extent_latlon(lon=list(lon), lat=list(lat)),
                                                           stepsize=stepsize,
                                                           poly_boundaries=measurement_boundaries,
                                                           masked=True,
                                                           return_vecs=True
                                                           )
        
        print(f'Number of grid points: {len(krig_mask[krig_mask==False])}')

        lons = LON[~krig_mask].ravel()
        lats = LAT[~krig_mask].ravel()
        z, ss = OK.execute("points", lons, lats)
        
        z_dict[fp] = z
        ss_dict[fp] = ss
        lon_dict[fp] = lons
        lat_dict[fp] = lats

        models = {}
        for i in range(reps):
            bin_center, vario = gs.vario_estimate((lats, lons), z,
                                                    latlon=True,
                                                    geo_scale=gs.KM_SCALE*1000,
                                                    sampling_size=3000,
                                                    )

            model = gs.Exponential(latlon=True, geo_scale=gs.KM_SCALE*1000)
            model.fit_variogram(bin_center, vario, sill=np.var(z), nugget=False)
            models[i] = model

        x = np.linspace(0,10*stepsize,1000)
        ys = [models[i].variogram(x) for i in models.keys()]
        mean_variogram = np.mean(ys, axis=0)
        model_mean = gs.Exponential(latlon=True, geo_scale=gs.KM_SCALE*1000)
        model_mean.fit_variogram(x, mean_variogram)

        models_dict[fp] = model_mean
        num_points.append(len(lons))


    # x = np.linspace(0,np.sqrt(max(footprint_areas)),1000)
    # ys = [models_dict[i].variogram(x) for i in models_dict.keys()]

    # mean_variogram = np.mean(ys, axis=0)
    # # std_up_variogram = np.mean(ys, axis=0) + np.std(ys, axis=0)
    # # std_down_variogram = np.mean(ys, axis=0) - np.std(ys, axis=0)

    # model_mean = gs.Exponential(latlon=True, geo_scale=gs.KM_SCALE*1000)
    # model_mean.fit_variogram(x, mean_variogram)

    return model_mean, num_points, lons, lats, z, ss, krig_mask


def krigging_vario_estimates_random_points(lon, lat, var, model, measurement_boundaries, reps, return_values=False, return_dl=True, plotting=True):
    '''
    Estimate multiple variograms from a given variogram which is krigged onto random locations within the boundaries
    (based on the length scale). 
    '''

    models_dict = {}
    z_dict = {}
    lon_dict = {}
    lat_dict = {}

    model_rescaled = gs.Exponential(dim=2, len_scale=model.len_scale, var=model.var, nugget=model.nugget, rescale=gs.KM_SCALE*1000, latlon=True)

    for i in range(reps):
                
        OK = OrdinaryKriging(
                lon,lat, var,
                variogram_model=model_rescaled,
                coordinates_type='geographic',
                exact_values=False
                )

        x,y = random_points_in_bounds(measurement_boundaries, int(len(lon)/ 3))
        pnts_in_poly = points_in_poly(poly=measurement_boundaries, points=list(zip(x,y)))

        lon_locs = pnts_in_poly.get_coordinates()['x']
        lat_locs = pnts_in_poly.get_coordinates()['y']

        lon_dict[i] = lon_locs
        lat_dict[i] = lat_locs

        z, ss = OK.execute("points", lon_locs, lat_locs)

        bin_center, vario = gs.vario_estimate((lat_locs, lon_locs), z,
                                                latlon=True,
                                                geo_scale=gs.KM_SCALE*1000,
                                                max_dist=60
                                                )

        model = gs.Exponential(latlon=True, geo_scale=gs.KM_SCALE*1000)
        model.fit_variogram(bin_center, vario, sill=np.var(z))

        models_dict[i] = model
        z_dict[i] = z

    x = np.linspace(0,60,1000)
    ys = [models_dict[i].variogram(x) for i in models_dict.keys()]
    mean_variogram = np.mean(ys, axis=0)
    std_up_variogram = np.mean(ys, axis=0) + np.std(ys, axis=0)
    std_down_variogram = np.mean(ys, axis=0) - np.std(ys, axis=0)

    if plotting:
        fig, ax = plt.subplots()
        dl = dense_lines(ys, x=x, ax=ax, cmap='Blues')
        ax.set_ylim(ymin=0)
        ax.plot(x, mean_variogram,color='red',zorder=100, label='mean', alpha=.6)
        ax.fill_between(x, std_up_variogram,std_down_variogram,color='black',zorder=100, alpha=.1, label='std')

        fig.colorbar(dl, ax=ax, shrink=.8,label='Density')
        ax.legend(loc='lower right')
        ax.set_xlabel('Lag distance h [m]')
        ax.set_ylabel(r'$\gamma (h)$')

    model_mean = gs.Exponential(latlon=True, geo_scale=gs.KM_SCALE*1000)
    model_mean.fit_variogram(x, mean_variogram)

    if (return_values==True and return_dl==False):
        return models_dict, lon_dict, lat_dict, z_dict, model_mean

    elif (return_values==False and return_dl==True):
        return model_mean, ys

    elif (return_values==False and return_dl==False):
        return model_mean




def spatial_average(lon, lat, var, radius):
    '''
    Spatial average with a given radius (of averaging). Behaves kind of like a gaussian blur. 
    Probably quite inefficiently written.
    '''

    averaging_circles = []
    for i in range(len(lon)):
        circ = Polygon(Geodesic().circle(lon[i], lat[i],
                                        radius=radius, 
                                        n_samples=1000
                                        ))

        circ = translate_poly_lon(circ)
        averaging_circles.append(circ)

    averaging_gdf = gpd.GeoDataFrame({'geometry': averaging_circles}, geometry='geometry')
    df = pd.DataFrame({'coords':list(zip(lon,lat)), 'var':var})
    df['coords'] = df['coords'].apply(Point)
    points = gpd.GeoDataFrame(df, geometry='coords')

    pointsInAverage = gpd.tools.sjoin(points, averaging_gdf, predicate="within", how='inner')
    indices = np.sort(pd.unique(pointsInAverage['index_right']))
    pointsInAverage.reset_index(inplace=True, drop=True)
    avgs = []
    for ind in indices:
        snw_dpth = pointsInAverage[pointsInAverage['index_right'] == ind]["var"].mean()
        avgs.append(snw_dpth)

    return avgs


def compute_h50(lon, lat, var, radius):
    '''
    Spatial average with a given radius (of averaging). Behaves kind of like a gaussian blur. 
    Probably quite inefficiently written.
    '''

    averaging_circles = []
    for i in range(len(lon)):
        circ = Polygon(Geodesic().circle(lon[i], lat[i],
                                        radius=radius, 
                                        n_samples=1000
                                        ))

        circ = translate_poly_lon(circ)
        averaging_circles.append(circ)

    averaging_gdf = gpd.GeoDataFrame({'geometry': averaging_circles}, geometry='geometry')
    df = pd.DataFrame({'coords':list(zip(lon,lat)), 'var':var})
    df['coords'] = df['coords'].apply(Point)
    points = gpd.GeoDataFrame(df, geometry='coords')

    pointsInAverage = gpd.tools.sjoin(points, averaging_gdf, predicate="within", how='inner')
    indices = np.sort(pd.unique(pointsInAverage['index_right']))
    pointsInAverage.reset_index(inplace=True, drop=True)
    values = []
    num_values = []
    for ind in indices: #95 - 5
        htopo = np.percentile(pointsInAverage[pointsInAverage['index_right'] == ind]["var"], 95) - np.percentile(pointsInAverage[pointsInAverage['index_right'] == ind]["var"], 5)
        values.append(htopo)
        
        num_values.append(len(pointsInAverage[pointsInAverage['index_right'] == ind]["var"]))

    return values, num_values


def measurement_bounds_with_dist(dist, lon, lat):
    '''
    Using a predetermined variogram model (with length scale) to construct a polygon that represents all the area 
    within the extent where the variable (e.g. snow depth) is still auto-correlated 

    Input:
    dist:
    lon:
    lat:

    Returns:

    '''
    circs = []
    for i in range(len(lon)):
        p = (lon[i],lat[i])
        circ = Polygon(Geodesic().circle(p[0],p[1],
                                            radius=dist, 
                                            n_samples=100
                                            ))
        circ = translate_poly_lon(circ)
        circs.append(circ)

    union = shapely.union_all(circs)

    try: 
        areas = []
        for shp in union.geoms:
            areas.append(shp.area)

        maxind = np.argmax(areas)
        union = union.geoms[maxind]
    except:
        pass

    bounds = get_extent_latlon(lon, lat)
    bounds_poly = Polygon.from_bounds(bounds[0],bounds[1],bounds[2],bounds[3])
    intersec = Polygon.intersection(union, bounds_poly)
    return intersec

def grid_in_extent(extent_latlon, stepsize, poly_boundaries=None, masked=False, return_vecs=False):
    '''
    Constructs a np.meshgrid from a given extent (in latlon) and stepsizes in x and y direction (in km)
    
    Inputs:
    extent_latlon: four size tuple (x0, y0, x1, y1), result from function get_extent_latlon, 
    stepsize: Stepsize in m
    return_vecs: [Bool] Whether to return the vector grids from which the meshgrid is constructed
    masked: [Bool] Whether the grid shall be masked within the given poly
    poly_boundaries: [Only used if masked == True], Boundaries of the polygon in which the grid shall be masked.
                     Must be within "extent_latlon" to work.

    Returns:
    xv, yv: np.meshgrid of a grid in the given extent with the given stepsizes in latlon
    gridlon, gridlat: [if return_vecs == True] Axis description of the meshgrid
    mask: [if masked == True] Mask of the grid. Can be used to execute Ordinary Krigging with "masked"
    '''

    minlon = extent_latlon[0]
    minlat = extent_latlon[1]
    maxlon = extent_latlon[2]
    maxlat = extent_latlon[3]

    p1 = (minlat, minlon)
    p2 = (minlat, maxlon)
    p3 = (minlat, minlon)
    p4 = (maxlat, minlon)

    width = haversine(p1,p2, normalize=True, unit=Unit.METERS)
    height = haversine(p3,p4, normalize=True, unit=Unit.METERS)

    num_steps_x = int(width / stepsize)
    num_steps_y = int(height / stepsize)


    grid_lon = np.linspace(minlon, maxlon, int(num_steps_x))
    if minlon > maxlon:
        number_cells1 = num_steps_x * (360 - minlon) / (360 - minlon + maxlon)
        number_cells2 = num_steps_x * (maxlon) / (360 - minlon + maxlon)
        grid_lon1 =  np.linspace(minlon, 360, int(number_cells1))
        grid_lon2 =  np.linspace(0, maxlon, int(number_cells2))
        grid_lon = np.append(grid_lon1, [x for x in grid_lon2])

    grid_lat = np.linspace(minlat, maxlat, int(num_steps_y))

    xv, yv = np.meshgrid(grid_lon, grid_lat)

    if masked:

        pos = np.vstack([xv.ravel(), yv.ravel()])
        pnts_in_poly = points_in_poly(poly_boundaries, list(zip(pos[0,:], pos[1,:])), reset_index=False)

        mask = np.ones(len(pos[0,:]))
        mask[list(pnts_in_poly.index)] = False
        mask = np.reshape(mask, np.shape(xv))
        mask = mask.astype(bool)


    if (return_vecs == True and masked == False):
        return xv, yv, grid_lon, grid_lat

    elif (return_vecs == False and masked == True):
        return xv, yv, mask.T

    elif (return_vecs == True and masked == True):
        return xv, yv, grid_lon, grid_lat, mask

    else:
        return xv, yv



def get_extent_latlon(lon, lat):
    '''
    Calculates the extent of a given list of latlon coordinates. 
    Jumps from 360 deg to 0 deg (or visce versa) in longitude are handled.

    Inputs:
    lon: List of longitude coordinates
    lat: List of latitude coordinates

    Returns:
    extent_latlon: Four point tuple containing the extent: (x0, y0, x1, y2)
    '''
    
    minlat = min(lat)
    maxlat = max(lat)
    minlon = min(lon)
    maxlon = max(lon)

    diff_lon = np.diff(lon)

    if max(abs(diff_lon)) > 100: #find discontinuity in lon
        indices_max,_ = find_peaks(diff_lon, height= 200)
        indices_min,_ = find_peaks(diff_lon*-1, height= 200)
        indices = sorted(np.append(indices_max, [x for x in indices_min]))
        indices.append(len(diff_lon)) 

        offset = np.ones_like(indices)

        if indices[0] in indices_max:
            indices.insert(0,0)
            offset = np.insert(offset,0,0)

        tmp_lon = lon.copy()
        for i in range(0,len(indices)-1,2):
            ind1 = indices[i] + offset[i]
            ind2 = indices[i+1] + 1
            tmp_lon[ind1:ind2] = tmp_lon.iloc[ind1:ind2] + 360
            
        minlon = min(tmp_lon)
        if minlon > 360:
            minlon -= 360
        maxlon = max(tmp_lon)
        if maxlon > 360:
            maxlon -= 360

    extent_latlon = (minlon, minlat, maxlon, maxlat)
    return extent_latlon


def most_populated_cells(field, cutoff):
    '''
    Finds the maximum entries of a field and returns their indices and respective values
    
    Input:
    field: 2D array of e.g. counted datapoints in grid cells 
    cutoff: Value at which the maximums are no longer relevent (could be a absolute or relative aprroach in the future)
    
    Returns:
    indices: Indices of the maximums in the flattened field
    values: Counts of the maximums at the respective indices
    '''
    
    indices = list(reversed(sorted(range(len(field.flatten())), key = lambda sub: field.flatten()[sub])[-len(field.flatten()):]))
    indices = np.array(indices)[list(field.flatten()[indices] > cutoff)] 
    values = field.flatten()[indices] > cutoff
    return indices, values


def scale_bar_left(ax, bars=4, length=None, location=(0.1, 0.05), linewidth=3, col='black'):
    """
    ax is the axes to draw the scalebar on.
    bars is the number of subdivisions of the bar (black and white chunks)
    length is the length of the scalebar in km.
    location is left side of the scalebar in axis coordinates.
    (ie. 0 is the left side of the plot)
    linewidth is the thickness of the scalebar.
    color is the color of the scale bar
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc aligned to the left of the map,
    # vertically at scale bar location
    sbllx = llx0 + (llx1 - llx0) * location[0]
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']:
                return int(x)
            else:
                return scale_number(x - 10 ** ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000 / bars]
    # Plot the scalebar chunks
    barcol = 'white'
    for i in range(0, bars):
        # plot the chunk
        ax.plot(bar_xs, [sby, sby], transform=tmc, color=barcol, linewidth=linewidth)
        # alternate the colour
        if barcol == 'white':
            barcol = 'dimgrey'
        else:
            barcol = 'white'
        # Generate the x coordinate for the number
        bar_xt = sbx + i * length * 1000 / bars
        # Plot the scalebar label for that chunk
        ax.text(bar_xt, sby, str(round(i * length / bars)), transform=tmc,
                horizontalalignment='center', verticalalignment='bottom',
                color=col)
        # work out the position of the next chunk of the bar
        bar_xs[0] = bar_xs[1]
        bar_xs[1] = bar_xs[1] + length * 1000 / bars
    # Generate the x coordinate for the last number
    bar_xt = sbx + length * 1000
    # Plot the last scalebar label
    ax.text(bar_xt, sby, str(round(length)), transform=tmc,
            horizontalalignment='center', verticalalignment='bottom',
            color=col)
    # Plot the unit label below the bar
    bar_xt = sbx + length * 1000 / 2
    bar_yt = y0 + (y1 - y0) * (location[1] / 4)
    ax.text(bar_xt, bar_yt, 'km', transform=tmc, horizontalalignment='center',
            verticalalignment='bottom', color=col)

def variogram_plot(bin_center, vario, counts, model, r2, save_path=None):
    #Variogram plot 
    fig, (ax1, ax2) = plt.subplots(2,1,constrained_layout=True, figsize=(6,4), gridspec_kw={'height_ratios': [1, 2]}, sharex=True)

    ax1.bar(bin_center,counts, align='center', width=(bin_center[1]-bin_center[0]), edgecolor='black', facecolor='silver')
    ax1.set_ylabel("$N$")


    ax2.scatter(bin_center, vario, marker='x',lw=1, color='orange',label='Empirical variogram')
    ax2.set_xlim(left=0)
    xlims = ax2.get_xlim()
    x = np.linspace(xlims[0],xlims[1],100)
    ax2.plot(x, model.variogram(x), label=f'{model.name} fit',color='orange',lw=2)

    ax2.axhline(model.sill, ls='--', color='grey')
    ax2.axvline(model.percentile_scale(0.632), ls='-.', color='orange', label=f'1 - 1/e scale')
    ax2.axvline(model.len_scale, ls='--', color='orange', label='Range')

    ax2.text(0.05, 0.05, f'Range: {model.len_scale:.2f} m\nR$^2$ = {r2:.3f}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform = ax2.transAxes
        )
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Lag class $h$ [m]")
    ax2.set_ylabel(r"$\gamma(h)$")
    ax2.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def bytespdate2num_buoy(b):
    try:
        return mdates.date2num(dt.datetime.strptime(b, '%Y-%m-%dT%H:%M:%S'))
    except:
        return mdates.date2num(dt.datetime.strptime(b, '%Y-%m-%dT%H:%M'))
    
def get_start_line(file_path):
        f = open(file_path, "r")
        data = f.read()
        for i, line in enumerate(data.splitlines()):
            try:
                d=bytespdate2num_buoy(line.split('\t', 1)[0])
                return i
            except:
                continue

