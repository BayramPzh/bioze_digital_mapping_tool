import networkx as nx
import osmnx as ox
import geopandas as gpd
import pandas as pd
import h3
from shapely.ops import nearest_points
from shapely.geometry import Polygon, Point
from utils.cflp_function import store_data_to_pickle

def cell_to_shapely_polygon(h3_index):
    """
    Converts H3 index to Shapely polygons.
    """
    coords = h3.h3_to_geo_boundary(h3_index)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)

def cell_to_shaply_point(h3_index):
    """
    Converts H3 indices to Shapely points (lat, lon). 
    """
    lat, lon = h3.h3_to_geo(h3_index)
    return Point(lon, lat)

def loi_to_gdf(loi):
    """
    Converts a DataFrame with H3 spatial indices to a GeoDataFrame.
    """
    loi['geometry'] = loi['hex9'].apply(cell_to_shaply_point)
    loi_gdf = gpd.GeoDataFrame(loi, geometry='geometry', crs=4326)
    return loi_gdf

def find_closest_osmid(gdf, n):
    """
    Finds the nearest road network node for each candidate site and each farm. 
    """
    gdf['closest_osmid'] = gdf['geometry'].apply(
        lambda location: n.loc[n['geometry'] == nearest_points(location, n.unary_union)[1], 'osmid'].iloc[0])

def calculate_od_matrix(farm_gdf, loi_gdf, cost_per_km=0.69, frequency_per_day=1, lifetime_in_days=1):
    """
    Finds the nearest road network node for each candidate site.
    """
    g = ox.load_graphml('./osm_network/G.graphml') 
    orig = farm_gdf['closest_os'].unique().tolist()
    dest = loi_gdf['closest_os'].unique().tolist()

    # Calculate shortest path between all pair orig (farm) and dest (set of candidate digester sites)
    od_matrix = {origin: {destination: nx.shortest_path_length(g, origin, destination, weight='length') / 1000 for destination in dest} for origin in orig}

    # Create a new nested dictionary with DataFrame indices as keys
    new_nested_dict = {idx: od_matrix[row['closest_os']] for idx, row in farm_gdf.iterrows() if row['closest_os'] in od_matrix}

    # A placeholder that maps digester candidate site index with the index of its closest node
    placeholders = {i:j for i, j in zip(loi_gdf.index.values, loi_gdf['closest_os'])}

    restructured_od = {farm: {index: distances.get(placeholder, None) for index, placeholder in placeholders.items()} for farm, distances in new_nested_dict.items()}

    new_dict = {(digester, farm): distance for farm, digester_distances in restructured_od.items() for digester, distance in digester_distances.items()}
   
    transport_cost = dict(sorted(new_dict.items(), key=lambda x: x[0][0]))

    # Convert from distance to cost
    C = {key: value * cost_per_km * frequency_per_day * lifetime_in_days for key, value in transport_cost.items()}
    plant = loi_gdf.index.tolist()

    return C, plant