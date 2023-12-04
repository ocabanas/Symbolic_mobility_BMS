import random
import numpy as np
import pandas as pd
import json
import zipfile
import gzip
import pickle
import torch
import string
import os

import geopandas
from skmob.tessellation import tilers
from math import sqrt, sin, cos, pi, asin

from importlib.machinery import SourceFileLoader

path = './models/deepgravity.py'
ffnn = SourceFileLoader('ffnn', path).load_module()

def _is_support_files_computed(db):
    if os.path.isdir(db+'/processed'):
        base = db + '/processed/'
        return os.path.isfile(base+'tileid2oa2handmade_features.json') and os.path.isfile(base+'oa_gdf.csv.gz') and os.path.isfile(base+'flows_oa.csv.zip') and os.path.isfile(base+'oa2features.pkl') and os.path.isfile(base+'oa2centroid.pkl')
    else:
        return False
    
def _check_base_files(db_dir):
                
    if not (os.path.isfile(db_dir+'/features.pkl')):
        raise ValueError('city_features file is missing')
        
    if not (os.path.isfile(db_dir+'city2meta.pkl')):
            raise ValueError('city2meta file is missing!')
            
    if not (os.path.isfile(db_dir+'flows.csv')):
            raise ValueError('flows.csv is missing')

    print('Flows, features and city2meta have been found....')    

def _compute_support_files(db_dir, tile_id_column, tile_geometry, oa_id_column, oa_geometry, flow_origin_column, flow_destination_column, flow_flows_column):
    
    # first, we check if there are at least the needed files into the base directory. 
    _check_base_files(db_dir)

    if not os.path.isdir(db_dir+'/processed'):
        create_dir = os.system('mkdir %s/processed'%db_dir)
    
    print('Generating the processed files - it may take a while....')
    print('Reading features....')
    
    try:
        features = pd.read_pickle(db_dir+'/features.pkl')
        if not oa_id_column in list(features.columns):
            raise ValueError('Features must be associated with an output area. Please add a column '++' to features.csv')
    except:
        features = None
        print('Running without features. features.csv not found....')
        
    #Takes our city2meta and creates oa_gdf.csv.gz and oa2centroid.pkl in processed
    
    city2meta = pd.read_pickle(db_dir+'city2meta.pkl')
    city2meta = pd.DataFrame.from_dict(city2meta, orient='index').reset_index().drop(columns=['population']).rename(
columns={'index':'geo_code', 'area':'area_km2'})
    city2meta = city2meta.reindex(columns=['geo_code', 'centroid', 'area_km2'])
    city2meta['centroid'] = city2meta.centroid.apply(lambda x : [float(i) for i in x])
    city2meta.to_csv(db_dir+'processed/oa_gdf.csv.gz', index=False)
    
    oa2centroid = {}
    for r, row in city2meta.iterrows():
        oa2centroid[row['geo_code']] = row['centroid']
        
    with open(db_dir+'processed/oa2centroid.pkl','wb') as out:
        pickle.dump(oa2centroid, out)
    
    #Creates the flows.csv file

    flows = pd.read_csv(db_dir+'flows.csv')
    flows.to_csv(db_dir+'/processed/flows_oa.csv.zip', index=False)
    
    #Creates od2flow.pkl file
    
    od2flow = {}
    for i,row in flows.iterrows():
        od2flow[(row['residence'],row['workplace'])] = row['commuters']    
        
    with open(db_dir+'/processed/od2flow.pkl', 'wb') as handle:
        pickle.dump(od2flow, handle)    
        
    #Creates oa2features.pkl and tileid2oa2handmade_features.json
        
    features = pd.read_pickle(db_dir+'features.pkl')  
    
    oa2features = {}
    for c in features.columns:
        if c.split('_')[-1] in ['line', 'landuse']:
            features[c] = features[c].apply(lambda x: np.float64(x))
            
    for i,row in features.iterrows():
        oa2features[row[0]]=list(row[1:].values)
        
    with open(db_dir+'processed/oa2features.pkl','wb') as out:
        pickle.dump(oa2features, out)

    tileid2oa2handmade_features = dict()

    for i,row in features.iterrows():
        if i not in tileid2oa2handmade_features:
            tileid2oa2handmade_features[i] = dict()
            tileid2oa2handmade_features[i][row[oa_id_column]]=dict()
    for i,row in features.iterrows():
        for item in zip(list(row[1:].keys()),row[1:].values):
            tileid2oa2handmade_features[i][row[oa_id_column]][item[0]]=[float(item[1])]

    with open(db_dir+'processed/tileid2oa2handmade_features.json', 'w') as f:
        json.dump(tileid2oa2handmade_features, f)     
    
    print('Processed files generated generated')
    
def tessellation_definition(db_dir,name,size):
    if not (os.path.isfile(db_dir+'/tessellation.shp') or os.path.isfile(db_dir+'/tessellation.geojson')):
        tessellation = tilers.tiler.get("squared", base_shape=name, meters=size)
        tessellation.to_file(db_dir+'/tessellation.shp')
    
def load_data(db_dir, tile_id_column, tile_geometry, oa_id_column, oa_geometry, flow_origin_column, flow_destination_column, flow_flows_column):
    # check if there are the computed information
    if not _is_support_files_computed(db_dir):
        _compute_support_files(db_dir, tile_id_column, tile_geometry, oa_id_column, oa_geometry, flow_origin_column, flow_destination_column, flow_flows_column)
        
    # tileid2oa2features2vals
    with open(db_dir + '/processed/tileid2oa2handmade_features.json') as f:
        tileid2oa2features2vals = json.load(f)

    # oa_gdf
    oa_gdf = pd.read_csv(db_dir + '/processed/oa_gdf.csv.gz', dtype={'geo_code': 'str'})

    # flow_df
    flow_df = pd.read_csv(db_dir + '/processed/flows_oa.csv.zip', \
                          dtype={'residence': 'str', 'workplace': 'str'})
    
    city2meta = pd.read_pickle(db_dir+'city2meta.pkl')
    oa2pop = {o: city2meta[o]['population'] for o in city2meta.keys()}
    
    # oa2features, od2flow, oa2centroid
    with open(db_dir + '/processed/oa2features.pkl', 'rb') as f:
        oa2features = pickle.load(f)

    with open(db_dir + '/processed/od2flow.pkl', 'rb') as f:
        od2flow = pickle.load(f)

    with open(db_dir + '/processed/oa2centroid.pkl', 'rb') as f:
        oa2centroid = pickle.load(f)

    return tileid2oa2features2vals, oa_gdf, flow_df, oa2pop, oa2features, od2flow, oa2centroid


def load_model(fname, oa2centroid, oa2features, oa2pop, device, dim_s=1, \
               distances=None, dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=True):
    loc_id = list(oa2centroid.keys())[0]

    model = ffnn.NN_MultinomialRegression(dim_s, dim_hidden, 'deepgravity',  dropout_p=dropout_p, device=device)
    checkpoint = torch.load(fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def instantiate_model(oa2centroid, oa2features, oa2pop, dim_input, device=torch.device("cpu"), dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=False):

    model = ffnn.NN_MultinomialRegression(dim_input, dim_hidden,  'deepgravity', dropout_p=dropout_p, device=device)

    return model


def earth_distance(lat_lng1, lat_lng2):
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds 