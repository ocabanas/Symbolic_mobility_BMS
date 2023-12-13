import shapefile as shp
import overpy as osm
import pandas as pd
import numpy as np
from math import sqrt, sin, cos, pi, asin
import overpy
import requests
import json
import time
import datacommons_pandas as dc
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pickle
import signal
from contextlib import contextmanager
from datetime import datetime
from area import area
from copy import deepcopy


import sys, getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hs:",["state="])
    except getopt.GetoptError:
        print('test.py -s <state>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <state>')
            sys.exit()
        elif opt in ("-s", "--state"):
            state = arg
    return state

def get_variable(variable):
    valid_ids=[return_id_dc(i) for i in places_df['GEOID'].tolist()]
    # Requesting additional features to DataCommons
    print(valid_ids)
    datacommons_features = [variable]
    return dc.build_multivariate_dataframe(valid_ids,datacommons_features)
    
def write_dict_to_file(d,file_name):
    """
    From python dict to pkl file
    """
    a_file = open(f"{file_name}.pkl", "wb")
    pickle.dump(d, a_file)
    a_file.close()
    return
def read_dict_from_file(file_name):
    """
    From pkl file to python dict
    """
    a_file = open(f"{file_name}.pkl", "rb")
    output = pickle.load(a_file)
    return output
def shapefile_to_pandas(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

def return_id_dc(geoid):
    """
    Return a string pattern for a given GEOID used to request DataCommons Api and select rows in DataFrame
    """
    return f"geoId/{geoid}"# if len(str(geoid))==11 else f"geoId/0{geoid}"

if __name__ == "__main__":
    """
    City/Village/Town Shapefile for each USA state
    
    # https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Places
    
    # https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Census+Tracts
    
    """
    state=main(sys.argv[1:])
    # Definig parameters
    # If True, the objects will be computed and saved (only fist time)
    # If False, the objects will be rescued from the file (very fast)
    compute_meta = True
    compute_city2ct2coords=True
    # If True, the objects will be computed and saved (first time or update of features)
    # If False, the objects will be rescued from the file (very fast)
    compute_osm=True
    if state=='MA':
        folder='Massachusetts/'
        trips = 'weekly_ct2ct_2019_01_07_merged.csv'
        cities_shp ='Massachusetts/tl_2021_25_place.shp'
        ct_shp='Massachusetts/tl_2021_25_tract.shp'
        usa_state ='Massachusetts'
    elif state=='NY':
        folder='NewYork/'
        trips = 'weekly_ct2ct_2019_01_07_merged.csv'
        cities_shp ='NewYork/tl_2021_36_place.shp'
        ct_shp='NewYork/tl_2021_36_tract.shp'
        usa_state ='New York'
    elif state=='CA':
        folder='California/'
        trips = 'weekly_ct2ct_2019_01_07_merged.csv'
        cities_shp ='California/tl_2021_06_place.shp'
        ct_shp='California/tl_2021_06_tract.shp'
        usa_state ='California'
    elif state=='FL':
        folder='Florida/'
        trips = 'weekly_ct2ct_2019_01_07_merged.csv'
        cities_shp ='Florida/tl_2021_12_place.shp'
        ct_shp='Florida/tl_2021_12_tract.shp'
        usa_state ='Florida'
    elif state=='WA':
        folder='Washington/'
        trips = 'weekly_ct2ct_2019_01_07_merged.csv'
        cities_shp ='Washington/tl_2021_53_place.shp'
        ct_shp='Washington/tl_2021_53_tract.shp'
        usa_state ='Washington'
    elif state=='TX':
        folder='Texas/'
        trips = 'weekly_ct2ct_2019_01_07_merged.csv'
        cities_shp ='Texas/tl_2021_48_place.shp'
        ct_shp='Texas/tl_2021_48_tract.shp'
        usa_state ='Texas'

    places_shp = shp.Reader(cities_shp)
    places_df = shapefile_to_pandas(places_shp)
    
    #c=get_variable('Median_Income_Household')
    #print(c)
    
    #city_meta = read_dict_from_file(folder+"city2meta")
    
    dict_={}
    for row in places_df.iterrows():
        try:
            dict_[row[1].NAME]=dc.build_multivariate_dataframe(return_id_dc(row[1].GEOID),['Median_Income_Person'])['Median_Income_Person'].values[0]
        except:
            print(row[1].NAME)#,dc.build_multivariate_dataframe(return_id_dc(row[1].GEOID),['Median_Income_Household']))
    print(dict_)
    write_dict_to_file(dict_,folder+"income")