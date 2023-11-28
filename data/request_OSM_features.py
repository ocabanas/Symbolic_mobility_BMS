import shapefile as shp
import pandas as pd
import numpy as np
from math import sqrt, sin, cos, pi, asin
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
import warnings


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


#overpass_url = "http://overpass-api.de/api/interpreter" #online request
overpass_url = 'http://localhost/api/interpreter'        #local request

features_and_subfeatures = {'food_point':['~"amenity"~"restaurant|bar|cafe|fast_food|pub|food_court|ice_cream|marketplace|biergarten"'],
                            'food_poly':['~"amenity"~"restaurant|bar|cafe|fast_food|pub|food_court|ice_cream|marketplace|biergarten"'],
                            'health_point':['~"amenity"~"hospital|clinic|dentist|doctors|nursing_home|pharmacy|veterinary"',
                                           '"healthcare"',
                                           '~"building"~"hospital"',
                                           '~"shop"~"herbalist|nutrition_supplements"'],
                            'health_poly':['~"amenity"~"hospital|clinic|dentist|doctors|nursing_home|pharmacy|veterinary"',
                                           '"healthcare"',
                                           '~"building"~"hospital"',
                                           '~"shop"~"herbalist|nutrition_supplements"'],
                            'industrial_landuse':['"landuse"="industrial"'],
                            'main_road_line':['"highway"'],
                            'natural_landuse':['"landuse"="village_green|greenfield"','"leisure"="garden|park|dog_park"', '"amenity"="grave_yard"','"natural"="wood|scrub|heath|grassland"'],
                            'residential_landuse':['"landuse"="residential"'],
                            'retail_landuse':['"landuse"="retail"'],
                            'commercial_landuse':['"landuse"="commercial"'],
                            'retail_point':['"landuse"="retail"','~"building"~"commercial|kiosk|retail|supermarket"',
                                     '"shop"'],
                            'retail_poly':['"landuse"="retail"','~"building"~"commercial|kiosk|retail|supermarket"',
                                     '"shop"'],
                            'school_point':['~"amenity"~"school|college|university|driving_school|kindergarten|university|library|language_school|music_school"',
                                        '~"building"~"college|school|university|kindergarten"','"landuse"="education"'],
                            'school_poly':['~"amenity"~"school|college|university|driving_school|kindergarten|university|library|language_school|music_school"',
                                        '~"building"~"college|school|university|kindergarten"','"landuse"="education"'],
                            'transport_point':['"aeroway"','~"amenity"~"bicycle_parking|bicycle_rental|boat_rental|bus_station|car_rental|ferry_terminal|motorcycle_parking|parking|taxi"',
                                        '"public_transport"',
                                         '~"building"~"train_station"'],
                            'transport_poly':['"aeroway"','~"amenity"~"bicycle_parking|bicycle_rental|boat_rental|bus_station|car_rental|ferry_terminal|motorcycle_parking|parking|taxi"',
                                        '"public_transport"',
                                         '~"building"~"train_station"'],
                            'entertainment_point':['~"amenity"~"arts_centre|casino|cinema|community_centre|conference_centre|events_venue|gambling|nightclub|planetarium|social_centre|theatre"','~"building"~"civic"','"historic"','"leisure"','"sport"','"tourism"'],
                            'entertainment_poly':['~"amenity"~"arts_centre|casino|cinema|community_centre|conference_centre|events_venue|gambling|nightclub|planetarium|social_centre|theatre"','~"building"~"civic"','"historic"','"leisure"','"sport"','"tourism"'],
                           }


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

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    """
    Context to limit the execution time for the code inside.
    Example: When requestin OSM sometime the code ends stucked.
            In this way we continue the execution after a cetain time
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        
def request_OSM_feature_count(list_features,poly,verbose=False):
    """
    Given a USA state and a city(poligon shape and name), this function returs the number of
    items of a certain feature.
    
    Procedure:
    1-Poligon: We request the feature inside the poligon. 
      If it fails(the poligon is very large) and an error is retured we try the next method.
    2-City Name: Given a city name we request the features asocciated with the city.
    3-Adress city: Some cities do not have features for its name. Instead, the features
      have a propery "addr:city" which is equal to the city name.
     
    Important Note: We do not sum the results of the three methods in order to avoid double count.
    There are excluding methods that are used in case the previous one fails.
    """
    
    count_poly=0
    timeout=1000
    try:
        t=type(poly[0])
        if t!=type(tuple()):
            poly=poly.values[0]
    except:
        poly=poly.values[0]
    # If the request fails (IP error, server error,...) we repeat the requests
    # OVERPASS API request: Features inside a polygon
    body=str()
    for f in list_features:
        body+=f'nwr[{f}](poly:"{" ".join(str(i[1])+" "+str(i[0]) for i in poly)}");'
    #Requesting...
    for niter in range(0,3):
        request=f'[out:json][timeout:{timeout}];({body});out count;'
        time.sleep(2)
        try:
            with time_limit(timeout+10):
                response = requests.get(overpass_url,data=request, timeout=timeout+10)
            #Parsing resonse. Empty 'elelents' is an error, next method.
            
            data = response.json()
            count_poly= int(data['elements'][0]['tags']['total'])
            break
        except Exception as e:
            timeout=timeout*2
            if verbose==True: print(request,e,'iter',niter)
    return count_poly #Number of POIs
def request_OSM_feature_poly(list_features,poly,verbose=False):
    """
    Given a USA state and a city(poligon shape and name), this function returs the number of
    items of a certain feature.
    
    Procedure:
    1-Poligon: We request the feature inside the poligon. 
      If it fails(the poligon is very large) and an error is retured we try the next method.
    2-City Name: Given a city name we request the features asocciated with the city.
    3-Adress city: Some cities do not have features for its name. Instead, the features
      have a propery "addr:city" which is equal to the city name.
     
    Important Note: We do not sum the results of the three methods in order to avoid double count.
    There are excluding methods that are used in case the previous one fails.
    """
    area_poly=0
    timeout=1000
    try:
        t=type(poly[0])
        if t!=type(tuple()):
            poly=poly.values[0]
    except:
        poly=poly.values[0]
    # If the request fails (IP error, server error,...) we repeat the requests
    # OVERPASS API request: Features inside a polygon
    body=str()
    for f in list_features:
        body+=f'nwr[{f}](poly:"{" ".join(str(i[1])+" "+str(i[0]) for i in poly)}");'
    request=f'[out:json][timeout:{timeout}];({body});out geom;'
    #Requesting...
    for niter in range(0,3):
        #print(f'City poly iter{niter}|count:{count_poly}|{count_ct}|{count_name}',end='\r')
        time.sleep(2)
        try:
            with time_limit(timeout+10):
                response = requests.get(overpass_url,data=request, timeout=timeout+10)
            #Parsing resonse. Empty 'elelents' is an error, next method.
            data = response.json()
            if len(data['elements'])>0: break
        except Exception as e:
            if verbose==True: print(request,e,'iter',niter) 
    # Iterate over all elements, build polygon and get area
    try:
        for geom in data['elements']:
            if geom['type']=="relation":
                for subrelation in geom["members"]:
                    if 'geometry' in subrelation.keys():
                        obj = {'type':'Polygon','coordinates':[[[i['lon'],i['lat']] for i in subrelation['geometry']]]}
                        if subrelation["role"]=='outer': area_poly+=area(obj)
                        elif subrelation["role"]=='inner': area_poly=area_poly-area(obj)
            elif 'geometry' in geom.keys():
                obj = {'type':'Polygon','coordinates':[[[i['lon'],i['lat']] for i in geom['geometry']]]}
                area_poly+=area(obj)
    except Exception as e:
        if verbose==True: print(e)
        area_poly=0.
    return area_poly/1e6 #Retutn the area in km^2
def request_OSM_feature_line(list_features,poly,verbose=False):
    """
    Given a USA state and a city(poligon shape and name), this function returs the number of
    items of a certain feature.
    
    Procedure:
    1-Poligon: We request the feature inside the poligon. 
      If it fails(the poligon is very large) and an error is retured we try the next method.
    2-City Name: Given a city name we request the features asocciated with the city.
    3-Adress city: Some cities do not have features for its name. Instead, the features
      have a propery "addr:city" which is equal to the city name.
     
    Important Note: We do not sum the results of the three methods in order to avoid double count.
    There are excluding methods that are used in case the previous one fails.
    """
    line_count=0
    timeout=1000
    try:
        t=type(poly[0])
        if t!=type(tuple()):
            poly=poly.values[0]
    except:
        poly=poly.values[0]
    # If the request fails (IP error, server error,...) we repeat the requests
    # OVERPASS API request: Features inside a polygon
    body=str()
    for f in list_features:
        body+=f'nwr[{f}](poly:"{" ".join(str(i[1])+" "+str(i[0]) for i in poly)}");'
    request=f'[out:json][timeout:{timeout}];({body});out geom;'
    #Requesting...
    for niter in range(0,3):
        #print(f'City poly iter{niter}|count:{count_poly}|{count_ct}|{count_name}',end='\r')
        time.sleep(2)
        try:
            with time_limit(timeout+10):
                response = requests.get(overpass_url,data=request, timeout=timeout+10)
            #Parsing resonse. Empty 'elelents' is an error, next method.
            data = response.json()
            if len(data['elements'])>0: break
        except Exception as e:
            if verbose==True: print(request,e,'iter',niter) 
    # Iterate over all elements, build polygon and get area
    try:
        for geom in data['elements']:
            if geom['type']=='relation' and geom['tags']['type']=='route':
                for subrelation in geom['members']:
                    for i,g in enumerate(subrelation['geometry'][:-2]):
                        line_count+=earth_distance((subrelation['geometry'][i]['lat'],subrelation['geometry'][i]['lon']), (subraltion['geometry'][i+1]['lat'],subrelation['geometry'][i+1]['lon']))
            elif 'geometry' in geom.keys():
                for i,g in enumerate(geom['geometry'][:-2]):
                    line_count+=earth_distance((geom['geometry'][i]['lat'],geom['geometry'][i]['lon']), (geom['geometry'][i+1]['lat'],geom['geometry'][i+1]['lon']))
    except Exception as e:
        if verbose==True: print(e)
        if verbose==True: print(data['elements'])
        if verbose==True: print(body)
    return line_count #Retutn the area in km
def city2ct2coords():
    """
    Compute a dictionary of census tracts and their centroid coordinates
    """
    ids=pd.unique(trips[['geoid_o', 'geoid_d']].values.ravel('K'))
    # Create a dict of census tracts and its coordinates
    def get_latlong(ct):
        """
        Function to acceletate the list compihension
        """
        try:
            row = trips[trips.geoid_o==ct].iloc[0]
            return (row.lat_o,row.lng_o)
        except:
            row = trips[trips.geoid_d==ct].iloc[0]
            return (row.lat_d,row.lng_d)
    ct2coords={ct:get_latlong(ct) for ct in ids}
    return ct2coords
def read_flows_and_census():
    def metadata_dict(row,ct_df_copy):
        """
        To accelerate de code, we put this function inside
        a list comprihension.
        For an input city we return a dictionary with the census tracts, population, land area and the centroid.
        """
        name=row[1].NAME
        poly=[tup for tup in row[1].coords]
        res={}
        res['cts']=[]
        res['population']=0
        res['area']=float(row[1].ALAND)/1e6
        res['centroid']=(row[1].INTPTLAT,row[1].INTPTLON)
        # Poligon that comes from the shapefile of the city
        try:
            polygon= row[1].geometry
        except:
            polygon = Polygon(poly) # Shapely library object
        # Loop for all census tracts and its coordinates
        for ct in ct_df_copy.iterrows():
            point=(float(ct[1].INTPTLON),float(ct[1].INTPTLAT))
            p = Point(point)# Shapely library object
            # If the census tract centroid is inside the city polygon,
            # we assing the census tract as part of the city
            if polygon.contains(p):
                #try:
                    #res['population']+=metadata[metadata.index==return_id_dc(ct[1].GEOID)].Count_Person.values[0]
                res['cts']+=[ct[1].GEOID]
                #print('point',name,len(res['cts']))
                #except:
                #    pass
            else:
                #ct_poly=Polygon([tup[::-1] for tup in ct[1].coords])
                try:
                    ct_poly=ct[1].geometry
                except:
                    ct_poly=Polygon([tup for tup in ct[1].coords])
                try:
                    #print(ct_poly.is_valid,polygon.is_valid)
                    if ct_poly.intersection(polygon).area/float(ct_poly.area) >= 0.25:
                        res['cts']+=[ct[1].GEOID]
                        #print('intersection',name,len(res['cts']))
                except:
                    #print(ct_poly.is_valid,polygon.is_valid)
                    pass
        try:
            res['population']=metadata[metadata.index==return_id_dc(row[1].GEOID)].Count_Person.values[0]
        except:
            res['population']=0
        if res['population']==0 or res['population']==float('NaN'==ct[1].GEOID) or len(res['cts'])==0:
            #print('None:',name,len(res['cts']))
            res={}
            
        else:
            #print('to delete',len(ct_df_copy[ct_df_copy.GEOID.isin(res['cts'])]))
            #print('to delete',len(ct_df_copy.drop(ct_df_copy[~ct_df_copy.GEOID.isin(res['cts'])].index)))
            ct_df_copy=ct_df_copy.drop(ct_df_copy[ct_df_copy.GEOID.isin(res['cts'])].index)
        return res,ct_df_copy
    # Fetching metadata for the GEOIDS
    #ids=pd.unique(trips[['geoid_o', 'geoid_d']].values.ravel('K'))
    valid_ids=[return_id_dc(i) for i in places_df['GEOID'].tolist()]
    # Requesting additional features to DataCommons
    datacommons_features = ['Count_Person']
    metadata = dc.build_multivariate_dataframe(valid_ids,datacommons_features)
    print('Requested census tract population:')
    print(metadata.head())
    print(len(places_df),len(metadata))
    """
    #Dict of census tract and position
    if compute_city2ct2coords==True:
        ct2coords = city2ct2coords()
        write_dict_to_file(ct2coords,folder+"ct2coords")
    else:
        ct2coords = read_dict_from_file(folder+"ct2coords")
    print('Coordinates of CT:',list(ct2coords.keys())[0],ct2coords[list(ct2coords.keys())[0]])
    """
    # Getting census tacts for each location
    city_meta={}
    print('Places from shapefile:')
    print(places_df.head())
    ct_df_copy=deepcopy(ct_df)
    #city_meta = {row[1].NAME:metadata_dict(row) for row in places_df.iterrows()}
    city_meta={}
    for row in places_df.iterrows():
        city_meta[row[1].NAME],ct_df_copy=metadata_dict(row,ct_df_copy)
        #print('Not assigned cts',len(ct_df_copy))
    del ct_df_copy
    # Removing empty values from dict:
    # If a city do not have any census tract, population will be zero
    # We exclude these cities.
    city_meta = {k: v for k, v in city_meta.items() if len(v)>0}
    print('City parameters (cts,population,coordiantes,area):')
    print(list(city_meta.keys())[0],city_meta[list(city_meta.keys())[0]])
    return city_meta
def s_ij(origin,destination):
    d_od=earth_distance(city_meta[origin]['centroid'],city_meta[destination]['centroid'])
    s=0
    for i in [x for x in city_meta.keys() if x not in [origin,destination]]:
        d=earth_distance(city_meta[origin]['centroid'],city_meta[i]['centroid'])
        if d<=d_od:
            s+=float(city_meta[i]['population'])
    return s
def s_k(origin,destination):
    d_od=earth_distance(city_meta[origin]['centroid'],city_meta[destination]['centroid'])
    sk=0
    for i in [x for x in city_meta.keys() if x not in [origin,destination]]:
        d=earth_distance(city_meta[origin]['centroid'],city_meta[i]['centroid'])
        if d<=d_od:
            sk+=float(np.sum(city_trips_features[city_trips_features.name_d==i].total_pop_flow.values))
    return sk
            
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
def earth_distance(lat_lng1, lat_lng2):
    """
    Earth distance, from DeepGravity repository
    """
    lat1, lng1 = [float(l)*pi/180 for l in lat_lng1]
    lat2, lng2 = [float(l)*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    if ds<0.:
        raise Warning('Negative distance')
    return 6371.01 * ds  # spherical earth...
def return_id_dc(geoid):
    """
    Return a string pattern for a given GEOID used to request DataCommons Api and select rows in DataFrame
    """
    return f"geoId/{geoid}"# if len(str(geoid))==11 else f"geoId/0{geoid}"

def scraping_all_usa(base='https://www2.census.gov/geo/tiger/TIGER2019/'):
    # Example scraping all of the zip urls on a page
    from bs4 import BeautifulSoup
    import pandas as pd
    import re
    import requests
    import geopandas as gpd
    import os

    def get_zip(url):
        front_page = requests.get(url,verify=False)
        soup = BeautifulSoup(front_page.content,'html.parser')
        zf = soup.find_all("a",href=re.compile(r"zip"))
        # Maybe should use href 
        zl = [os.path.join(url,i['href']) for i in zf]
        return zl

    base_place = base+'PLACE/'
    base_tract = base+'TRACT/'
    place_list = get_zip(base_place)
    tract_list = get_zip(base_tract)
    
    print(len(place_list))
    print(len(tract_list))

    geo_tract = []
    geo_place = []
    for t,p in zip(tract_list,place_list):
        geo_place.append(gpd.read_file(p))
        geo_tract.append(gpd.read_file(t))

    place_full = pd.concat(geo_place, ignore_index=False).reset_index(drop=True)
    tract_full = pd.concat(geo_tract, ignore_index=False).reset_index(drop=True)
    
    print(len(place_full),len(tract_full))
    
    print(place_full.sample(10))
    print(tract_full.sample(10))
    
    return place_full,tract_full

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
        trips = 'weekly_ct2ct_2019_03_04.csv'
        cities_shp ='Massachusetts/tl_2021_25_place.shp'
        ct_shp='Massachusetts/tl_2021_25_tract.shp'
        usa_state ='Massachusetts'
    elif state=='NY':
        folder='NewYork/'
        trips = 'weekly_ct2ct_2019_03_04.csv'
        cities_shp ='NewYork/tl_2021_36_place.shp'
        ct_shp='NewYork/tl_2021_36_tract.shp'
        usa_state ='New York'
    elif state=='CA':
        folder='California/'
        trips = 'weekly_ct2ct_2019_03_04.csv'
        cities_shp ='California/tl_2021_06_place.shp'
        ct_shp='California/tl_2021_06_tract.shp'
        usa_state ='California'
    elif state=='FL':
        folder='Florida/'
        trips = 'weekly_ct2ct_2019_03_04.csv'
        cities_shp ='Florida/tl_2021_12_place.shp'
        ct_shp='Florida/tl_2021_12_tract.shp'
        usa_state ='Florida'
    elif state=='WA':
        folder='Washington/'
        trips = 'weekly_ct2ct_2019_03_04.csv'
        cities_shp ='Washington/tl_2021_53_place.shp'
        ct_shp='Washington/tl_2021_53_tract.shp'
        usa_state ='Washington'
    elif state=='TX':
        folder='Texas/'
        trips = 'weekly_ct2ct_2019_03_04.csv'
        cities_shp ='Texas/tl_2021_48_place.shp'
        ct_shp='Texas/tl_2021_48_tract.shp'
        usa_state ='Texas'
    elif state=='USA':
        from os.path import exists
        
        if exists('./USA/place.pkl') and exists('./USA/tract.pkl'):
            print('Using shapefiles from folder')
            places_df = pd.read_pickle('./USA/place.pkl')
            ct_df = pd.read_pickle('./USA/tract.pkl')
        else:
            places_df,ct_df = scraping_all_usa()
            places_df.to_pickle('./USA/place.pkl')
            ct_df.to_pickle('./USA/tract.pkl')
        print(ct_df[ct_df.index.duplicated(keep=False)])
        def list_coords(row):
            try:
                return [i for i in row.geometry.exterior.coords]
            except:
                return [point for polygon in row.geometry.geoms for point in polygon.exterior.coords]
            
        places_df['coords']=places_df.apply(lambda row: list_coords(row) , axis=1)
        ct_df['coords']=ct_df.apply(lambda row: list_coords(row) , axis=1)
            
        folder='USA/'
        trips = 'weekly_ct2ct_2019_01_07_merged.csv'
        usa_state ='USA'
        
    else:
        raise ValueError('State not implemented')
    print(usa_state)
    
    
    # Reading files
    # Shape polygon of each city
    if state != 'USA':
        places_shp = shp.Reader(cities_shp)
        places_df = shapefile_to_pandas(places_shp)
        ct_shp = shp.Reader(ct_shp)
        ct_df = shapefile_to_pandas(ct_shp)
    print(places_df.head())
    print(ct_df.head())
    duplicated=places_df[places_df.duplicated(subset=['NAME'],keep=False)]
    if len(duplicated)>0:
        warnings.warn('Duplicated city names')
        places_df.drop_duplicates(subset=['NAME'],inplace=True)
    #Reading census tract flows
    trips = pd.read_csv(trips)
    # Cities, census tracts and positions
    if compute_meta == True:
        city_meta = read_flows_and_census()
        #Saving dicts
        write_dict_to_file(city_meta,folder+"city2meta")
    else:
        # Rescue dict from file
        city_meta = read_dict_from_file(folder+"city2meta")
    
    # Requesting osm data
    if compute_osm==True:
        city_features = pd.DataFrame(columns=['name',*features_and_subfeatures.keys()])
        print('Colums of features DataFrame')
        print(city_features.columns)
        print('Number of cities:',len(places_df))
        print('Number of cities with cts:',len(city_meta.keys()))
        count_timeout=0
        cities_null_features=[]
        len_loop=len(places_df[places_df.NAME.isin(list(city_meta.keys()))])
        count_done=0
        for place in places_df[places_df.NAME.isin(list(city_meta.keys()))].iterrows():
        #for place in places_df.iterrows():
            tmp_feature={}
            for feature in features_and_subfeatures:
                if feature[-4:]=='poly':
                    tmp_feature[feature]=request_OSM_feature_poly(features_and_subfeatures[feature],#Feature list
                                                                    place[1].coords,#City polygon
                                                                 )
                elif feature[-4:]=='line':
                    tmp_feature[feature]=request_OSM_feature_line(features_and_subfeatures[feature],#Feature list
                                                                    place[1].coords,#City polygon
                                                                 )
                else:
                    tmp_feature[feature]=request_OSM_feature_count(features_and_subfeatures[feature],#Feature list
                                                                    place[1].coords,#City polygon
                                                                 )
            # If the city has null counts in all features (error?)
            # We maybe don't want to use the city
            if sum([i for i in tmp_feature.values()])==0:
                #print(tmp_feature)
                cities_null_features+=[place[1].NAME]
            #else: #Uncomment and indent to delete null feature cities from the list
            city_features = pd.concat([city_features,pd.DataFrame({'name':place[1].NAME,
                                                  **tmp_feature},index=[0])],ignore_index=True)
            count_done+=1
            #print(f'OSM features progress: {count_done}/{len_loop}  | Cities w empty features: {len(cities_null_features)}                               ',end='\r')
            print(f'[log {datetime.now().strftime("%Y/%m/%d-%I:%M:%S")}]')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(city_features.tail(1))
        print()
        print('City features from OSM:')
        print(city_features.head())
        print(city_features.tail())
        #Saving orm features DataFrame to pkl
        city_features.to_pickle(folder+'city_features_DG.pkl')
        print('Number of cities with null features:',len(cities_null_features))
    elif compute_osm==False:
        # Read pkl of ofm features
        city_features = pd.read_pickle(folder+'city_features_DG.pkl')
    
    # Double loop over citites: for each combination we compute
    # the total flow and build de DataFrame with trips and features
    city_trips_features = pd.DataFrame(columns=['name_o','name_d',
                                                'total_pop_flow',
                                                'd','s','area_o','area_d',
                                                'm_o','m_d',
                                                *[f'{i}_o' for i in list(features_and_subfeatures.keys())],
                                                *[f'{i}_d' for i in list(features_and_subfeatures.keys())],
                                               ])
    print('Columns of the trips and features DataFrame')
    print(city_trips_features.columns)
    cities_with_cts=list(city_features.name)
    s=pd.Series(cities_with_cts)
    if len(s[s.duplicated(keep=False)])>0:print('duplicated cities',s[s.duplicated(keep=False)]);
        
    len_loop=len(cities_with_cts)
    count_done=0
    print(len_loop)
    for origin in cities_with_cts:
        for destination in [x for x in cities_with_cts if x != origin]:
            #print(city_meta[origin]['cts'],city_meta[destination]['cts'])
            total_flow=np.sum(trips[(trips.geoid_o.isin(list(int(num) for num in city_meta[origin]['cts'])))&
                                    (trips.geoid_d.isin(list(int(num) for num in city_meta[destination]['cts'])))].pop_flows)
            #print('total_flow:',total_flow)
            if total_flow > 0.0:
                new_trip=pd.DataFrame({ 'name_o':origin,'name_d':destination,
                                        'total_pop_flow':total_flow,
                                        'd':earth_distance(city_meta[origin]['centroid'], 
                                                           city_meta[destination]['centroid']),
                                        's':s_ij(origin,destination),
                                        'area_o':city_meta[origin]['area'],'area_d':city_meta[destination]['area'],
                                        'm_o':city_meta[origin]['population'],'m_d':city_meta[destination]['population'],
                                        **{f'{feature}_o': city_features[city_features.name==origin][feature].values[0] 
                                                                   for feature in list(features_and_subfeatures.keys())},
                                        **{f'{feature}_d': city_features[city_features.name==destination][feature].values[0] 
                                                                   for feature in list(features_and_subfeatures.keys())}
                                        },index=[0])
                if len(new_trip)>1:
                    print('####','duplicated?')
                    print(total_flow)
                    print(new_trip)
                city_trips_features = pd.concat([city_trips_features,new_trip],ignore_index=True)
        count_done+=1
        print(f'Trips progress: {count_done}/{len_loop}',end='\r')
    print()
    city_trips_features = city_trips_features.apply(pd.to_numeric, errors='ignore')
    print('Number of duplicated rows',city_trips_features[city_trips_features.duplicated(keep=False)])
    #city_trips_features['sk'] = city_trips_features.apply(lambda x: s_k(x['name_o'],x['name_d']), axis = 1)
    print('Trips between cities with origin/destination metadata')
    print(city_trips_features.head())
    print(city_trips_features.tail())            
    city_trips_features.to_pickle(folder+f'city_trips_features_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}.pkl')
