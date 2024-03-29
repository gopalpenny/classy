#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:36:36 2022

@author: gopal
"""

import streamlit as st
import pandas as pd
import math
import geopandas as gpd
import ee
import geemap
import os
import sys
import numpy as np
import appmodules.OverviewPageFunctions as opf
import appmodules.ClassifyPageFunctions as cpf


gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research/Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemodules import rs


# %%

    
    
# %%
# import the shapefile to the project directory
def ImportShapefile(sample_locations_dir_path, path_to_shp_import):
    
    region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")
    # st.write('hello world')
    if not os.path.isdir(sample_locations_dir_path): os.mkdir(sample_locations_dir_path)
    if os.path.isfile(region_shp_path):
        st.write('region.shp already exists')
    else:
        region_gpd = gpd.read_file(path_to_shp_import)
        region_gpd.to_file(region_shp_path)
        
    opf.checkProjStatus()
        

def GenerateRandomPts(ic_name, numRandomPts, eeRandomPtsSeed, addcropmask):    
    # %% GENERATE THE SAMPLES IF random_locations.shp DOES NOT EXIST
    if os.path.exists(st.session_state['paths']['random_locations_path']):
        st.warning('random_locations.shp already exists')
        
    else:
        # Initialize earth engine (if necessary)
        try:
            pt_test = ee.Geometry.Point([100, 10])
        except:
            ee.Initialize()
        
        # Import region.shp
        region_shp = gpd.read_file(st.session_state['paths']['region_shp_path'])
        
        # Convert region.shp to ee.FeatureCollection
        region_fc_full = geemap.geopandas_to_ee(region_shp)
        region_fc = region_fc_full.union()
        
        # Get Image Collection for random locations
        ic = ee.ImageCollection(ic_name)
        im = ic.mosaic()
        
        if addcropmask: # GFSAD == 2 represents cropland
            gfsad = ee.Image('users/gopalpenny/GFSAD30')
            im = im.updateMask(gfsad.eq(2))
            
        # Generate the random sample 
        samp_fc = im.sample(
            region = region_fc,
            scale = 10,
            numPixels = numRandomPts,
            seed = eeRandomPtsSeed,
            geometries = True).map(rs.set_feature_id_func('ee_pt_id')).select('ee_pt_id')
        
        # Export the sample
        task = ee.batch.Export.table.toDrive(
            collection = samp_fc,
            description = 'Generating random locations',
            fileNamePrefix = 'random_locations',
            fileFormat = 'SHP',
            folder = st.session_state['paths']['samples_dir_name'])
        
        task.start()
        
        st.success("Sent task to Earth Engine, will save random points to " + st.session_state['paths']['random_locations_path'])
        
# %%

def InitializeSampleLocations():
    print('running InitializeSampleLocations()')
    
    print(st.session_state['app_path'])
    # Generate subdirectories
    proj_path = st.session_state['paths']['proj_path']
    
    random_pts_path = st.session_state['paths']['random_locations_path']
    samples_path = st.session_state['paths']['sample_locations_path']
    
    # %% GENERATE THE SAMPLES IF random_locations.shp DOES NOT EXIST
    if os.path.exists(samples_path):
        st.write('sample_locations.shp already exists')
        
    elif os.path.exists(random_pts_path) != True:
        st.write('random_locations.shp does not yet exist. Need to create it first')
    else:
        # if not os.path.exists(samples_dir_path): 
        #     os.mkdir(samples_dir_path)
        
        random_pts = gpd.read_file(random_pts_path)
        random_pts['loc_id'] = np.arange(random_pts.shape[0])
        random_pts.to_file(random_pts_path)
        
        sample_pts_prep = random_pts
                
        sample_pts_prep['orig_lon'] = random_pts.geometry.x
        sample_pts_prep['orig_lat'] = random_pts.geometry.y
        
        sample_pts_prep['shift_x_m'] = 0
        sample_pts_prep['shift_y_m'] = 0
        sample_pts_prep['loc_set'] = False
        
        
        sample_pts_prep['sample_lon'] = np.nan
        sample_pts_prep['sample_lat'] = np.nan
        
        sample_pts_prep.to_file(samples_path)
    
    st.session_state['class_df'] = cpf.InitializeClassDF()
    opf.checkProjStatus()
    # print(st.session_state['app_path'])
    
    
# %%


def left_shift():
    st.session_state['x_shift'] -= 10 + (st.session_state['shift30m'] * 20)
def right_shift():
    st.session_state['x_shift'] += 10 + (st.session_state['shift30m'] * 20)
def down_shift():
    st.session_state['y_shift'] -= 10 + (st.session_state['shift30m'] * 20)
def up_shift():
    st.session_state['y_shift'] += 10 + (st.session_state['shift30m'] * 20)
    
def set_shift(loc_id, Class, new_class, loc_pt_xy):
    UpdateClassOnly(loc_id, Class,  new_class)
    UpdateSamplePt(loc_id)
    get_pixel_poly(loc_id, 'oli8', loc_pt_xy, 'LANDSAT/LC08/C02/T1_L2', 'SR_B5', buffer_m = 0, vector_type = 'gpd', option = 'local-check')
    get_pixel_poly(loc_id, 's2', loc_pt_xy, 'COPERNICUS/S2', 'B4', buffer_m = 0, vector_type = 'gpd', option = 'local-check')
    
def reset_shift(loc_id):
    st.session_state['x_shift'] = 0
    st.session_state['y_shift'] = 0
    ResetSamplePt(loc_id)
    loc_idx = st.session_state.class_df.loc_id == loc_id
    orig_pt = st.session_state['sample_pts'].loc[loc_idx]
    orig_xy = [float(orig_pt.orig_lon), float(orig_pt.orig_lat)]
    get_pixel_poly(loc_id, 'oli8', orig_xy, 'LANDSAT/LC08/C02/T1_L2', 'SR_B5', buffer_m = 0, vector_type = 'gpd', option = 'local-check')
    get_pixel_poly(loc_id, 's2', orig_xy, 'COPERNICUS/S2', 'B4', buffer_m = 0, vector_type = 'gpd', option = 'local-check')
    
    
# %%

def next_button():
    sample_pts_prep = st.session_state.sample_pts
    current_loc_id = st.session_state.loc_id
    new_locid = sample_pts_prep.loc_id[sample_pts_prep['loc_id'] > current_loc_id].min()
    
    # loc_id is max for filters, then cycle back to beginning
    if np.isnan(new_locid):
        new_locid = sample_pts_prep.loc_id.min()
    st.session_state['loc_id'] = int(new_locid)
    
    
    sample_pts = st.session_state['sample_pts']
    sample_pt = sample_pts[sample_pts.loc_id == new_locid]
    print('######################')
    print(sample_pt)
    new_x_shift = sample_pt['shift_x_m'].iloc[0]
    new_y_shift = sample_pt['shift_y_m'].iloc[0]
    print('np.isnan(new_x_shift)')
    print(type(new_x_shift))
    print(new_x_shift)
    st.session_state['x_shift'] = new_x_shift if not type(new_x_shift) == 'NoneType' else 0
    st.session_state['y_shift'] = new_y_shift if not type(new_y_shift) == 'NoneType' else 0
    
def prev_button():
    sample_pts_prep = st.session_state.sample_pts
    current_loc_id = st.session_state.loc_id
    new_locid = sample_pts_prep.loc_id[sample_pts_prep['loc_id'] < current_loc_id].max()
    
    # loc_id is min for filters, then cycle back to end
    if np.isnan(new_locid):
        new_locid = sample_pts_prep.loc_id.max()
    st.session_state.loc_id = int(new_locid)
    sample_pts = st.session_state['sample_pts']
    sample_pt = sample_pts[sample_pts.loc_id == new_locid]
    print('######################')
    print(sample_pt)
    new_x_shift = sample_pt['shift_x_m'].iloc[0]
    new_y_shift = sample_pt['shift_y_m'].iloc[0]
    print(new_x_shift)
    st.session_state['x_shift'] = new_x_shift if not type(new_x_shift) == 'NoneType' else 0
    st.session_state['y_shift'] = new_y_shift if not type(new_y_shift) == 'NoneType' else 0


def go_to_id(id_to_go):
    st.session_state.loc_id = int(id_to_go)
    
# %%

def UpdateClassOnly(loc_id, Class,  new_class):
    class_path = st.session_state['paths']['class_path']
    loc_idx = st.session_state.class_df.loc_id == loc_id
    print('Class:', Class)
    
    if Class == 'Input new':
        st.session_state.class_df.loc[loc_idx, 'Class'] = new_class
    else:
        st.session_state.class_df.loc[loc_idx, 'Class'] = Class
        
    st.session_state.class_df.to_csv(class_path, index = False)
    
def UpdateSamplePt(loc_id):
    sample_pts_path = st.session_state['paths']['sample_locations_path']
    loc_idx = st.session_state['sample_pts'].loc_id == loc_id
    
    shift_x = st.session_state['x_shift']
    shift_y = st.session_state['y_shift']
    
    # orig_pt used
    orig_pt = st.session_state['sample_pts'].loc[loc_idx]
    # print(orig_pt)
    # print(orig_pt.crs)
    # orig_crs = orig_pt.crs
    orig_pt.geometry = gpd.points_from_xy(orig_pt.orig_lon, orig_pt.orig_lat) #.set_crs(orig_crs)

    # orig_geometry = orig_pt.geometry
    
    new_pt = shift_points_m(orig_pt, shift_x, shift_y)
    new_geometry = new_pt.geometry.iloc[0]
    
    st.session_state['sample_pts'].loc[loc_idx,'shift_x_m'] = shift_x
    st.session_state['sample_pts'].loc[loc_idx,'shift_y_m'] = shift_y
    st.session_state['sample_pts'].loc[loc_idx,'loc_set'] = True
    
    st.session_state['sample_pts'].loc[loc_idx,'sample_lon'] = new_geometry.x
    st.session_state['sample_pts'].loc[loc_idx,'sample_lat'] = new_geometry.y
    st.session_state['sample_pts'].loc[loc_idx,'geometry'] = new_geometry
        
    st.session_state['sample_pts'].to_file(sample_pts_path)
    
def ResetSamplePt(loc_id):
    sample_pts_path = st.session_state['paths']['sample_locations_path']
    loc_idx = st.session_state['sample_pts'].loc_id == loc_id
    
    orig_pt = st.session_state['sample_pts'].loc[loc_idx]
    orig_geometry = gpd.points_from_xy(orig_pt.orig_lon, orig_pt.orig_lat)
    
    st.session_state['sample_pts'].loc[loc_idx,'shift_x_m'] = 0
    st.session_state['sample_pts'].loc[loc_idx,'shift_y_m'] = 0
    st.session_state['sample_pts'].loc[loc_idx,'loc_set'] = False
    
    st.session_state['sample_pts'].loc[loc_idx,'sample_lon'] = np.nan
    st.session_state['sample_pts'].loc[loc_idx,'sample_lat'] = np.nan
    st.session_state['sample_pts'].loc[loc_idx,'geometry'] = orig_geometry
        
    st.session_state['sample_pts'].to_file(sample_pts_path)
    
# %%
def gis_longitude_to_utm_zone(lon):
    """
    Parameters
    ----------
    lon : float
        Longitude.

    Returns
    -------
    utm_zone : int
        UTM zone.

    """
    utm_zone = (math.floor((lon + 180)/6) % 60) + 1
    return utm_zone



def gis_utm_zone_to_proj4(utm_zone):
    """
    Convert UTM Zone to Proj 4 string

    Parameters
    ----------
    utm_zone : Int
        UTM zone as numeric.

    Returns
    -------
    Str
        Proj 4 string for the UTM crs.

    """
    proj4_base = "+proj=utm +zone=UTM_ZONE +datum=WGS84 +units=m +no_defs"
    return proj4_base.replace("UTM_ZONE",str(utm_zone))

def shift_points_m(pts_gpd, xshift_m, yshift_m):
    """Shift points by x, y meters
    

    Parameters
    ----------
    pts_gpd : geopandas points DataFrame
        Points.
    xshift_m : float or int
        Distance to shift points in meters.
    yshift_m : float or in
        Distance to shift points in meters.

    Returns
    -------
    pts_shifted : geopandas pointsDataFrame
        DataFrame with points shifted.

    """
    # xshift_m = 10
    # yshift_m = 10
    # pts_gpd = gdf

    orig_crs = pts_gpd.crs
    
    
    # get longitude
    pts_4326 = pts_gpd.to_crs(4326)
    pts_lon = pts_4326.geometry.x.mean()
    
    # get proj4 string for UTM
    proj4str = gis_utm_zone_to_proj4(gis_longitude_to_utm_zone(pts_lon))
    
    # convert to UTM
    pts_utm_orig = pts_gpd.to_crs(proj4str)
    
    # get x, y to columns
    pts_utm_orig['x'] = pts_utm_orig.geometry.x
    pts_utm_orig['y'] = pts_utm_orig.geometry.y
    
    # drop geometry
    pts_utm_update = pd.DataFrame(pts_utm_orig).drop('geometry', axis = 1)
    
    # adjust x, y columns
    pts_utm_update['xnew'] = pts_utm_update.x + xshift_m
    pts_utm_update['ynew'] = pts_utm_update.y + yshift_m
    
    # convert to geopandas with UTM coordinates
    pts_shifted_utm = gpd.GeoDataFrame(pts_utm_update,  
                                         geometry = gpd.points_from_xy(pts_utm_update.xnew, pts_utm_update.ynew),
                                         crs = proj4str)
    # transform to original crs
    pts_shifted = pts_shifted_utm.to_crs(orig_crs)
    
    return pts_shifted


def get_pixel_poly(loc_id, ic_name, coords_xy, ic_str, band_name, buffer_m = 0, vector_type = 'ee_fc', option = 'earthengine-save'):
    """Download a polygon for the pixel containing coords_xy

    Args:
        loc_id (int): id for the location
        ic_name (str): short name to label the image collection
        coords_xy (list): list of length 2, with x, y coordinates
        ic_str (_type_): earth engine name of the image collection from which to get the pixel poly
        band_name (_type_): earth engine band name to use for the pixel poly
        buffer_m (int, optional): Buffer (m) around which to get pixel polygons. Defaults to 0.
        vector_type (str, optional): gpd for geopandas or ee_fc to get a feature collection. Defaults to 'ee_fc'.
        option (str, optional): 'local-check', 'local-enforce', 'earthengine' or (default) 'earthengine-save' which retrieves and downloads

    Returns:
        gpd: geopandas dataframe with the pixel polygon

        Options:

        - 'local-check': use local file if it exists and pt falls within existing polygon, otherwise download from Earth Engine
        - 'local-enforce': use local file if it exists, do not download from Earth Engine
        - 'earthengine': download from Earth Engine but don't save
        - 'earthengine-save': download from Earth Engine and save to local file
    """
    
    print('Running get_pixel_poly()')

    if not option in ['local-check', 'local-enforce', 'earthengine', 'earthengine-save']:
        raise Exception("option must be 'local-check', 'local-enforce', 'earthengine', or 'earthengine-save'")
    
    px_poly_dir_path = st.session_state['paths']['px_poly_dir_path']
    if os.path.exists(px_poly_dir_path):
        print('px_poly_dir_path exists')
    else:
        print('px_poly_dir_path does not exist')
    
    # print(px_poly_dir_path)
    if not os.path.exists(px_poly_dir_path):
        os.mkdir(px_poly_dir_path)

    pt_xy = gpd.points_from_xy([coords_xy[0]], [coords_xy[1]], crs = 'epsg:4326')
    pt_xy_gpd = gpd.GeoSeries(pt_xy)
        
    loc_px_poly_path = os.path.join(px_poly_dir_path, 'px_poly_' + str(loc_id) + '_' + ic_name + '.shp')


    print(option)
    
    if (os.path.exists(loc_px_poly_path)) and (option in ['local-check', 'local-enforce']):

        # raise Exception('stop')
        px_group_poly = gpd.read_file(loc_px_poly_path)
        # select the pixel poly as the one that contains the point
        px_poly = ([px_group_poly.loc[i:i] for 
                    i in px_group_poly.index 
                    if pt_xy_gpd.within(px_group_poly.loc[i,'geometry'])[0]])
    else:
        # create empty px_poly list to ensure execution of subsequent if statement
        px_poly = []
    
    if len(px_poly) == 0:
        if (os.path.exists(loc_px_poly_path)) and (option == 'local-enforce'):
            # If the local file exists but the point is not within the polygon, get any polygon
            px_poly = ([px_group_poly.loc[i:i] for 
                        i in px_group_poly.index]) 
        else:
            # get the pixel poly from Earth Engine
            print("Getting pixel poly from Earth Engine") 
            px_group_poly = get_ee_pixel_poly(coords_xy, ic_str, band_name, buffer_m, vector_type)
            # select the pixel poly as the one that contains the point
            px_poly = ([px_group_poly.loc[i:i] for 
                        i in px_group_poly.index 
                        if pt_xy_gpd.within(px_group_poly.loc[i,'geometry'])[0]])
            if option in ['local-check','local-enforce','earthengine-save']:
                print('Saving pixel poly to local file')
                px_group_poly.to_file(loc_px_poly_path)

        
    
    # print('px_group_poly')
    # print(px_group_poly)
    # print(pt_xy_gpd)

    return px_poly[0]

def get_ee_pixel_poly(coords_xy, ic_str, band_name, buffer_m = 0, vector_type = 'ee_fc'):
    """Get polygon of pixel containing x, y coordinates
    

    Parameters
    ----------
    coords_xy : list (float)
        Coordinates as x, y locations [lon, lat].
    ic_str : str
        String containing identifier of image collection.
    band_name : str
        Name of band to use for pixel poly.
    vector_type : str, optional
        Determined type of returned object. 'ee_fc' or 'gpd'. The default is 'ee_fc'.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    px_poly : TYPE
        Polygon of pixel containing coords_xy.
        
    ic_str = "LANDSAT/LC08/C02/T1_L2"
    coords_xy = [104.9995, 20.0005]
    landsat_grid_poly = get_ee_pixel_poly(coords_xy, ic_str, 'SR_B5', buffer_m = 60, vector_type = 'gpd')
    """    
    try:
        pt = ee.Geometry.Point(coords_xy)
    except:
        ee.Initialize()
        pt = ee.Geometry.Point(coords_xy)
    
    print('Running get_ee_pixel_poly()')
    # print('pt')
    # print(pt)
    ic = ee.ImageCollection(ic_str)
    ic_im = ic.filterBounds(pt).first().select(band_name)
    # oli8_px_int = oli8_px.select('SR_B5').gt(25000).rename('test')
    px_poly = ic_im.reduceToVectors(
        geometry = pt.buffer(buffer_m, 1),
        scale = ic_im.projection().nominalScale())
    
    if vector_type == 'ee_fc':
        # do nothing -- good to go
        pass
    elif vector_type == 'gpd':
        px_poly = geemap.ee_to_geopandas(px_poly).set_crs(epsg=4326)
    else:
        raise Exception("vector_type must be ee_fc or gpd")
        
    return px_poly