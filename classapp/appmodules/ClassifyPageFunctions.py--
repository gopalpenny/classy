#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 08:42:39 2022

@author: gopal
"""

# pageClassify
import streamlit as st
from datetime import datetime, timedelta
import ee
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import math
import appmodules.manclass as mf
import appmodules.DownloadPageFunctions as dpf
import os

# %%



def InitializeClassDF():
    print('running InitializeClassDF()')
    proj_years = st.session_state['proj_vars']['proj_years']
    subclass_years = ['Subclass' + str(y) for y in proj_years]
    # default_subclass_year = 'Subclass' + str(st.session_state['proj_vars']['classification_year_default'])
    
    # Single year in file
    # if os.path.exists(class_path): 
    #     class_df = pd.read_csv(class_path)
    # else:
    #     class_df = pd.DataFrame(loc).drop(['geometry'], axis = 1)
    #     class_df['Class'] = np.nan
    #     class_df['Subclass'] = np.nan
    #     class_df['Year'] = st.session_state['proj_vars']['classification_year_default']
    
    class_path = st.session_state['paths']['class_path']
    random_path = st.session_state['paths']['random_locations_path']
    
    if os.path.exists(class_path): 
        class_df = pd.read_csv(class_path)
        subclass_columns = [col for col in list(class_df.columns) if 'Subclass' in col]
        # print(subclass_columns)
        proj_years_missing = [y for y in subclass_years if y not in subclass_columns]
        
        for i in range(len(proj_years_missing)):
            if proj_years_missing[i] not in list(class_df.columns): # double checking to ensure we don't replace existing column
                class_df[proj_years_missing[i]] = str(np.nan)
            
    elif st.session_state['status']['random_status']:
        loc = gpd.read_file(random_path).to_crs(4326)
        # print(pd.DataFrame(loc['loc_id']))
        class_df = pd.DataFrame({'loc_id':np.arange(loc.shape[0])}) #.drop(['geometry'], axis = 1)
        class_df['Class'] = np.nan
        
        for i in range(len(subclass_years)):
            if subclass_years[i] not in list(class_df.columns): # double checking to ensure we don't replace existing column
                class_df[subclass_years[i]] = str(np.nan)
    else:
        # st.warning(random_path + ' not yet created')
        class_df = None
        
        
    # OLD CODE WHEN I TRIED PUTTING ALL YEARS IN ONE CSV (WAS TOO SLOW)
    # class_df_blank = pd.DataFrame(loc).drop(['geometry'], axis = 1)
    # class_df_blank['Class'] = np.nan
    # class_df_blank['Subclass'] = np.nan
    
    # if os.path.exists(class_path):
    #     class_df = pd.read_csv(class_path)
    #     class_df_years = list(set(class_df['Year']))
        
    #     proj_years_missing = [y for y in proj_years if y not in class_df_years]
    #     for i in range(len(proj_years_missing)):
    #         class_df_blank['Year'] = proj_years_missing[i]
    #         class_df = class_df.append(class_df_blank, ignore_index = True)
        
    # else:
    #     # add section for each year
    #     for i in range(len(proj_years)):
    #         class_df_blank['Year'] = proj_years[i]
    #         if i == 0:
    #             class_df = class_df_blank
    #         else:
    #             class_df = class_df.append(class_df_blank, ignore_index = True)
    #     class_df.to_csv(class_path, index = False)
        
    return class_df

def UpdateClassDF(loc_id, Class, Subclass,  new_class, new_subclass, subclass_year):
    class_path = st.session_state['paths']['class_path']
    loc_idx = st.session_state.class_df.loc_id == loc_id
    
    
    # if subclass_year not in list(st.session_state.class_df.columns): # double checking to ensure we don't replace existing column
    #     st.session_state.class_df[subclass_year] = np.nan
    # year_idx = st.session_state.class_df.Year == year
    # idx = [x & y for (x, y) in zip(loc_idx, year_idx)]
    if Class == 'Input new':
        st.session_state.class_df.loc[loc_idx, 'Class'] = new_class
    else:
        st.session_state.class_df.loc[loc_idx, 'Class'] = Class
        
        
    if Subclass == 'Input new':
        st.session_state.class_df.loc[loc_idx, subclass_year] = new_subclass
    else:
        st.session_state.class_df.loc[loc_idx, subclass_year] = Subclass
        
    st.session_state.class_df.to_csv(class_path, index = False)
    
    allpts = build_allpts(st.session_state['paths']['proj_path'])
    st.session_state['allpts'] = allpts
    filterargs = st.session_state['filterargs']
    apply_filter(lat_range = filterargs['lat'], 
                lon_range = filterargs['lon'], 
                class_type = filterargs['Class'], 
                subclass_type = filterargs['Subclass'], 
                downloaded = filterargs['Downloaded'])

    

# %%

def build_allpts(proj_path):
    loc = gpd.read_file(st.session_state['paths']['sample_locations_path']).to_crs(4326)
    ts_status_path = dpf.TimeseriesStatusInit(proj_path)
    ts_status = pd.read_csv(ts_status_path)[['loc_id','allcomplete']]
    allpts = pd.merge(loc, ts_status, 'outer', on = 'loc_id').merge(st.session_state['class_df'], on = 'loc_id')
    allpts['Downloaded'] = pd.Categorical(allpts.allcomplete, categories = [False, True])
    allpts['Downloaded'] = allpts.Downloaded.cat.rename_categories(['No','Yes'])
    allpts['lat'] = allpts.geometry.y
    allpts['lon'] = allpts.geometry.x
    
    return allpts

# %%

def apply_filter(lat_range, lon_range, class_type, subclass_type, downloaded):
    
    st.session_state['filterargs'] = {
        'lon' : lon_range,
        'lat' : lat_range,
        'Class' : class_type,
        'Subclass' : subclass_type,
        'Downloaded' : 'Yes'
        }
    
    filterpts = st.session_state['allpts']
    
    # class
    if class_type != 'Any':
        filterpts = filterpts[filterpts['Class'] == class_type]
        
    # subclass
    if subclass_type != 'Any':
        filterpts = filterpts[filterpts[st.session_state['subclass_year']] == subclass_type]
        
        
    # filter for downloaded points
    if subclass_type != 'All':
        filterpts = filterpts[filterpts['Downloaded'] == downloaded]
    
    # latitude
    filterpts = filterpts[filterpts['lat'] >= lat_range[0]]
    filterpts = filterpts[filterpts['lat'] <= lat_range[1]]
    # print(filterpts)
    # longitude
    filterpts = filterpts[filterpts['lon'] >= lon_range[0]]
    filterpts = filterpts[filterpts['lon'] <= lon_range[1]]
    st.session_state['class_df_filter'] = filterpts
    
    
def clear_filter(lat_min, lat_max, lon_min, lon_max):
    filterpts = st.session_state['allpts']
    st.session_state['filterargs'] = {
        'lon' : [lon_min, lon_max],
        'lat' : [lat_min, lat_max],
        'Class' : 'Any',
        'Subclass' : 'Any',
        'Downloaded' : 'Yes'
        }
    st.session_state.class_df_filter = filterpts
    
    
# %% 
# @st.cache
def get_image_near_point(im_collection_id, im_date,  bands_rgb, latitude, longitude, buffer_px, 
                        return_geopandas = False):
    """
    Get an earth engine image near a point

    Parameters
    ----------
    im_collection_id : str
        DESCRIPTION.
    im_date : str
        DESCRIPTION.
    bands_rgb : list
        Band names.
    latitude : float
        Latitude (EPSG 4326).
    longitude : float
        Longitude (EPSG 4326).
    buffer_px : int
        Number of pixels to buffer on each side.
    return_geopandas : bool, optional
        If True, return geopandas.DataFrame, otherwise np.array. The default is False.

    Returns
    -------
    return_val : np.array or geopandas.DataFrame
        m x n grid with bands specified by bands_rgb.
        
    
    Examples
    --------
    im_array = get_image_near_point(im_collection_id = 'COPERNICUS/S2_SR', 
                                    im_date = '2020-02-03',  
                                    bands_rgb = ['B8','B4','B3'], 
                                    latitude = 11.4086, 
                                    longitude = 77.7791, 
                                    buffer_px = 10, 
                                    return_geopandas = False)
    plt = plot_array_image(im_array1)
    plt.show()

    """
    
    
        
    start_datetime = datetime.strptime(im_date,'%Y-%m-%d')
    end_date = datetime.strftime(start_datetime + timedelta(days = 1), '%Y-%m-%d')
    
    try:
        pt = ee.Geometry.Point([longitude, latitude])
    except:
        ee.Initialize()
        pt = ee.Geometry.Point([longitude, latitude])
        
    # pt_bbox = pt.buffer(buffer_m, 1).bounds()
    ic = ee.ImageCollection(im_collection_id).filterBounds(pt).filterDate(im_date, end_date)
    # ic = ic.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 0.25))
    im = ic.first().select(bands_rgb)

    if return_geopandas:
        im = im.addBands(im.pixelCoordinates(im.projection()))
    
    # buffer_px = 10
    # generate kernel
    imdim = (buffer_px * 2) + 1
    kernel_list = ee.List.repeat(1, imdim)
    kernel_lists = ee.List.repeat(kernel_list, imdim)
    kernel = ee.Kernel.fixed(imdim, imdim, kernel_lists, 
                             x = buffer_px, y = buffer_px)
    
    
    im_eearray = im.neighborhoodToArray(kernel)
    
    
    # sample the region and return array from ee to python
    im_dict = im_eearray.reduceRegion(ee.Reducer.first(), geometry = pt, scale = 10).getInfo()
    # # old, with bounding box:
    # im_dict = im.sampleRectangle(region = pt_bbox, properties = []).getInfo()
    # im_props = im_dict['properties']

    # len(im_dict['properties'])
    im_props = im_dict
    im_props_keys = list(im_props.keys())

    if return_geopandas:
        df = pd.DataFrame()
        for i in range(len(im_props)):
            colname = im_props_keys[i]
            data = np.array(im_props[colname]).flatten()
            df[colname] = data

        im_projection = im.projection().getInfo()
        # convert to geopandas
        gdf = gpd.GeoDataFrame(df, 
                 geometry = gpd.points_from_xy(df.x, df.y),
                 crs = im_projection['crs'])
        
        return_val = gdf


    else:
        # extract each band separately
        Bred = np.expand_dims(np.array(im_props[bands_rgb[0]]), 2)
        Bgreen = np.expand_dims(np.array(im_props[bands_rgb[1]]), 2)
        Bblue = np.expand_dims(np.array(im_props[bands_rgb[2]]), 2)

        im_array_rgb = np.concatenate((Bred, Bgreen, Bblue), axis = 2)
        return_val = im_array_rgb
    
    return return_val



# @st.cache
def plot_array_image(im_array):
    xcenter = math.floor(im_array.shape[0] / 2)
    ycenter = math.floor(im_array.shape[1] / 2)
    
    maxval = 5000
    
    im_array[im_array > maxval] = maxval
    
    arrow_spacing = 1
    # Scale the data to [0, 255] to show as an RGB image.
    rgb_img_test = (255*((im_array - 0)/maxval)).astype('uint8')
    plt.figure(figsize = (5,5),dpi = 100)
    plt.axis('off')
    plt.imshow(rgb_img_test)
    plt.plot(xcenter, ycenter - arrow_spacing, marker = 'v', color = 'white')
    plt.plot(xcenter, ycenter + arrow_spacing,marker = '^', color = 'white')
    plt.plot(xcenter + arrow_spacing, ycenter, marker = '<', color = 'white')
    plt.plot(xcenter - arrow_spacing, ycenter,marker = '>', color = 'white')
    return plt

# %% 
# Cached funtions for each date to more quickly disply images on reload

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_image_near_point1(im_collection_id, im_date, bands_rgb, loc_pt_latlon, buffer_px):
    im_array = get_image_near_point(
        'COPERNICUS/S2_SR', im_date, bands_rgb, loc_pt_latlon[0].iloc[0], 
        loc_pt_latlon[1].iloc[0], buffer_px, return_geopandas = False)
    return im_array

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_image_near_point2(im_collection_id, im_date, bands_rgb, loc_pt_latlon, buffer_px):
    im_array = get_image_near_point(
        'COPERNICUS/S2_SR', im_date, bands_rgb, loc_pt_latlon[0].iloc[0], 
        loc_pt_latlon[1].iloc[0], buffer_px, return_geopandas = False)
    return im_array

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_image_near_point3(im_collection_id, im_date, bands_rgb, loc_pt_latlon, buffer_px):
    im_array = get_image_near_point(
        'COPERNICUS/S2_SR', im_date, bands_rgb, loc_pt_latlon[0].iloc[0], 
        loc_pt_latlon[1].iloc[0], buffer_px, return_geopandas = False)
    return im_array


@st.cache(suppress_st_warning=True)
def get_image_near_point4(im_collection_id, im_date, bands_rgb, loc_pt_latlon, buffer_px):
    im_array = get_image_near_point(
        'COPERNICUS/S2_SR', im_date, bands_rgb, loc_pt_latlon[0].iloc[0], 
        loc_pt_latlon[1].iloc[0], buffer_px, return_geopandas = False)
    return im_array
