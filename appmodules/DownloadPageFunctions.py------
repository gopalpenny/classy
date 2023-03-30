#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:38:29 2022

@author: gopal
"""
import streamlit as st
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import ee
import sys
import re
from itertools import compress
import plotnine as p9
import math
import geemap
import json
# import ?

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research/Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemodules import rs
from geemodules import eesentinel as ees
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pathlib import Path

import importlib
importlib.reload(rs)

def DownloadPoints(loc, date_range, timeseries_dir_path, ts_status):
    """
    This function downloads all the points in loc using DownloadSamplePt()

    Parameters
    ----------
    loc : gpd.DataFrame
        GeoPandas dataframe to be downloaded
    date_range : LIST (STR)
        Start date and end date as ['YYYY-MM-DD', 'YYYY-MM-DD'].
    timeseries_dir_path : STR
        Path to the google drive timeseries directory where output will be stored.

    Returns
    -------
    None.

    """
    
    # TimeseriesUpdateAllStatus(timeseries_dir_path)
    
    print('Downloading ' + str(loc.shape[0]) + ' points')
    pbar = st.progress(0)
    infobox = st.empty()
    
    # Initialize Earth Engine, if necessary
    try:
        pt_dummy = ee.Geometry.Point([100, 10])
    except:
        ee.Initialize()

    for i in range(loc.shape[0]):
        # print(i)
        # i = 1
        pt_gpd = loc.iloc[i]
        sample_pt_xy = [pt_gpd.geometry.x, pt_gpd.geometry.y]
        
        loc_id = loc.loc_id.iloc[i]
        DownloadSamplePt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox)
        pbar.progress((float(i) + 1) / loc.shape[0])
        
        
    st.success("All sample locations are now being processed by Google Earth Engine")


def DownloadCHIRPSpt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox):
    
    sample_pt_name = 'pt_ts_loc' + str(loc_id)
    
    sample_pt = ee.Geometry.Point(sample_pt_xy)
    
    timeseries_dir_name = Path(timeseries_dir_path).parts[-1]
    
    # Export chirps
    chirps_colname = 'pt_ts_loc_chirps'
    chirps_pt_filename = re.sub('loc_', 'loc' + str(loc_id) +'_', chirps_colname) #sample_pt_name + '_' + chirps_colname
    chirps_pt_filepath = os.path.join(timeseries_dir_path, chirps_pt_filename + '.csv')
    
    chirps_pt_status = TimeseriesCheckLocStatus(loc_id, chirps_colname, timeseries_dir_path)
    if os.path.exists(chirps_pt_filepath):
        dummyvariable = chirps_pt_filepath
        infobox.info(chirps_pt_filename + '.csv already exists')
        # st.write(chirps_pt_filename + '.csv already exists')
    elif str(chirps_pt_status) != 'nan':
        msgchirps = chirps_pt_filename + ' status is ' + str(chirps_pt_status)
        infobox.info(msgchirps)
        # st.write(msgchirps)
    else:
        chirps_output_bands = ['precipitation']
        chirps_ic = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD") \
          .filterBounds(sample_pt) \
          .filterDate(date_range[0],date_range[1])
          
        # Get chirps pixel timeseries
        chirps_ts = rs.get_pixel_timeseries(
            pts_fc = ee.FeatureCollection(sample_pt),
            image_collection = chirps_ic,
            bands = ['precipitation'],
            ic_property_id = 'system:index',
            scale = 10) # for Landsat resolution
        # time_series_pd_load = geemap.ee_to_pandas(time_series_fc)
            
        task_chirps = ee.batch.Export.table.toDrive(
            collection = chirps_ts,
            selectors = chirps_output_bands + ['image_id'],
            folder = timeseries_dir_name,
            description = chirps_pt_filename,
            fileNamePrefix = chirps_pt_filename)
        
        task_chirps.start()
        
        TimeseriesUpdateLocStatus(loc_id, chirps_colname, 'Running', timeseries_dir_path)
        
        infobox.info('Generating ' + chirps_pt_filename + '.csv')
        # st.write('Generating ' + chirps_pt_filename + '.csv')
        
def DownloadS1pt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox):
    
    sample_pt_name = 'pt_ts_loc' + str(loc_id)
    
    sample_pt = ee.Geometry.Point(sample_pt_xy)
    
    timeseries_dir_name = Path(timeseries_dir_path).parts[-1]
    
    # Export S1
    s1_colname = 'pt_ts_loc_s1'
    s1_pt_filename = re.sub('loc_', 'loc' + str(loc_id) +'_', s1_colname) #sample_pt_name + '_' + s1_colname
    s1_pt_filepath = os.path.join(timeseries_dir_path, s1_pt_filename + '.csv')
    
    s1_pt_status = TimeseriesCheckLocStatus(loc_id, s1_colname, timeseries_dir_path)
    if os.path.exists(s1_pt_filepath):
        dummyvariable = s1_pt_filepath
        infobox.info(s1_pt_filename + '.csv already exists')
        # st.write(s1_pt_filename + '.csv already exists')
    elif str(s1_pt_status) != 'nan':
        msgs1 = s1_pt_filename + ' status is ' + str(s1_pt_status)
        infobox.info(msgs1)
        # st.write(msgs1)
    else:
        s1_output_bands = ['HH','VV','HV','VH','angle']
        s1_ic = ee.ImageCollection("COPERNICUS/S1_GRD") \
          .filterBounds(sample_pt) \
          .filterDate(date_range[0],date_range[1])
          
        # Get S1 pixel timeseries
        s1_ts = rs.get_pixel_ts_allbands(
            pts_fc = ee.FeatureCollection(sample_pt),
            image_collection = s1_ic,
            ic_property_id = 'system:index',
            scale = 10) # for Landsat resolution
        # time_series_pd_load = geemap.ee_to_pandas(time_series_fc)
            
        task_s1 = ee.batch.Export.table.toDrive(
            collection = s1_ts,
            selectors = s1_output_bands + ['image_id'],
            folder = timeseries_dir_name,
            description = s1_pt_filename,
            fileNamePrefix = s1_pt_filename)
        
        task_s1.start()
        
        TimeseriesUpdateLocStatus(loc_id, s1_colname, 'Running', timeseries_dir_path)
        
        infobox.info('Generating ' + s1_pt_filename + '.csv')
        # st.write('Generating ' + s1_pt_filename + '.csv')
        
def DownloadS2pt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox):
    
    sample_pt_name = 'pt_ts_loc' + str(loc_id)
    
    sample_pt = ee.Geometry.Point(sample_pt_xy)
    
    timeseries_dir_name = Path(timeseries_dir_path).parts[-1]
    s2_colname = 'pt_ts_loc_s2'
    s2_pt_filename = re.sub('loc_', 'loc' + str(loc_id) +'_', s2_colname) #sample_pt_name + '_s2'
    s2_pt_filepath = os.path.join(timeseries_dir_path, s2_pt_filename + '.csv')
    
    s2_pt_status = TimeseriesCheckLocStatus(loc_id, s2_colname, timeseries_dir_path)
    if os.path.exists(s2_pt_filepath):
        dummy = s2_pt_filepath
        # print(s2_pt_filename + '.csv already exists')
        infobox.info(s2_pt_filename + '.csv already exists')
        
    elif s2_pt_status != 'nan':
        msgs2 = s2_pt_filename + ' status is ' + str(s2_pt_status)
        # print(msgs2)
        infobox.info(msgs2)
        # st.write(type(s2_pt_status))
    else:
    
        s2_output_bands = ['B8','B4','B3','B2','clouds','cloudmask','shadows','probability']
        
        # params variable is used to pass  information to the cloud masking functions.
        # see help(add_cld_shadow_mask_func)
        s2params = {
            'START_DATE' : date_range[0],
            'END_DATE' : date_range[1],
            'CLOUD_FILTER' : 50,
            'CLD_PRB_THRESH' : 53, # 53 for Cauvery # 55 for Indus
            'NIR_DRK_THRESH' : 0.2,
            'CLD_PRJ_DIST' : 1,
            'BUFFER' : 50
        }
        
        s2_clouds_ic = ees.get_s2_sr_cld_col(sample_pt, s2params) \
          .map(ees.add_cld_shadow_mask_func(s2params))
        
        # For some reason the reproject() works so that subsequent sampling returns the whole rectangular array
        # see https://stackoverflow.com/questions/64012752/gee-samplerectangle-returning-1x1-array
        # s2_clouds_im = s2_clouds_ic.mosaic().reproject(crs = ee.Projection('EPSG:4326'), scale=10) #.clip(hyd_watershed)
        
        # Get pixel timeseries
        s2_ts = rs.get_pixel_ts_allbands(
            pts_fc = ee.FeatureCollection(sample_pt),
            image_collection = s2_clouds_ic,
            ic_property_id = 'system:index',
            scale = 10) # for Landsat resolution
        # time_series_pd_load = geemap.ee_to_pandas(time_series_fc)
            
        task_s2 = ee.batch.Export.table.toDrive(
            collection = s2_ts,
            selectors = s2_output_bands + ['image_id'],
            folder = timeseries_dir_name,
            description = s2_pt_filename,
            fileNamePrefix = s2_pt_filename)
        
        task_s2.start()
        
        TimeseriesUpdateLocStatus(loc_id, s2_colname, 'Running', timeseries_dir_path)
        
        # print('Generating ' + s2_pt_filename + '.csv')
        infobox.info('Generating ' + s2_pt_filename + '.csv')
        
def DownloadOLI8pt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox):
    
    sample_pt_name = 'pt_ts_loc' + str(loc_id)
    
    sample_pt = ee.Geometry.Point(sample_pt_xy)
    
    timeseries_dir_name = Path(timeseries_dir_path).parts[-1]
    oli8_colname = 'pt_ts_loc_oli8'
    oli8_pt_filename = re.sub('loc_', 'loc' + str(loc_id) +'_', oli8_colname) #sample_pt_name + '_oli8'
    oli8_pt_filepath = os.path.join(timeseries_dir_path, oli8_pt_filename + '.csv')
    
    oli8_pt_status = TimeseriesCheckLocStatus(loc_id, oli8_colname, timeseries_dir_path)
    if os.path.exists(oli8_pt_filepath):
        dummy = oli8_pt_filepath
        infobox.info(oli8_pt_filename + '.csv already exists')
        
    elif oli8_pt_status != 'nan':
        msgoli8 = oli8_pt_filename + ' status is ' + str(oli8_pt_status)
        # print(msgoli8)
        infobox.info(msgoli8)
        # st.write(type(oli8_pt_status))
    else:
        
        oli8_output_bands = ['SR_B7','SR_B6','SR_B5','SR_B4','SR_B3','SR_B2','clouds','clouds_shadows','cloudmask']
        oli8_ic = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
          .filterBounds(sample_pt) \
          .filterDate(date_range[0],date_range[1])
          
        get_qaband_clouds_shadows = rs.get_qaband_clouds_shadows_func(
              qa_bandname = 'QA_PIXEL', 
              cloud_bit = 3, 
              shadow_bit = 4,
              keep_orig_bands = True) 
        oli8_clouds_ic = (oli8_ic
          .map(get_qaband_clouds_shadows))
          # .map(lambda im: im.addBands(im.expression('im.clouds | im.clouds_shadows', {'im' : im}).rename('cloudmask'))))
          
        # Get oli8 pixel timeseries
        oli8_ts = rs.get_pixel_ts_allbands(
            pts_fc = ee.FeatureCollection(sample_pt),
            image_collection = oli8_clouds_ic,
            ic_property_id = 'system:index',
            scale = 30) # for Landsat resolution
        # time_series_pd_load = geemap.ee_to_pandas(time_series_fc)
    
        # oli8_output_bands = ['B8','B4','B3','B2','clouds','cloudmask','shadows','probability']
        
        
        # For some reason the reproject() works so that subsequent sampling returns the whole rectangular array
        # see https://stackoverflow.com/questions/64012752/gee-samplerectangle-returning-1x1-array
        # oli8_clouds_im = oli8_clouds_ic.mosaic().reproject(crs = ee.Projection('EPSG:4326'), scale=10) #.clip(hyd_watershed)
        
            
        task_oli8 = ee.batch.Export.table.toDrive(
            collection = oli8_ts,
            selectors = oli8_output_bands + ['image_id'],
            folder = timeseries_dir_name,
            description = oli8_pt_filename,
            fileNamePrefix = oli8_pt_filename)
        
        task_oli8.start()
        
        TimeseriesUpdateLocStatus(loc_id, oli8_colname, 'Running', timeseries_dir_path)
        
        # print('Generating ' + oli8_pt_filename + '.csv')
        infobox.info('Generating ' + oli8_pt_filename + '.csv')
    
def DownloadSamplePt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox):
    """
    This function is used to sample imagery using Google Earth Engine
    The point coordinate is used to generate timeseries within the date_range
    and export the results to Google Drive. It runs one point at a time. 
    Intended to be used within a for loop or mapped over a list of points.

    Parameters
    ----------
    sample_pt_xy : list (float)
        List of length 2 as [x, y] coordinates.
    loc_id : INT
        loc_id for the point
    timeseries_dir_path : str
        path to the directory where results will be saved /proj_name/proj_name_download_timeseries.
    date_range : list (str)
        List of length 2 as [start_date, end_date] for downloading data.

    Returns
    -------
    None.

    """
    
    # Export S1
    DownloadS1pt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox)
    
    # Export S2
    DownloadS2pt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox)
    
    # Export CHIRPS
    DownloadCHIRPSpt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox)
    
    # Export OLI8
    DownloadOLI8pt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox)
        
        
#%%


def TimeseriesStatusInit(proj_path):
    proj_name = st.session_state['proj_name']
    timeseries_dir_path = os.path.join(proj_path, proj_name + '_download_timeseries')

    # Create timeseries directory if it doesn't exist
    if not os.path.exists(timeseries_dir_path): os.mkdir(timeseries_dir_path)
    
    # Generate status path
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    
    # If it doesn't exist, create a blank file with ts_status
    if not os.path.exists(ts_status_path):
        
        ts_all_files = os.listdir(timeseries_dir_path)
        ts_all_filenames = list(set([re.sub('.csv','',re.sub('loc[0-9]+_','loc_',x)) for x in ts_all_files]))
        file_colnames = list(compress(ts_all_filenames, ['loc' in x for x in ts_all_filenames]))
        
        sample_locations_path = os.path.join(proj_path, proj_name + "_sample_locations/sample_locations.shp")
        loc = gpd.read_file(sample_locations_path)
        # loc[['loc_id']]
        ts_status = pd.DataFrame({'loc_id' : loc.loc_id})
        ts_status['allcomplete'] = False
        
        for colname in file_colnames:
            ts_status[colname] = np.nan
        
        ts_status.to_csv(ts_status_path, index= False)
    
    TimeseriesUpdateAllStatus(timeseries_dir_path)
    
    return ts_status_path
        

def rowStatus(rowList):
    """Helper function for TimeseriesUpdateLocStatus
    Checks to see if a csv file is available for all output files
    """
    val = all(['.csv' in str(x) for x in rowList]) & (len(rowList) > 0)
    return val

def TimeseriesUpdateLocStatus(loc_id, colname, new_status, timeseries_dir_path):
    """
    Update the status of a specific loc_id and colname

    Parameters
    ----------
    loc_id : INT
        ID of location.
    colname : STR
        name of column to update.
    new_status : STR
        description of updated status. if 'check', update status if file exists
    proj_path : STR
        path to the project.

    Returns
    -------
    None.

    """
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    ts_status = pd.read_csv(ts_status_path)
    idx = ts_status.index[ts_status.loc_id == loc_id]
    
    if not colname in ts_status.columns:
        ts_status[colname] = np.nan
        
    # # Check to 
    # if new_status == 'check':
    #     loc_column_csv_filename = re.sub('loc_','loc' + str(loc_id) + '_',colname) + '.csv'
    #     loc_column_csv_path = os.path.join(timeseries_dir_path, loc_column_csv_filename)
    #     if os.path.exists(loc_column_csv_path):
    #         ts_status.loc[idx, colname] = loc_column_csv_filename
    # else:
    ts_status.loc[idx, colname] = new_status
        
    ts_status.to_csv(ts_status_path, index = False)
    
# %%
foo = 'hi'

if foo == 'hi':
    print('bye')
# %%

def TimeseriesUpdateAllStatus(timeseries_dir_path):
    """
    Update the status of all loc_id's by checking for .csv files for every colname

    Parameters
    ----------
    timeseries_dir_path : STR
        path to the timeseries subdirectory.

    Returns
    -------
    None.

    """
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    ts_status = pd.read_csv(ts_status_path)
    
    
    all_loc = ts_status.loc_id
    all_columns = ts_status.drop(['loc_id', 'allcomplete'], axis = 1).columns
    
    
    for loc_id in all_loc:
        idx = ts_status.index[ts_status.loc_id == loc_id]
    
        for colname in all_columns:
            loc_column_csv_filename = re.sub('loc_','loc' + str(loc_id) + '_',colname) + '.csv'
            loc_column_csv_path = os.path.join(timeseries_dir_path, loc_column_csv_filename)
            # raise ValueError('We made it into the if statement.')
            if os.path.exists(loc_column_csv_path): # if file exists, add it to loc_id, column cell
                ts_status.loc[idx, colname] = loc_column_csv_filename
                
            else:
                # enter_if is a pandas timeseries with 1 value -- need to extract first value
                enter_if = ts_status.loc[idx, colname] == loc_column_csv_filename
                if enter_if.iloc[0]: # if file does not exist but status is filename, set to np.nan
                    # raise ValueError('We made it into the if statement.')
                    ts_status.loc[idx, colname] = np.nan
                
            # TimeseriesUpdateLocStatus(loc_id, colname, 'check', timeseries_dir_path)
    
    ts_status['allcomplete'] = ts_status.drop(['loc_id','allcomplete'], axis = 1).apply(rowStatus, axis = 1).to_list()
    
    ts_status.to_csv(ts_status_path, index = False)
    

    
def TimeseriesCheckLocStatus(loc_id, colname, timeseries_dir_path):
    """
    Update the status of a specific loc_id and colname

    Parameters
    ----------
    loc_id : INT
        ID of location.
    colname : STR
        name of column to update.
    new_status : STR
        description of updated status.
    proj_path : STR
        path to the project.

    Returns
    -------
    Status of loc_id in ts_status.csv, as cell contents.

    """
    ts_status_path = os.path.join(timeseries_dir_path, 'ts_status.csv')
    ts_status = pd.read_csv(ts_status_path)
    idx = ts_status.index[ts_status.loc_id == loc_id]
    
    if colname in list(ts_status.columns):
        return_val = str(ts_status.loc[idx, colname].to_list()[0])
    else:
        return_val = str(np.nan)
        
    return return_val



        
        