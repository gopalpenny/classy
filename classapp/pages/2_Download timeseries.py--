#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:42:12 2022

@author: gopal
"""



import streamlit as st
st.set_page_config(page_title="Grab sample data", layout="wide", page_icon="🌏")
import pandas as pd
# import numpy as np
import geopandas as gpd
# import geemap
import os
import ee
import sys
import plotnine as p9
# import re
import importlib
import appmodules.manclass as mf
import appmodules.DownloadPageFunctions as dpf

# gdrive_path = '/Users/gopal/Google Drive'
# gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# sys.path.append(gdrive_ml_path)
# from geemodules import rs
# from geemodules import eesentinel as ees


# app_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/manclassify/app_data'
# proj_path = os.path.join(app_path, 'region1')

# sample_locations_dir_path = os.path.join(st.session_state.proj_path,st.session_state.proj_name + "_sample_locations")
# sample_locations_path = os.path.join(sample_locations_dir_path, "sample_locations.shp")
# region_shp_path = os.path.join(sample_locations_dir_path,"region.shp")

# proj_name = 'region1'

# region_status = os.path.exists(region_shp_path)
# sample_status = os.path.exists(sample_locations_path)

# %%


st.title("Download timeseries for sample locations")

# %%

# 
importlib.reload(mf)
#

if st.session_state['status']['sample_status']:
    loc = gpd.read_file(st.session_state['paths']['sample_locations_path'])
    loc['loc_set'] = loc['loc_set'].astype('int64') == 1
    
    ts_status_path = dpf.TimeseriesStatusInit(st.session_state['paths']['proj_path'])
    
    # def plot(region_status, sample_status, ts_status_path, )
    ts_status = pd.read_csv(ts_status_path)
    ts_status_downloaded = ts_status.loc[ts_status.allcomplete].loc_id
    
    
    # st.write(ts_status)
    loc_downloaded = loc.loc[loc.loc_id.isin(ts_status_downloaded)]
    
    # Points not yet downloaded
    ts_status_notdownloaded = ts_status.loc[~ts_status.allcomplete].loc_id
    loc_notdownloaded = loc.loc[loc.loc_id.isin(ts_status_notdownloaded)]
    
    # st.write(ts_status_downloaded)

# %%

if not st.session_state['status']['region_status']:
    p_map = p9.ggplot() + p9.geom_blank() + p9.coord_equal()
else:
    region_shp = gpd.read_file(st.session_state['paths']['region_shp_path'])
    p_map = (p9.ggplot() + mf.MapTheme() +
             p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black"))
    
    if st.session_state['status']['sample_status']:
        # sample_locations_shp = gpd.read_file(sample_locations_path)
        p_map = (p_map + 
                 p9.geom_map(data = loc_notdownloaded.query('loc_set == False'), mapping = p9.aes(), color = '#888888', shape = '+', size = 1) + 
                 p9.geom_map(data = loc_notdownloaded.query('loc_set'), mapping = p9.aes(), color = 'red', shape = '+', size = 3) + 
                 p9.geom_map(data = loc_downloaded, mapping = p9.aes(), shape = 'o', fill = 'blue', color = None, size = 2))




col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.pyplot(p9.ggplot.draw(p_map))

# %%
try:
    date_range = ([st.session_state['proj_vars']['proj_start_date'], 
                   st.session_state['proj_vars']['proj_end_date']])
except:
    st.error('Need to set project timespan before proceeding')

# %%

if not st.session_state['status']['sample_status']:
    st.markdown("Generate sample locations on `Sample Locations` page before proceeding")
else:
    # Number of points to download at a time
    st.markdown("Date range to download: `" + date_range[0] + '` to `' + date_range[1] + '`')
    download_columns = st.columns([1,1,3])
    with download_columns[0]:
        NumPts = int(st.number_input('Number of points', 1, value = 5))
    with download_columns[1]:
        st.markdown("###")
        st.text("")
        # points to download
        loc_selected_numpts = loc_notdownloaded.loc[loc_notdownloaded['loc_set']].iloc[0:NumPts]
        timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']
        st.button('Download points', on_click = dpf.DownloadPoints, 
                  args = (loc_selected_numpts, date_range, timeseries_dir_path,ts_status,))
    with download_columns[2]:
        st.markdown("###")
        st.text("")
        st.button('Check and update status', on_click = dpf.TimeseriesUpdateAllStatus, args = (timeseries_dir_path,))
        
    st.write('Points to download:')
    st.write(pd.DataFrame(loc_selected_numpts).drop('geometry', axis = 1))
        
    
    
with st.expander('Downloaded files for all points'):
    st.write(ts_status)

