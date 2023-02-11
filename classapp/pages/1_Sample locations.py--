#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:03:28 2022

@author: gopal
"""


import streamlit as st
st.set_page_config(page_title="Sample region", layout="wide", page_icon="🌏")
import pandas as pd
import geopandas as gpd
import geemap
import os
import ee
import sys
import plotnine as p9
import re
import numpy as np
from streamlit_folium import st_folium
from itertools import compress
import folium
import appmodules.manclass as mf
import appmodules.SamplePageFunctions as spf
import appmodules.ClassifyPageFunctions as cpf
from io import BytesIO
from pathlib import Path

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research/Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemodules import rs


import importlib
importlib.reload(spf)

# %%
# region_path = os.path.join(st.session_state.proj_path,"region")



if 'class_df' not in st.session_state:
    st.session_state['class_df'] = cpf.InitializeClassDF()
else:
    st.session_state['class_df'] = cpf.InitializeClassDF()

if not st.session_state['status']['region_status']:
    p_map = p9.ggplot() + p9.geom_blank()
else:
    region_shp = gpd.read_file(st.session_state['paths']['region_shp_path'])
    p_map = (p9.ggplot() + mf.MapTheme() +
             p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black"))# +
             # p9.coord_equal() +
             # p9.theme(figure_size = (4,4)))
    
    if st.session_state['status']['sample_status']:
        sample_pts_shp = gpd.read_file(st.session_state['paths']['sample_locations_path'])
        p_map = (p_map + 
                 p9.geom_map(data = sample_pts_shp, mapping = p9.aes(color = 'loc_set', fill = 'loc_set')) +
                 p9.theme(legend_position=(0.8,0.8)))
        
    elif st.session_state['status']['random_status']:
        random_pts = gpd.read_file(st.session_state['paths']['random_locations_path'])
        if 'ee_pt_id' not in random_pts: ## for backward compatibility
            random_pts['ee_pt_id'] = random_pts['loc_id']
            random_pts.to_file(st.session_state['paths']['random_locations_path'])
        p_map = (p_map + 
                 p9.geom_map(data = random_pts, mapping = p9.aes(color = 'ee_pt_id')))


# %% Layout

st.title("Sampling region")

main_columns = st.columns([1,1])

# with col1:
#     st.pyplot(p9.ggplot.draw(p_map))
with main_columns[0]:
    
    # buf = BytesIO()
    # p9.ggplot.draw(p_map).savefig(buf, format="png")
    # st.image(buf)
    st.pyplot(p9.ggplot.draw(p_map))
    

if not 'default_zoom_sample' in st.session_state:
    st.session_state['default_zoom_sample'] = 17

# loc_pt_orig = allpts[allpts.loc_id == loc_id]




if 'loc_id' not in st.session_state:
    st.session_state['loc_id'] = 0
    
loc_id = st.session_state['loc_id']

# %%

sample_locations_dir_path = st.session_state['paths']['sample_locations_dir_path']
sample_locations_path = st.session_state['paths']['sample_locations_path']
random_locations_path = st.session_state['paths']['random_locations_path']
region_shp_path = st.session_state['paths']['region_shp_path']
# %%

# SHAPEFILE PREP
st.markdown(
    """---
### 1. Import shapefile

Add the path to a local shapefile (.shp) in the box below. This shapefile sets the boundary on the sampling region.
    """)
st.markdown('')

path_to_shp_import = st.text_input('Path to shapefile',
              value = '-')

if st.session_state['status']['region_status']: 
    app_path = st.session_state['app_path']
    st.markdown('`Already imported (' + os.path.join(*Path(region_shp_path).parts[-3:]) + ')`')
else:
    st.button('Import', on_click = spf.ImportShapefile, args = (sample_locations_dir_path, path_to_shp_import,))
        
        
st.markdown("""---
### 2. Generate random locations

Clicking the button `Generate random locations` uploads region.shp to Google Earth Engine and generates
`N` random samples using the image collection specified.
            """)

gen_random_columns = st.columns([2,3,3,3,4])
ic_name_list = ['COPERNICUS/S2_SR']

    
with gen_random_columns[0]:
    st.markdown("#### ")
    st.text("")
    addcropmask = st.checkbox('GFSAD mask?', value = False, key = 'gfsadCropMask')
with gen_random_columns[1]:
    numRandomPts = st.number_input('Num pts', 1, 5000, value = 10, key = 'numRandomPts')
with gen_random_columns[2]:
    eeRandomPtsSeed = st.number_input('Earth Engine seed', 0, 5000, value = 10, key = 'eeRandomPtsSeed')

with gen_random_columns[3]:
    ic_name = st.selectbox(label = 'GEE Image Collection', options = ic_name_list)
    
with gen_random_columns[4]:
    if not st.session_state['status']['region_status']:
        st.markdown("#### ")
        st.markdown(" ")
        st.markdown('Need to import region shapefile *before* generating random locations`')
    elif not st.session_state['status']['random_status']: 
        st.markdown("### ")
        st.markdown(" ")
        # numRandomPts = st.session_state['numRandomPts']
        st.button('Generate random locations', on_click = spf.GenerateRandomPts, args = (ic_name, numRandomPts, eeRandomPtsSeed, addcropmask))
    else:
        st.markdown("#### ")
        st.markdown(" ")
        random_pts = gpd.read_file(st.session_state['paths']['random_locations_path']).to_crs(4326)
        st.markdown('`Locations already generated (' + os.path.join(*Path(random_locations_path).parts[-3:]) + ')`')

st.markdown("""---
### 3. Initialize sample locations

Initialize sample locations *after the random locations are generated*. This copies the shapefile
`random_locations.shp` to `sample_locations.shp` and allows you to identifying the main landcover
classes from the basemap, as well as adjust the locations of the random points (e.g., so that they
fall in the center of farms rather than on the border).
            """)
            
st.markdown('`app_path`')
st.write(st.session_state['app_path'])

if not st.session_state['status']['random_status']:
    st.markdown('`Generate random locations before initializing sample locations`')
elif not st.session_state['status']['sample_status']: 
    st.button('Initialize sample locations', on_click = spf.InitializeSampleLocations)
else:
    st.markdown('`Already done (' + os.path.join(*Path(sample_locations_path).parts[-3:]) + ')`')
    #if ('sample_pts' not in st.session_state):
    st.session_state['sample_pts'] = gpd.read_file(st.session_state['paths']['sample_locations_path']).to_crs(4326)   
    print(st.session_state['sample_pts'])
    st.session_state['sample_pts']['loc_set'] = st.session_state['sample_pts']['loc_set'].astype('int64') == 1
    print(st.session_state['sample_pts'])
    # st.session_state['sample_pts'] = sample_pts_read
    
    st.markdown("""
With sample_locations.shp initialized, adjust the points using the sidebar and set each piont with the `SET` button.
    """)
    
    # print(st.session_state['sample_pts'])


# %% 
# GO TO EXPANDER

if st.session_state['status']['sample_status']:
    
    
    st.sidebar.markdown("# Adjust points")
    
    s1colA, s1colB, s1colC = st.sidebar.columns([3,1,1])
    
    # side_layout = st.sidebar.beta_columns([1,1])
    with s1colA: #scol2 # side_layout[-1]:
        st.markdown('### Location ID: ' + str(loc_id))
        # loc_id = int(st.number_input('Location ID', 1, allpts.query('allcomplete').loc_id.max(), 1))
    
    with s1colC:
        st.button('Next', on_click = spf.next_button, args = ())
    with s1colB:
        st.button('Prev', on_click = spf.prev_button, args = ())
        

    sample_pts = st.session_state['sample_pts']
    
    # st.write('st.session_state[sample_pts]')
    # st.write(st.session_state['sample_pts']['loc_set'])
    loc_pt_orig = random_pts[random_pts.loc_id == loc_id]
    sample_pt_set = sample_pts[sample_pts.loc_id == loc_id]
    sample_pt_set_latlon = [sample_pt_set.geometry.y, sample_pt_set.geometry.x]
    
    
    arrow_columns = st.sidebar.columns([2, 1,1,1.3, 2])
    
    if 'y_shift' not in st.session_state:
        st.session_state['y_shift'] = 0
    if 'x_shift' not in st.session_state:
        st.session_state['x_shift'] = 0
    # y_shift = -10
    # x_shift = 10
    up_text = re.sub('^([0-9]+)','+\\1',str(st.session_state['y_shift'])) + ' m'
    right_text = re.sub('^([0-9]+)','+\\1',str(st.session_state['x_shift'])) + ' m'
    
    st.write(sample_pts)
    st.write(sample_pt_set)
    if sample_pt_set.loc_set.iloc[0]:
        pt_set_str = 'YES'
    else:
        pt_set_str = 'NO'
        
    with arrow_columns[0]:
        # st.write(pt_set_str)
        st.text('Set? ' + pt_set_str)
    with arrow_columns[1]:
        st.markdown('###')
        st.text('')
        st.button('←', on_click = spf.left_shift)
    with arrow_columns[2]:
        st.button('↑', on_click = spf.up_shift)
        st.text('')
        st.text('')
        st.button('↓', on_click = spf.down_shift)
    with arrow_columns[3]:
        st.markdown(up_text)
        # st.markdown('')
        st.button('→', on_click = spf.right_shift)
    with arrow_columns[4]:
        st.markdown('##')
        st.text('')
        st.markdown(right_text)
    
    set_columns = st.sidebar.columns([2,1,1])
    
    Class_prev = list(st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'Class'])[0]
    Classes =  list(st.session_state.class_df.Class.unique()) + ['Input new']
    Classes = [x for x in Classes if x != '-']
    Classes = ['-'] + list(compress(Classes, [str(x) != 'nan' for x in Classes]))
    Classesidx = [i for i in range(len(Classes)) if Classes[i] == Class_prev] + [0]
    new_class = "-"
    
    with set_columns[0]:
        ClassBox = st.selectbox("Class: " + str(Class_prev), 
                     options = Classes, 
                     index = Classesidx[0])
        if ClassBox == 'Input new':
            new_class = st.text_input('New Class')
    with set_columns[1]:
        st.markdown('###')
        st.text('')
        st.button('SET', on_click = spf.set_shift, args = (loc_id, ClassBox, new_class, ))
    with set_columns[2]:
        st.markdown('###')
        st.text('')
        st.button('RESET', on_click = spf.reset_shift, args = (loc_id, ))
        
        
    expander_go_to = st.sidebar.expander('Go to')
    
    with expander_go_to:
        # st.text('go to year not working')
        expander_go_to_cols = expander_go_to.columns([2,1])
        
    with expander_go_to_cols[0]:
        id_to_go = st.text_input("ID", value = str(loc_id))
    with expander_go_to_cols[1]:
        st.text("")
        st.text("")
        st.button('Go', on_click = spf.go_to_id, args = (id_to_go, ))
    

    loc_pt_latlon = [loc_pt_orig.geometry.y, loc_pt_orig.geometry.x]
    loc_pt_latlon_shifted = spf.shift_points_m(loc_pt_orig, st.session_state['x_shift'], st.session_state['y_shift'])
    
    
    # loc_pt_latlon = [13, 77]
    loc_pt_latlon_adj = [loc_pt_latlon_shifted.geometry.y, loc_pt_latlon_shifted.geometry.x]
    
    

    
    
    # m_folium = folium.Map()
    m_folium = folium.Map(location = loc_pt_latlon, zoom_start = st.session_state['default_zoom_sample'])
    tile = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = False,
            control = True
            ).add_to(m_folium)
    
    # get pixel polygonsloc_id, ic_name, coords_xy, ic_str, band_name,
    loc_pt_xy = [float(loc_pt_latlon_adj[1]), float(loc_pt_latlon_adj[0])]
    landsat_px_poly = spf.get_pixel_poly(loc_id,'oli8', loc_pt_xy, 'LANDSAT/LC08/C02/T1_L2', 'SR_B5', buffer_m = 60, vector_type = 'gpd')
    s2_px_poly = spf.get_pixel_poly(loc_id,'s2',loc_pt_xy, 'COPERNICUS/S2', 'B4', buffer_m = 60, vector_type = 'gpd')
    def style(feature):
        return {
            'fill': False,
            'color': 'white',
            'weight': 1
        }
    folium.GeoJson(data = landsat_px_poly['geometry'], 
                    style_function = style).add_to(m_folium)
    folium.GeoJson(data = s2_px_poly['geometry'], 
                    style_function = style).add_to(m_folium)
    
    
    
    m_folium \
        .add_child(folium.CircleMarker(location = loc_pt_latlon, radius = 5)) \
        .add_child(folium.CircleMarker(location = loc_pt_latlon_adj, radius = 5, color = 'red')) \
        .add_child(folium.CircleMarker(location = sample_pt_set_latlon, radius = 5, color = 'limegreen'))
        
    with main_columns[1]:
        st_folium(m_folium, height = 400, width = 600)
        default_zoom_sample = st.number_input('Default zoom', min_value= 10, max_value= 20, value = 18, key = 'default_zoom_sample')
        

# sample_pts_raw = gpd.read_file(st.session_state['paths']['sample_locations_path']).set_crs(4326)   
# sample_pts_fix = sample_pts_raw

# sample_pts_fix['loc_set'] = sample_pts_fix['loc_set'].astype('int64') == 1
# st.write(type(sample_pts_fix['loc_set'].iloc[0]))
# st.write((sample_pts_fix['loc_set'].iloc[0]))

# st.write(type(sample_pts_raw['loc_set'].iloc[0]))
# st.write((sample_pts_raw['loc_set'].iloc[0]))




    