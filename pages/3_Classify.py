#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:40:03 2022

@author: gopal
"""

import streamlit as st
st.set_page_config(page_title="Classify", layout="wide", page_icon="🌏")
import pandas as pd
import numpy as np
import os
import re
import plotnine as p9
import leafmap
import appmodules.manclass as mf
import appmodules.ClassifyPageFunctions as cpf
from streamlit_folium import st_folium
import folium
import geopandas as gpd
from itertools import compress
import math
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import OrderedDict
import appmodules.SamplePageFunctions as spf
import time

import importlib
importlib.reload(mf)
importlib.reload(cpf)

print('\n####################################################')
print('############ INITIALIZING CLASSIFY PAGE ############')
start_timer = time.time()
gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
# sys.path.append(gdrive_ml_path)
out_folder = 'gee_sentinel_ts'
data_path = os.path.join(gdrive_ml_path, 'classapp/script_output', out_folder)

# %%
if 'classification_year' not in st.session_state:
    st.session_state['classification_year'] = st.session_state['proj_vars']['classification_year_default']

if 'subclass_year' not in st.session_state:
    st.session_state['subclass_year'] = 'Subclass' + str(st.session_state['classification_year'])
    

st.title('Pixel classification (' + str(st.session_state['classification_year']) + ')')
if not st.session_state['status']['sample_status']:
    st.markdown("Generate sample locations on `Sample Locations` page before proceeding")

if 'show_snapshots' not in st.session_state:
    st.session_state['show_snapshots'] = False


# %%
classification_dir_path = st.session_state['paths']['classification_dir_path']
class_path = st.session_state['paths']['class_path']


def print_time(string, last_timer):
    diff_timer = round(time.time() - last_timer,3)
    string_print = string + ' (' + str(diff_timer) + 's)'
    print(string_print)
    last_timer = time.time()
    return last_timer

# %%
last_timer = print_time('Creating class_df in session_state', start_timer)
if 'class_df' not in st.session_state:
    st.session_state['class_df'] = cpf.InitializeClassDF()
else:
    st.session_state['class_df'] = cpf.InitializeClassDF()

# %5



allpts = cpf.build_allpts(st.session_state['paths']['proj_path'])
st.session_state['allpts'] = allpts


# %%
lon_pts = allpts.geometry.x
lat_pts = allpts.geometry.y
lon_min = float(math.floor(lon_pts.min()))
lon_max = float(math.ceil(lon_pts.max()))
lat_min = float(math.floor(lat_pts.min()))
lat_max = float(math.ceil(lat_pts.max()))


if 'filterargs' not in st.session_state:
    st.session_state['filterargs'] = {
        'lon' : [lon_min, lon_max],
        'lat' : [lat_min, lat_max],
        'Class' : 'Any',
        'Subclass' : 'Any',
        'Downloaded' : 'Yes'
        }

if 'class_df_filter' not in st.session_state:
    filterargs = st.session_state['filterargs']
    # st.session_state['class_df_filter'] = 1#
    
    # class_df_filter set within apply_filter function
    cpf.apply_filter(lat_range = filterargs['lat'], 
                    lon_range = filterargs['lon'], 
                    class_type = filterargs['Class'], 
                    subclass_type = filterargs['Subclass'], 
                    downloaded = filterargs['Downloaded'])
# %%
last_timer = print_time('Done creating class_df_filter in session_state', last_timer)

st_session_state = {}
if 'loc_id' not in st.session_state:
    st.session_state['loc_id'] = 0
    
    
loc_id = st.session_state['loc_id']



def go_to_id_year(id_to_go, year):
    st.session_state.loc_id = int(id_to_go)
    st.session_state.classification_year = int(year)
    st.session_state.subclass_year = 'Subclass' + str(year)
    

view_options_expander = st.sidebar.expander('View options, zoom, and go to ID')
with view_options_expander:
    s0colA, s0colB = view_options_expander.columns([1,1])
# s0colA, s0colB = st.sidebar.columns([1,1])

with s0colA:
    st.checkbox('Snapshots and spectra', value = False, key = 'show_snapshots')

with s0colB:
    st.number_input('Zoom', min_value= 10, max_value= 20, value = 18, key = 'default_zoom_classify', label_visibility='collapsed')


# %% 
# GO TO EXPANDER

# go_to_expander = st.sidebar.expander('Go to')

with view_options_expander:
    # st.text('go to year not working')
    s2colA, s2colB, s2colC = view_options_expander.columns([1.5,2,1.5])
    
with s2colA:
    id_to_go = st.text_input("ID", value = str(loc_id))
with s2colB:
    proj_years = st.session_state['proj_vars']['proj_years']
    classification_year = st.session_state['classification_year']
    idx_class_year = [i for i in range(len(proj_years)) if proj_years[i] == classification_year][0]
    # year_to_go = st.text_input("Year", value = str(st.session_state['classification_year']))
    year_to_go = st.selectbox("Year", options = proj_years, index = idx_class_year)
with s2colC:
    st.text("")
    st.text("")
    st.button('Go', on_click = go_to_id_year, args = (id_to_go, year_to_go, ))

s1colA, s1colB, s1colC = st.sidebar.columns([2.25,1.25,1.25])

with s1colC:
    st.button('Next', on_click = cpf.next_button, args = ())
with s1colB:
    st.button('Prev', on_click = cpf.prev_button, args = ())

# side_layout = st.sidebar.beta_columns([1,1])
with s1colA: #scol2 # side_layout[-1]:
    st.markdown('### Location ID: ' + str(loc_id))
    # loc_id = int(st.number_input('Location ID', 1, allpts.query('allcomplete').loc_id.max(), 1))
    

# loc_id_num = loc_id
# loc_id = 1

loc_pt = allpts[allpts.loc_id == loc_id]
loc_pt_latlon = [loc_pt.geometry.y, loc_pt.geometry.x]

last_timer = print_time('Done getting coordinates for loc_id: ' + str(loc_id), last_timer)

# %%
region_shp_path = st.session_state['paths']['region_shp_path']
region_shp = gpd.read_file(region_shp_path)

print('#### DEBUG LINE ######')
no_classes_set = st.session_state['class_df'][st.session_state['subclass_year']].isnull().all()
no_classes_in_filter = st.session_state['class_df_filter'][st.session_state['subclass_year']].isnull().all()

print('loc_id', loc_id)

print('class_df_filter', st.session_state['class_df_filter'])
# print('no_classes_in_filter', no_classes_in_filter)
if no_classes_in_filter and not no_classes_set:

    # If there are no points in the filter, then set the Class and SubClass to all points
    st.warning('No points matched the filter. Resetting filter to all Classes & Subclasses.')
    cpf.apply_filter(lat_range = st.session_state['filterargs']['lat'], 
                    lon_range = st.session_state['filterargs']['lon'], 
                    class_type = 'Any', 
                    subclass_type = 'Any', 
                    downloaded = st.session_state['filterargs']['Downloaded'])
    # st.session_state['class_df_filter'][st.session_state['subclass_year']].iloc[0] = 'None'

p_map = (p9.ggplot() + 
          p9.geom_map(data = region_shp, mapping = p9.aes(), fill = 'white', color = "black") +
           p9.geom_map(data = allpts, mapping = p9.aes(), fill = 'lightgray', shape = 'o', color = None, size = 1, alpha = 1) +
           mf.MapTheme() + 
          p9.theme(legend_position = (0.8,0.7), figure_size = (4,4)) +
          p9.coord_equal())
if not no_classes_in_filter:
    p_map = (p_map +
    p9.geom_map(data = st.session_state['class_df_filter'], mapping = p9.aes(fill = st.session_state['subclass_year']), shape = 'o', color = None, size = 2))

p_map = (p_map +
           p9.geom_map(data = loc_pt, mapping = p9.aes(), fill = 'black', shape = 'o', color = 'black', size = 4))

last_timer = print_time('Done creating sidebar map', last_timer)
# %%
main_col1, main_col2 = st.columns(2)

# %%
plot_theme = p9.theme(panel_background = p9.element_rect())

# %%
# st.write(st.session_state.class_df)
Class_prev = list(st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, 'Class'])[0]
Classes =  list(st.session_state.class_df.Class.unique()) + ['Input new']
Classes = [x for x in Classes if x != '-']
Classes = ['-'] + list(compress(Classes, [str(x) != 'nan' for x in Classes]))
Classesidx = [i for i in range(len(Classes)) if Classes[i] == Class_prev] + [0]
last_timer = print_time('Done getting class_df for loc_id: ' + str(loc_id), last_timer)

new_class = "-"


scol1, scol2 = st.sidebar.columns([1,1])

with scol1:
    # date_start = st.date_input('Date', value = '2019-06-01')
    ClassBox = st.selectbox("Class: " + str(Class_prev), 
                 options = Classes, 
                 index = Classesidx[0])
    if ClassBox == 'Input new':
        new_class = st.text_input('New Class')

Subclass_prev = list(st.session_state.class_df.loc[st.session_state.class_df.loc_id == loc_id, st.session_state['subclass_year']])[0]
# Subclasses = list(st.session_state.class_df.Subclass.unique()) + ['Input new']
Class_subset = st.session_state.class_df #[Class_subset.Class == ClassBox]
# Subclasses = list(Class_subset[st.session_state['subclass_year']].unique()) + ['Input new']
# print('Class_subset')
# print(Class_subset)
Class_subset_names = Class_subset.columns.tolist()
Subclasses = np.unique(np.concatenate([Class_subset[col].astype('str') for col in Class_subset_names if 'Subclass' in col])).tolist() + ['Input new']
Subclasses = [x for x in Subclasses if x != '-']
Subclasses = ['-'] + list(compress(Subclasses, [str(x) != 'nan' for x in Subclasses]))
Subclassesidx = [i for i in range(len(Subclasses)) if Subclasses[i] == Subclass_prev] + [0]
new_subclass = "-"
last_timer = print_time('Done getting subclass_df for loc_id: ' + str(loc_id), last_timer)
        
with scol2:
    Subclass = st.selectbox("Sub-class: " + str(Subclass_prev), 
                 options = Subclasses, 
                 index = Subclassesidx[0])
    if Subclass == 'Input new':
        new_subclass = st.text_input('New Sub-class')
        
with st.sidebar:
    st.button('Update classification', on_click = cpf.UpdateClassDF,
              args = (loc_id, ClassBox, Subclass, new_class, new_subclass, st.session_state['subclass_year'], ))
    # test



# %%

# def next_button():
#     st.session_state.loc_id += 1
# def prev_button():
#     st.session_state.loc_id += 1
# with scol1: #side_layout[0]:
#     st.text(' ')
#     st.text(' ')
#     st.button('Prev', on_click = prev_button, args = ())
# with scol3: #side_layout[-1]:
#     st.text(' ')
#     st.text(' ')
#     st.button('Next', on_click = next_button, args = ())
    
with st.sidebar:
    st.pyplot(p9.ggplot.draw(p_map))

# with st.sidebar:
#     lat = st.number_input('Location ID', 0.0, 90.0, 13.0, ) #, step = 0.1)
#     lon = st.number_input('Lon', -180.0, 180.0, 77.0) #, step = 0.1)
    # st.write(data_path_files)
    
# m = leafmap.Map(center=(lat, lon), zoom=18)
# m.add_tile_layer(
#     url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
#     name="Google Satellite",
#     attribution="Google",
# )


if not 'default_zoom_classify' in st.session_state:
    st.session_state['default_zoom_classify'] = 17
    last_timer = print_time('Done setting default_zoom_classify', last_timer)

# m_folium = folium.Map()
m_folium = folium.Map(location = loc_pt_latlon, zoom_start = st.session_state['default_zoom_classify'])
tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
        ).add_to(m_folium)

# get pixel polygonsloc_id, ic_name, coords_xy, ic_str, band_name,
loc_pt_xy = [float(loc_pt_latlon[1]), float(loc_pt_latlon[0])]
landsat_px_poly = spf.get_pixel_poly(loc_id,'oli8', loc_pt_xy, 'LANDSAT/LC08/C02/T1_L2', 'SR_B5', buffer_m = 60, vector_type = 'gpd', option = 'local-check')
last_timer = print_time('Done getting landsat_px_poly', last_timer)
s2_px_poly = spf.get_pixel_poly(loc_id,'s2',loc_pt_xy, 'COPERNICUS/S2', 'B4', buffer_m = 60, vector_type = 'gpd', option = 'local-check')
last_timer = print_time('Done getting s2_px_poly', last_timer)
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

last_timer = print_time('Adding point to map', last_timer)
m_folium \
    .add_child(folium.CircleMarker(location = loc_pt_latlon, radius = 5)) #\
    # .add_child(folium.CircleMarker(location = loc_pt_latlon_adj, radius = 5, color = 'red'))
# point = 
# tile1 = folium.TileLayer(
#         tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
#         attr = 'Google',
#         name = 'Google Satellite',
#         overlay = False,
#         control = True
#        ).add_to(m)

last_timer = print_time('Done adding pixel polygons to map', last_timer)


start_date_string_full = (str(st.session_state['classification_year']) + '-' + 
                          st.session_state['proj_vars']['classification_start_month'] + '-' +
                          str(st.session_state['proj_vars']['classification_start_day']))

start_datetime = datetime.strptime(start_date_string_full, '%Y-%B-%d')
end_datetime = start_datetime + relativedelta(years = 1)
datetime_range = [start_datetime, end_datetime]
date_range = [datetime.strftime(x, '%Y-%m-%d') for x in datetime_range]
start_date = date_range[0]

last_timer = print_time('Done getting dates for classification', last_timer)
# %%
def FilterPts(allpts, lat_range):
    last_timer = print_time('Running FilterPts()', last_timer)
    filterpts = st.session_state['allpts']
    # latitude
    filterpts = filterpts[allpts['lat'] >= lat_range[0]]
    filterpts = filterpts[allpts['lat'] <= lat_range[1]]
    # longitude
    filterpts = filterpts[allpts['lon'] >= lon_range[0]]
    filterpts = filterpts[allpts['lon'] <= lon_range[1]]
    return filterpts

# filterpts = FilterPts(allpts, lat_range)

sideexp = st.sidebar.expander('Filter points')
with sideexp:
    se1col1, se1col2 = sideexp.columns([1, 1])
    class_types = ['Any'] + list(st.session_state.class_df.Class.unique())
    cur_class_type = st.session_state['filterargs']['Class']
    class_types_idx = [i for i in range(len(class_types)) if class_types[i] == cur_class_type][0]
    
    with se1col1:
        class_type = st.selectbox('Class (' + cur_class_type + ')', options = class_types, 
                                   index = class_types_idx)
    
    # Only filter points in current classification year
    subclass_types = ['Any'] + list(st.session_state.class_df[st.session_state['subclass_year']].unique())
    # subclass_types = ['Any'] + np.unique(np.concatenate([Class_subset[col].astype('str') for col in Class_subset_names if 'Subclass' in col])).tolist()
    cur_subclass_type = st.session_state['filterargs']['Subclass']
    subclass_types_idx = [i for i in range(len(subclass_types)) if subclass_types[i] == cur_subclass_type][0]
    with se1col2:
        subclass_type = st.selectbox(str(st.session_state['subclass_year']) + ' (' + cur_subclass_type + ')', options = subclass_types, 
                                   index = subclass_types_idx)
                              
    cur_lat = st.session_state['filterargs']['lat']
    cur_lon = st.session_state['filterargs']['lon']
    lat_header = 'Latitude [' + str(cur_lat[0]) + \
      ', ' + str(cur_lat[1]) + ']'
    lon_header = 'Longitude [' + str(cur_lon[0]) + \
      ', ' + str(cur_lon[1]) + ']'
    # lat_header = 'Latitude ('
    lat_range = st.slider(lat_header, min_value = lat_min, max_value = lat_max, 
              value = (st.session_state['filterargs']['lat'][0], st.session_state['filterargs']['lat'][1]))
    lon_range = st.slider(lon_header, min_value = lon_min, max_value = lon_max, 
              value = (st.session_state['filterargs']['lon'][0], st.session_state['filterargs']['lon'][1]))
    
    download_options = ['All', 'Yes', 'No']
    cur_download_type = st.session_state['filterargs']['Downloaded']
    download_idx = [i for i in range(len(download_options)) if download_options[i] == cur_download_type][0]
    downloaded = st.selectbox('Downloaded (' + cur_download_type + ')', options = download_options, index = download_idx)
    se2col1, se2col2 = sideexp.columns([1, 1])
    
    
with se2col1:
    st.button('Apply filter', on_click=cpf.apply_filter, args = (lat_range, lon_range, class_type, subclass_type, downloaded, ))

with se2col2:
    st.button('Clear filter', on_click=cpf.clear_filter, args = (lat_min, lat_max, lon_min, lon_max, ))
    
    
# %%

# VIEW SNAPSHOTS
last_timer = print_time('Done setting up side expander.', last_timer)
start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
end_date = datetime.strptime(date_range[1], '%Y-%m-%d')

tsS2 = mf.GenS2data(loc_id) #
print('tsS2', tsS2)
tsS2 = tsS2.query('cloudmask == 0')
# landsat = mf.GenLandsatData(loc_id) #.query('clouds_shadows==0')
last_timer = print_time('Done getting timeseries data via GenS2data() and GenLandsatData()', last_timer)

# st.markdown("""###""")
break_col_width = 0.15
st_snapshots_title_cols = st.columns([1, 2, break_col_width,2])

print('Setting parameters for timeseries vertical line dividers and buffer pixels for snapshots')
month_increment = 4
length_out = 4
month_seq = [start_date + relativedelta(months = x) for x in np.arange(length_out) * month_increment]

if st.session_state['show_snapshots']:


    # st.write(month_seq)
    buffer_px = 10

    def update_spectra_range():
        print("Running update_spectra_range")
        st.session_state['spectrum_range_1'] = st.session_state['spectrum_r1']
        st.session_state['spectrum_range_2'] = st.session_state['spectrum_r2']

    with st_snapshots_title_cols[0]:
        st.markdown('### Snapshots')
    with st_snapshots_title_cols[1]:
        st.markdown('')
        st.radio('Snapshot satellite', options = ['Sentinel 2', 'Landsat'], index = 0, 
                horizontal = True, key = 'snapshot_satellite', label_visibility = 'collapsed')
    with st_snapshots_title_cols[3]:
        st.markdown('#### Reflectance spectra')
    # with st_snapshots_title_cols[2]:
    #     st.text('')
    #     st.button('Update', key = 'go_to_spectra', on_click = update_spectra_range, )
        
    st_snapshots_dates_cols = st.columns([1,1,1,break_col_width,0.8,0.8,0.35])
    # snapshot_dates_cols = st_snapshots_dates_cols[0:3]

    if st.session_state['snapshot_satellite'] == 'Landsat':
        dates_datetime = [x.to_pydatetime() for x in list(OrderedDict.fromkeys(tsS2['datetime']))]
        dates_str = [datetime.strftime(x, '%Y-%m-%d') for x in dates_datetime]
    elif st.session_state['snapshot_satellite'] == 'Sentinel 2':
        dates_datetime = [x.to_pydatetime() for x in list(OrderedDict.fromkeys(tsS2['datetime']))]
        dates_str = [datetime.strftime(x, '%Y-%m-%d') for x in dates_datetime]

    def getDatesInRange(all_datetimes, start_date_inclusive, end_date_exclusive, no_dates_month_buff = 0):
        dates_str = [datetime.strftime(x, '%Y-%m-%d') for x in dates_datetime if (start_date_inclusive <= x < end_date_exclusive)]
        if len(dates_str) == 0:
            start_date_inclusive += relativedelta(months = no_dates_month_buff)
            end_date_exclusive += relativedelta(months = no_dates_month_buff)
            dates_str = [datetime.strftime(x, '%Y-%m-%d') for x in dates_datetime if (start_date_inclusive <= x < end_date_exclusive)]
        if len(dates_str) == 0:
            dates_str = ["No dates in range"]
            # select_datetime = month_seq[0] + (month_seq[1] - month_seq[0])/2
            # datetime_nearest = getNearestDatetime(all_datetimes, select_datetime)
            # dates_str = datetime.strptime(datetime_nearest, '%Y-%m-%d')
        return dates_str

        
    with st_snapshots_dates_cols[0]:
        dates_str_1 = getDatesInRange(dates_datetime, month_seq[0], month_seq[1], -1)
        im_date1 = st.selectbox('Select date 1', options = dates_str_1)

    with st_snapshots_dates_cols[1]:
        dates_str_2 = getDatesInRange(dates_datetime, month_seq[1], month_seq[2])
        im_date2 = st.selectbox('Select date 2', options = dates_str_2)
        
    with st_snapshots_dates_cols[2]:
        dates_str_3 = getDatesInRange(dates_datetime, month_seq[2], month_seq[3], 1)
        im_date3 = st.selectbox('Select date 3', options = dates_str_3)

        snapshot_dates = [im_date1, im_date2, im_date3]
    last_timer = print_time('Done getting snapshot dates', last_timer)

    spectrum_slider_date_col = st_snapshots_dates_cols[4:7]
    # spectrum_slider_col = spectrum_col.columns([1,1])

    if 'spectrum_range_1' not in st.session_state:
        st.session_state['spectrum_range_1'] = (month_seq[1], month_seq[2])
    if 'spectrum_range_2' not in st.session_state:
        st.session_state['spectrum_range_2'] = (month_seq[2], end_date)
        
    last_timer = print_time('Done setting default spectrum ranges', last_timer)

def bound_val(val, bounds):
    if val < bounds[0]:
        val = bounds[0]
    elif val > bounds[1]:
        val = bounds[0]
        
    return val
    
def outside_bounds(spec_range_tuple, datetime_range):
    outside = False
    if spec_range_tuple[0] < datetime_range[0]:
        outside = True
    if spec_range_tuple[1] > datetime_range[1]:
        outside = True
        
    return outside

if st.session_state['show_snapshots']:
    with spectrum_slider_date_col[0]:
        init_range1 = st.session_state['spectrum_range_1']
        if outside_bounds(init_range1, datetime_range):
            init_range1 = (month_seq[1], month_seq[2])
        spectrum_range_1 = st.slider('Spectrum range 1', min_value = start_date, max_value = end_date, 
                                    value = init_range1, key = 'spectrum_r1', format = 'MMM')
    with spectrum_slider_date_col[1]:
        init_range2 = st.session_state['spectrum_range_2']
        if outside_bounds(init_range2, datetime_range):
            init_range2 = (month_seq[2], month_seq[3])
        spectrum_range_2 = st.slider('Spectrum range 2', min_value = start_date, max_value = end_date, 
                                    value = init_range2, key = 'spectrum_r2', format = 'MMM')
        
    with spectrum_slider_date_col[2]:
        st.text('')
        st.text('')
        st.button(':heavy_check_mark:', key = 'go_to_spectra', on_click = update_spectra_range, )
    
    last_timer = print_time('Done creating spectrum sliders', last_timer)
    spectra_list = [st.session_state['spectrum_range_1'], st.session_state['spectrum_range_2']]

if not st.session_state['show_snapshots']:
    snapshot_dates = None
    spectra_list = None

p_sentinel = mf.plotTimeseries(loc_id, date_range, month_seq, snapshot_dates, spectra_list) 
last_timer = print_time('Done getting timeseries plot', last_timer)

# %%    
with main_col1:
    st.pyplot(p9.ggplot.draw(p_sentinel))
    last_timer = print_time('Done plotting timeseries', last_timer)

with main_col2:
    # st.write(m.to_streamlit())
    st_folium(m_folium, height = 300, width = 600)
    last_timer = print_time('Done printing map to right column', last_timer)

# st_snapshots_cols = st_snapshots.columns([1,1,1,2])

if st.session_state['show_snapshots']:
    st_snapshots_cols = st.columns([1,1,1,break_col_width,2])
        
    if st.session_state['snapshot_satellite'] == 'Landsat':
        ee_satellite_name = 'COPERNICUS/S2_SR'
        ee_band_names = ['B8','B4','B3']
    elif st.session_state['snapshot_satellite'] == 'Sentinel 2':
        ee_satellite_name = 'COPERNICUS/S2_SR'
        ee_band_names = ['B8','B4','B3']
    with st_snapshots_cols[0]:
        if im_date1 != 'No dates in range':
            im_array1 = cpf.get_image_near_point1(ee_satellite_name, im_date1, ee_band_names, loc_pt_latlon, buffer_px)
            last_timer = print_time('Done getting snapshot 1', last_timer)
            plt1 = cpf.plot_array_image(im_array1)
            st.pyplot(plt1)
            last_timer = print_time('Done plotting snapshot 1', last_timer)
        else:
            st.text('No dates in range')
        

    with st_snapshots_cols[1]:
        if im_date2 != 'No dates in range':
            im_array2 = cpf.get_image_near_point2(ee_satellite_name, im_date2, ee_band_names, loc_pt_latlon, buffer_px)
            last_timer = print_time('Done getting snapshot 2', last_timer)
            plt2 = cpf.plot_array_image(im_array2)
            st.pyplot(plt2)
            last_timer = print_time('Done plotting snapshot 2', last_timer)
        else:
            st.text('No dates in range')
        

    with st_snapshots_cols[2]:
        if im_date3 != 'No dates in range':
            im_array3 = cpf.get_image_near_point3(ee_satellite_name, im_date3, ee_band_names, loc_pt_latlon, buffer_px)
            last_timer = print_time('Done plotting snapshot 3', last_timer)
            plt3 = cpf.plot_array_image(im_array3)
            st.pyplot(plt3)
            last_timer = print_time('Done plotting snapshot 3', last_timer)
        else:
            st.text('No dates in range')
        
        
    with st_snapshots_cols[4]:
        p_spectra = mf.plotSpectra(loc_id, date_range, spectra_list)
        st.pyplot(p9.ggplot.draw(p_spectra))
        last_timer = print_time('Done plotting spectra', last_timer)
    
# with stexp1col4:
#     im_date4 = st.selectbox('Select date 4', options = dates_str)
#     im_array4 = cpf.get_image_near_point4('COPERNICUS/S2_SR', im_date4, ['B8','B4','B3'], loc_pt_latlon, buffer_px)
#     plt4 = cpf.plot_array_image(im_array4)
#     st.pyplot(plt4)
    
with st.expander('Selected points'):
    # st.write(st.session_state.class_df_filter)
    st.write(pd.DataFrame(st.session_state.class_df_filter).drop('geometry', axis = 1))
    # st.write(st.session_state.class_df_filter)
    # st.dataframe(pd.DataFrame(st.session_state.class_df_filter))
    # st.write(pd.DataFrame(allpts).drop('geometry', axis = 1))
last_timer = print_time('Done printing selected points in expander', last_timer)

last_timer = print_time('Finished', start_timer)