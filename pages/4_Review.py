

# %%
import streamlit as st
st.set_page_config(page_title="Review", layout="wide", page_icon="🌏")
import pandas as pd
import numpy as np 
import os

# %%
wds = ["/Users/gopalpenny/Projects/ml/classy",
       "/Users/gopal/Projects/ml/classy",
       "/home/svu/gpenny/Projects/classy"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
os.chdir(wd_exists)

# %%
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

# %%

# if 'class_df' not in st.session_state:
#     st.session_state['class_df'] = cpf.InitializeClassDF()
# else:
#     st.session_state['class_df'] = cpf.InitializeClassDF()
    
# %%

class_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/classapp/app_data/TGHalli/TGHalli_classification/location_classification.csv'
class_df = pd.read_csv(class_path)
class_df_long = class_df.melt(id_vars = ('loc_id','Class'), 
              value_vars = [col for col in class_df.columns if 'Subclass' in col],
              var_name = 'SubclassYear', value_name = 'Subclass') \
              .dropna()
class_df_long['Year'] = class_df_long.SubclassYear.str.extract(r'(\d+)').astype(int)
# class_df_long

# %%
timeseries_dir_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/classapp/app_data/TGHalli/TGHalli_download_timeseries'
# %%

def get_yearless_date(df):
    date_yearless = [d - relativedelta(years = y - 2000) for d,y in zip(df['datetime'],df['Year'])]
    return date_yearless

def prep_label_timeseries_all(class_df_long, generate_ts = False, timeseries_dir_path = None):
    if timeseries_dir_path is None:
        timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']

    
    label_timeseries_s1_path = os.path.join(timeseries_dir_path, 'label_timeseries_s1.csv')
    label_timeseries_s2_path = os.path.join(timeseries_dir_path, 'label_timeseries_s2.csv')
    label_timeseries_landsat_path = os.path.join(timeseries_dir_path, 'label_timeseries_landsat.csv')
    all_exist = all([os.path.exists(x) for x in [label_timeseries_s1_path, label_timeseries_s2_path, label_timeseries_landsat_path]])

    if all_exist and not generate_ts:
        print('Reading existing timeseries files')
        ts_s1 = pd.read_csv(label_timeseries_s1_path)
        ts_s2 = pd.read_csv(label_timeseries_s2_path)
        ts_landsat = pd.read_csv(label_timeseries_landsat_path)

    else:
        print('Generating timeseries files')
        progress_text = "Operation in progress. Please wait."
        # my_bar = st.progress(0, text=progress_text)

        num_labels = class_df_long.shape[0]
        for i in np.arange(num_labels):
            # print(i)
            # my_bar.progress(i / num_labels * 100, text=progress_text)
            loc_id = class_df_long.loc_id.iloc[i]
            year = class_df_long.Year.iloc[i]
            date_range = (str(year) + '-06-01', str(year + 1) + '-06-01')

            # Read and prep data for loc_id / year
            loc_ts_s1 = mf.GenS1data(loc_id, date_range, 'wide', timeseries_dir_path)
            loc_ts_s1['loc_id'] = loc_id
            loc_ts_s1['Year'] = year
            loc_ts_s2 = mf.GenS2data(loc_id, date_range, 'wide', timeseries_dir_path)
            loc_ts_s2['loc_id'] = loc_id
            loc_ts_s2['Year'] = year
            loc_ts_landsat = mf.GenLandsatData(loc_id, date_range, 'wide', timeseries_dir_path)
            loc_ts_landsat['loc_id'] = loc_id
            loc_ts_landsat['Year'] = year

            # combine loc_id / year data with previous data
            if i == 0:
                ts_s1, ts_s2, ts_landsat = loc_ts_s1, loc_ts_s2, loc_ts_landsat
            else:
                ts_s1 = pd.concat((ts_s1, loc_ts_s1))
                ts_s2 = pd.concat((ts_s2, loc_ts_s2))
                ts_landsat = pd.concat((ts_landsat, loc_ts_landsat))

        ts_s1 = pd.merge(ts_s1, class_df_long[['loc_id','Year','Subclass']], on = ['loc_id', 'Year'])
        ts_s2 = pd.merge(ts_s2, class_df_long[['loc_id','Year','Subclass']], on = ['loc_id', 'Year'])
        ts_landsat = pd.merge(ts_landsat, class_df_long[['loc_id','Year','Subclass']], on = ['loc_id', 'Year'])

        ts_s1['date_yearless'] = get_yearless_date(ts_s1)
        ts_s2['date_yearless'] = get_yearless_date(ts_s2)
        ts_landsat['date_yearless'] = get_yearless_date(ts_landsat)

        ts_s1 = ts_s1.drop(['datetime','source'], axis = 1)
        ts_s2 = ts_s2.query('cloudmask == 0').drop(['datetime','source'], axis = 1)
        ts_landsat = ts_landsat.query('cloudmask == 0').drop(['datetime','source'], axis = 1)

        ts_s1.to_csv(label_timeseries_s1_path, index = False)
        ts_s2.to_csv(label_timeseries_s2_path, index = False)
        ts_landsat.to_csv(label_timeseries_landsat_path, index = False)

    return ts_s1, ts_s2, ts_landsat

# %%
print('hello world')
ts_s1, ts_s2, ts_landsat = prep_label_timeseries_all(class_df_long, generate_ts = False, timeseries_dir_path = timeseries_dir_path)

# %%

ts_s1_long = ts_s1 \
    .assign(group_var = lambda x: x['loc_id'].astype(str) + '_' + x['Year'].astype(str)) \
    .query('backscatter != Inf')

# %%
Subclass_options = pd.unique(ts_s1.Subclass)
s1_bands = ['backscatter']
s2_bands = ['B8','B4','B3','B2','NDVI']
landsat_bands = ['swir2','swir1','nir','red','green','blue','NDVI']

with st.sidebar:
    st.multiselect('Subclasses', options = Subclass_options, key = 'review_subclasses')
    st.multiselect('S1 bands', options = s1_bands, key = 'review_s1_bands')
    st.multiselect('S2 bands', options = s2_bands, key = 'review_s2_bands')
    st.multiselect('Landsat bands', options = landsat_bands, key = 'review_landsat_bands')
    review_subset_expander = st.expander('Subset data', expanded = False)

Subclass_vals = st.session_state['review_subclasses']
s1_vars = st.session_state['review_s1_bands']
s2_vars = st.session_state['review_s2_bands']
landsat_vars = st.session_state['review_landsat_bands']

# %%
with review_subset_expander:
    st.text_input('S1 pandas query', key = 'subset_s1_query')
    st.text_input('S2 pandas query', key = 'subset_s2_query')
    st.text_input('Landsat pandas query', key = 'subset_landsat_query')

s1_query = st.session_state['subset_s1_query'] if st.session_state['subset_s1_query'] != '' else 'backscatter != Inf'
s2_query = st.session_state['subset_s2_query'] if st.session_state['subset_s2_query'] != '' else 'NDVI != Inf'
landsat_query = st.session_state['subset_landsat_query'] if st.session_state['subset_landsat_query'] != '' else 'nir != Inf'

if len(Subclass_vals) == 0:
    st.markdown('### Please select subclasses in the sidebar')

if all([len(s1_vars) == 0, len(s2_vars) == 0, len(landsat_vars) == 0]):
    st.markdown('### Please select bands from at least one satellite mission')

text_angle = 0
text_hjust = 0.5

# %%
# Plot Sentinel 1 timeseries
if len(Subclass_vals) > 0 and len(s1_vars) > 0:
    try:
        ts_s1_prep = ts_s1.query(s1_query)
    except:
        st.warning('Ignoring invalid S1 query -- must be valid pandas.DataFrame.query() string.')
        ts_s1_prep = ts_s1

    ts_s1_long = ts_s1_prep \
        .query('Subclass in @Subclass_vals') \
        .melt(id_vars = ['loc_id','Year','Subclass','date_yearless'], value_vars=s1_vars) \
        .assign(group_var = lambda x: x['loc_id'].astype(str) + '_' + x['Year'].astype(str))
    p_s1 = (p9.ggplot() + 
    p9.geom_line(data = ts_s1_long, 
                mapping = p9.aes(x = 'date_yearless', y = 'value', group = 'group_var'), alpha = 0.1) +
    # p9.geom_smooth(data = ts_s1_long, mapping = p9.aes(x = 'date_yearless', y = 'value'), color = 'black', size = 3) +
    p9.facet_grid('variable ~ Subclass', scales = 'free_y') +
    p9.scale_x_datetime(date_breaks = '3 months', date_labels = '%b') +
    p9.theme(axis_text_x = p9.element_text(angle = text_angle, hjust =text_hjust),
                figure_size = (8,len(s1_vars)),
                axis_title = p9.element_blank()))
    
    st.markdown('### S1 timeseries')
    st.pyplot(p9.ggplot.draw(p_s1)) 

# %%
# Plot Sentinel 2 timeseries
if len(Subclass_vals) > 0 and len(s2_vars) > 0:
    try: 
        ts_s2_prep = ts_s2.query(s2_query)
    except:
        st.warning('Ignoring invalid S2 query -- must be valid pandas.DataFrame.query() string')
        ts_s2_prep = ts_s2

    ts_s2_long = ts_s2_prep \
        .query('Subclass in @Subclass_vals') \
        .melt(id_vars = ['loc_id','Year','Subclass','date_yearless'], value_vars=s2_vars) \
        .assign(group_var = lambda x: x['loc_id'].astype(str) + '_' + x['Year'].astype(str))
    p_s2 = (p9.ggplot() + 
    p9.geom_line(data = ts_s2_long, 
                mapping = p9.aes(x = 'date_yearless', y = 'value', group = 'group_var'), alpha = 0.1) +
    # p9.geom_smooth(data = ts_s2_long, mapping = p9.aes(x = 'date_yearless', y = 'value'), color = 'black', size = 3) +
                p9.facet_grid('variable ~ Subclass', scales = 'free_y') +
                p9.scale_x_datetime(date_breaks = '3 months', date_labels = '%b') +
                p9.theme(axis_text_x = p9.element_text(angle = text_angle, hjust =text_hjust),
                            figure_size = (8,len(s2_vars)),
                            axis_title = p9.element_blank()))
    st.markdown('### S2 timeseries')
    st.pyplot(p9.ggplot.draw(p_s2))


# %%
# Plot Landsat timeseries
if len(Subclass_vals) > 0 and len(landsat_vars) > 0:
    try:
        ts_landsat_prep = ts_landsat.query(landsat_query)
    except:
        st.warning('Ignoring invalid Landsat query -- must by valid pandas.DataFrame.query() string')
        ts_landsat_prep = ts_landsat

    ts_landsat_long = ts_landsat_prep \
        .query('Subclass in @Subclass_vals') \
        .melt(id_vars = ['loc_id','Year','Subclass','date_yearless'], value_vars=landsat_vars) \
        .assign(group_var = lambda x: x['loc_id'].astype(str) + '_' + x['Year'].astype(str))

    p_landsat = (p9.ggplot() + 
    p9.geom_line(data = ts_landsat_long, 
                mapping = p9.aes(x = 'date_yearless', y = 'value', group = 'group_var'), alpha = 0.1) +
                p9.facet_grid('variable ~ Subclass', scales = 'free_y') +
                p9.scale_x_datetime(date_breaks = '3 months', date_labels = '%b') +
                p9.theme(axis_text_x = p9.element_text(angle = text_angle, hjust =text_hjust),
                            figure_size = (8,len(landsat_vars)),
                            axis_title = p9.element_blank()))
    st.markdown('### Landsat timeseries')
    st.pyplot(p9.ggplot.draw(p_landsat))
# %%
