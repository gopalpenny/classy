

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
ts_s1, ts_s2, ts_landsat = prep_label_timeseries_all(class_df_long, generate_ts = False, timeseries_dir_path = timeseries_dir_path)

# %%
Subclass_options = pd.unique(ts_s1.Subclass)
s1_bands = ['backscatter']
s2_bands = ['B8','B4','B3','B2','NDVI']
landsat_bands = ['swir2','swir1','nir','red','green','blue','NDVI']

with st.sidebar:
    st.multiselect('Subclasses', options = Subclass_options)
    st.multiselect('S1 bands', options = s1_bands)
    st.multiselect('S2 bands', options = s2_bands)
    st.multiselect('Landsat bands', options = landsat_bands)


# %%
Subclass_vals = ['Single crop','Double crop','Plantation']
s1_vars = ['backscatter']
ts_s1_long = ts_s1 \
    .query('Subclass in @Subclass_vals') \
    .melt(id_vars = ['loc_id','Year','Subclass','date_yearless'], value_vars=s1_vars) 

# %%
p_s1 = (p9.ggplot() + 
 p9.geom_line(data = ts_s1_long, 
               mapping = p9.aes(x = 'date_yearless', y = 'value', group = 'loc_id'), alpha = 0.1) +
               p9.facet_grid('variable ~ Subclass', scales = 'free_y') +
               p9.scale_x_datetime(date_breaks = '3 months', date_labels = '%b') +
               p9.theme(axis_text_x = p9.element_text(angle = 45, hjust =1),
                        figure_size = (8,3),
                        axis_title_x = p9.element_blank()))
# %%
s2_vars = ['NDVI']
ts_s2_long = ts_s2 \
    .query('Subclass in @Subclass_vals') \
    .melt(id_vars = ['loc_id','Year','Subclass','date_yearless'], value_vars=s2_vars) 
p_s2 = (p9.ggplot() + 
 p9.geom_line(data = ts_s2_long, 
               mapping = p9.aes(x = 'date_yearless', y = 'value', group = 'loc_id'), alpha = 0.1) +
               p9.facet_grid('variable ~ Subclass', scales = 'free_y') +
               p9.scale_x_datetime(date_breaks = '3 months', date_labels = '%b') +
               p9.theme(axis_text_x = p9.element_text(angle = 45, hjust =1),
                        figure_size = (8,3),
                        axis_title_x = p9.element_blank()))
# %%
landsat_vars = ['NDVI','swir1']
ts_landsat_long = ts_landsat \
    .query('Subclass in @Subclass_vals') \
    .melt(id_vars = ['loc_id','Year','Subclass','date_yearless'], value_vars=landsat_vars)

p_landsat = (p9.ggplot() + 
 p9.geom_line(data = ts_landsat_long, 
               mapping = p9.aes(x = 'date_yearless', y = 'value', group = 'loc_id'), alpha = 0.1) +
               p9.facet_grid('variable ~ Subclass', scales = 'free_y') +
               p9.scale_x_datetime(date_breaks = '3 months', date_labels = '%b') +
               p9.theme(axis_text_x = p9.element_text(angle = 45, hjust =1),
                        figure_size = (8,3),
                        axis_title_x = p9.element_blank()))
# %%

st.pyplot(p9.ggplot.draw(p_s1)) 

st.pyplot(p9.ggplot.draw(p_s2))

st.pyplot(p9.ggplot.draw(p_landsat))