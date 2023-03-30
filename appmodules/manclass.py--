#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:10:02 2022

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
# appmodule

def testfunc():
    st.write("success")
        
        
# %% TIME SERIES STATUS


# def GetLocTimeseries(loc_id, timeseries_dir_path, plot_theme):
def GenS1data(loc_id, date_range):
    timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']
    s1_filename = 'pt_ts_loc' + str(loc_id) + '_s1.csv'
    s1 = pd.read_csv(os.path.join(timeseries_dir_path,s1_filename))
    
    s1['backscatter'] = (s1['VV']**2 + s1['VH']**2) ** (1/2)
    
    s1['datestr'] = [re.sub('.*?_1S.V_([0-9T]+)_.*','\\1',x) for x in s1['image_id']]
    
    s1['datetime'] = pd.to_datetime(s1['datestr'])
    
    s1_long = s1.melt(id_vars = 'datetime', value_vars = 'backscatter')
    s1_long['source'] = 'Sentinel 1'
    
    return s1_long


def GenS2data(loc_id, date_range):
    timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']
    s2_filename = 'pt_ts_loc' + str(loc_id) + '_s2.csv'
    s2 = pd.read_csv(os.path.join(timeseries_dir_path,s2_filename))
        
    # time_series_pd['datestr'] = [re.sub('([0-9T])_.*','\\1',x) for x in time_series_pd_load['image_id']]
    s2['datestr'] = [re.sub('([0-9T])_.*','\\1',x) for x in s2['image_id']]
    
    s2['datetime'] = pd.to_datetime(s2['datestr'])
    s2 = s2.assign(NDVI = lambda df: (df.B8 - df.B4)/(df.B8 + df.B4))
    # s2 = s2.assign(NDWI = lambda df: (df.B8 - df.B4)/(df.B8 + df.B4))
    
    
    s2_long = s2.melt(id_vars = ['datetime','cloudmask'], value_vars = ['B8','B4','B3','B2','NDVI'])
    s2_long['source'] = 'Sentinel 2'
    
    return s2_long

def GenOLI8data(loc_id, date_range):
    timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']
    oli8_filename = 'pt_ts_loc' + str(loc_id) + '_oli8.csv'
    oli8 = pd.read_csv(os.path.join(timeseries_dir_path,oli8_filename))
        
    # time_series_pd['datestr'] = [re.sub('([0-9T])_.*','\\1',x) for x in time_series_pd_load['image_id']]
    oli8['datestr'] = [re.sub('.*_([0-9]+)','\\1',x) for x in oli8['image_id']]
    
    oli8['datetime'] = pd.to_datetime([datetime.strptime(x, '%Y%m%d') for x in oli8['datestr']])
    oli8 = oli8.assign(NDVI = lambda df: (df.SR_B5 - df.SR_B4)/(df.SR_B5 + df.SR_B4))
    
    
    oli8_long = oli8.melt(id_vars = ['datetime','cloudmask'], value_vars = ['SR_B7','SR_B6','SR_B5','SR_B4','SR_B3','SR_B2','NDVI'])
    oli8_long['source'] = 'Landsat 8'
    
    return oli8_long
    


def GenCHIRPSdata(loc_id, date_range):
    timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']
    chirps_filename = 'pt_ts_loc' + str(loc_id) + '_chirps.csv'
    chirps = pd.read_csv(os.path.join(timeseries_dir_path,chirps_filename))
    
    chirps['datestr'] = [str(x) for x in chirps['image_id']]
    
    chirps['datetime'] = pd.to_datetime([datetime.strptime(x, '%Y%m%d') for x in chirps['datestr']])
    
    chirps_long = chirps.melt(id_vars = 'datetime', value_vars = 'precipitation')
    chirps_long['source'] = 'CHIRPS'
    
    return chirps_long
    # # s2['backscatter'] = (s2['VV']**2 + s2['VH']**2) ** (1/2)

def plotTimeseries(loc_id, date_range, month_seq, snapshot_dates, spectra_list):
    """
    

    Parameters
    ----------
    loc_id : Int
        Integer of the location ID.
    date_range : list
        List of strings in "%Y-%m-%d" format
    month_seq : List
        List of datetime.datetime values indicating the axis ticks and vertical gridlines.
    snapshot_dates : List
        List of datetime.datetime objects indicating the dates of the snapshot images in Classify page.
    spectra_list : list
        List of lists. Each sub-list contains the date ranges for spectra plot in Classify page.

    Returns
    -------
    p_sentinel : matplotlib.plt
        Plot of timeseries for the points in question.

    """
    
    timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']
    
    s1 = GenS1data(loc_id, date_range)
    s2 = GenS2data(loc_id, date_range)
    s2 = s2[s2['variable'] == 'NDVI']
    oli8 = GenOLI8data(loc_id, date_range)
    oli8 = oli8[oli8['variable'] == 'NDVI']
    chirps = GenCHIRPSdata(loc_id, date_range)
    
    datetime_range = [datetime.strptime(x, '%Y-%m-%d') for x in date_range]
    
    year_begin_datetime = datetime.strftime(datetime_range[len(datetime_range)-1],"%Y-01-01")
    
    alltimeseries = pd.concat([s1, s2, oli8, chirps])
    
    start_date = datetime_range[0]
    end_date = datetime_range[1]
    pre_date = start_date - relativedelta(months = 6)
    post_date = end_date + relativedelta(months = 6)
    
    month_seq_df = pd.DataFrame({'datetime':month_seq})

    line_vars = ['NDVI']
    smooth_vars_25 = ['backscatter']
    smooth_vars_05 = ['precipitation']
    alltimeseries_cloudfree = alltimeseries.query('cloudmask != 1')
    alltimeseries_cloudfree['date'] = [datetime.strftime(x, '%Y-%m-%d') for x in alltimeseries_cloudfree['datetime']]
    
    vert_scales = pd.DataFrame({
        'variable': ['NDVI']*3 + ['backscatter']*3 + ['precipitation']*3,
        'value': [0, 0.5, 1] + [0, 20, 40] + [0, 50, 100],
        'line': ['a', 'b', 'b']*3})
    
    # snapshots
    snapshot_datetimes = pd.DataFrame({
        'date' : snapshot_dates,
        'snapshot' : [str(x) for x in list(range(len(snapshot_dates)))]})
    alltimeseries_snapshots = alltimeseries_cloudfree.merge(snapshot_datetimes, how = 'inner', on = 'date')

    # Build date range for spectra
    spectral_range = pd.DataFrame(columns = ['id','start_date', 'end_date', 'variable'])
    for i in range(len(spectra_list)):
        row_list = [i] + list(spectra_list[i]) + [line_vars[0]]
        spectral_range.loc[i] = row_list
        
    var_min = alltimeseries_cloudfree[alltimeseries_cloudfree['variable'] == line_vars[0]].value.min()
    var_max = alltimeseries_cloudfree[alltimeseries_cloudfree['variable'] == line_vars[0]].value.max()
    spectral_range['yval'] = var_min + spectral_range.id * (var_max - var_min) * 0.05
    
    month_seq_labels_full = [datetime.strftime(x, "%b %d, %Y") for x in month_seq]
    month_seq_labels_brief = [datetime.strftime(x, "%b %d") for x in month_seq]
    month_seq_labels = [month_seq_labels_full[i] if i == 0 or i == (len(month_seq)-len(month_seq)) else month_seq_labels_brief[i] for i in range(len(month_seq))]
    
    p_timeseries = (
        p9.ggplot(data = alltimeseries_cloudfree, mapping = p9.aes('datetime', 'value')) + 
        p9.annotate('rect',xmin = start_date, xmax = end_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'white', color = 'black', alpha = 1) +
        p9.geom_segment(data = spectral_range, mapping = p9.aes(x = 'start_date', xend = 'end_date', y = 'yval', yend = 'yval',color = 'id'), size = 2) +
        p9.geom_vline(data = month_seq_df, mapping = p9.aes(xintercept = 'datetime'), color = 'black', alpha = 0.5) +
        p9.annotate('vline', xintercept = year_begin_datetime, color = 'gray', linetype = 'dashed', alpha = 0.5) +
        p9.geom_abline(data = vert_scales, mapping = p9.aes(intercept = 'value', slope = 0, linetype = 'line'), color = 'red', alpha = 0.35) +
        p9.geom_point(mapping = p9.aes(shape = 'source')) + 
        p9.geom_line(data = alltimeseries_cloudfree[alltimeseries_cloudfree.variable.isin(line_vars)], mapping = p9.aes(group = 'source')) + 
        p9.geom_smooth(data = alltimeseries_cloudfree[alltimeseries_cloudfree.variable.isin(smooth_vars_25)], span = 0.1) + 
        p9.geom_smooth(data = alltimeseries_cloudfree[alltimeseries_cloudfree.variable.isin(smooth_vars_05)], span = 0.05) + 
        p9.geom_point(data = alltimeseries_snapshots, mapping=p9.aes(fill = 'snapshot'), size = 4) + 
        p9.scale_color_continuous(guide = False) +
        p9.scale_fill_discrete(guide = False) +
        p9.facet_wrap('variable', scales = 'free_y',ncol = 1) +
        # p9.xlim()+
        p9.guides(linetype = False) +
        # p9.scale_x_datetime(limits = [datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)], 
        p9.scale_linetype_manual(values = ['solid','dashed']) +
        p9.scale_x_datetime(limits = [pre_date, post_date], breaks = month_seq, labels = month_seq_labels) + # date_labels = '%Y-%b', date_breaks = '1 year') +
        p9.expand_limits(y = 0) +# PlotTheme() + 
        # p9.theme(legend_position = (0.3,0.3))
        PlotTheme() + 
        p9.theme(
                figure_size=(7.5,4.8),
                axis_title_x = p9.element_blank(),
                panel_background= p9.element_rect(fill = 'gray', color = None),
                # panel_border= p9.element_rect(fill = 'gray'),
                legend_title = p9.element_blank(),
                legend_background = p9.element_rect(fill = 'black', color = None, alpha = 0.5),
                legend_text = p9.element_text(color = 'white'),
                axis_title_y = p9.element_blank(),
                legend_position = 'bottom')
        )
    
    return p_timeseries

def plotSpectra(loc_id, date_range, spectra_list):
    timeseries_dir_path = st.session_state['paths']['timeseries_dir_path']
    
    s1 = GenS1data(loc_id, date_range)
    s2 = GenS2data(loc_id, date_range)
    sentinel = pd.concat([s1, s2]).query('cloudmask != 1')
    # sentinel['date'] = [datetime.strftime(x, '%Y-%m-%d') for x in sentinel['datetime']]
    
    s2_bands = pd.DataFrame({
        'variable' : ['B2','B3','B4','B8','backscatter','NDVI'],
        'freq_nm': [495, 560, 660, 835, -1, -1]
        })
    
    vert_scales = pd.DataFrame({
        'value': [0, 0.25, 0.5],
        'line': ['a', 'b', 'b']})
    
    sent_range_all = pd.DataFrame(columns = list(sentinel.columns) + ['rangenum'])
    for i in range(len(spectra_list)):
        range_prep = sentinel.loc[(spectra_list[i][0] <= sentinel['datetime']) & (sentinel['datetime'] <= spectra_list[i][1])]
        range_prep['rangenum'] = i
        sent_range_all = sent_range_all.append(range_prep)
        
    sent_range_all = sent_range_all.merge(s2_bands, how = 'left', on = 'variable')
    
    sent_range = (sent_range_all
      .groupby(['variable','rangenum','freq_nm'], as_index = False)
      .agg({'value' : 'mean'}))
    
    sent_range['Reflectance'] = sent_range['value'] / 1e4
    
    print(sent_range)
    strlit_color = '#0F1116'
    p_spectra = (
        p9.ggplot(data = sent_range[sent_range['freq_nm'] > 0], mapping = p9.aes(x = 'freq_nm', y = 'Reflectance', color = 'rangenum')) +
        p9.geom_abline(data = vert_scales, mapping = p9.aes(intercept = 'value', slope = 0, linetype = 'line'), color = 'red', alpha=0.75) +
        p9.scale_linetype_manual(values = ['solid', 'dashed'], guide = False) + 
        p9.geom_line(size = 2) + p9.geom_point(size = 5) + 
        p9.xlab('Frequency, nm') +
        p9.expand_limits(y = (0, 0.3)) +
        p9.scale_color_manual(['#572851','#FAE855']) +
        PlotTheme() +
        p9.theme(legend_position = 'none',
                 panel_background = p9.element_rect(fill = strlit_color, color = 'lightgray'),
                  panel_grid_major = p9.element_line(color = 'lightgray', size = 0.5, alpha = 0.2),
                 figure_size = (4.5,2)))
    
    return p_spectra
    
    # line_vars = ['NDVI']
    # bar_vars = ['backscatter']
    # sentinel_cloudfree = sentinel.query('cloudmask != 1')
    
    # # snapshots
    # snapshot_datetimes = pd.DataFrame({
    #     'date' : snapshot_dates,
    #     'snapshot' : [str(x) for x in list(range(len(snapshot_dates)))]})
    # sentinel_snapshots = sentinel_cloudfree.merge(snapshot_datetimes, how = 'inner', on = 'date')
    
    # # date range for spectra
    # spectral_range = pd.DataFrame(columns = ['id','start_date', 'end_date', 'variable'])
    # for i in range(len(spectra_list)):
    #     row_list = [i] + list(spectra_list[i]) + [line_vars[0]]
    #     spectral_range.loc[i] = row_list
        
        
    # var_min = sentinel_cloudfree[sentinel_cloudfree['variable'] == line_vars[0]].value.min()
    # print(var_min)
    # # .value.min()
    # var_max = sentinel_cloudfree[sentinel_cloudfree['variable'] == line_vars[0]].value.max()
    # # print('var_min')
    # # print(type(var_min))
    # # print(type(spectral_range.id))
    # spectral_range['yval'] = var_min + spectral_range.id * (var_max - var_min) * 0.05
        
    # print(spectral_range)
    # # spectral_range
    # # print(sentinel_cloudfree.variable[0])
    
    # p_sentinel = (
    #     p9.ggplot(data = sentinel_cloudfree, mapping = p9.aes('datetime', 'value')) + 
    #     p9.annotate('rect',xmin = start_date, xmax = end_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'white', color = 'black', alpha = 1) +
    #     p9.geom_segment(data = spectral_range, mapping = p9.aes(x = 'start_date', xend = 'end_date', y = 'yval', yend = 'yval',color = 'id'), size = 2) +
    #     p9.geom_vline(data = month_seq_df, mapping = p9.aes(xintercept = 'datetime'), color = 'black', alpha = 0.5) +
    #     # p9.annotate('rect',xmin = end_date, xmax = post_date, ymin = -np.Infinity, ymax = np.Infinity, fill = 'black', alpha = 0.5) +
    #     p9.geom_point() + 
    #     p9.geom_line(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(line_vars)]) + 
    #     p9.geom_point(data = sentinel_snapshots, mapping=p9.aes(fill = 'snapshot'), size = 4) + 
    #     p9.geom_smooth(data = sentinel_cloudfree[sentinel_cloudfree.variable.isin(smooth_vars)], span = 0.25) + 
    #     p9.facet_wrap('variable', scales = 'free_y',ncol = 1) +
    #     # p9.xlim()+
    #     # p9.scale_x_datetime(limits = [datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)], 
    #     p9.scale_x_datetime(limits = [pre_date, post_date], 
    #                         date_labels = '%Y-%b', date_breaks = '1 year') +
    #     PlotTheme() + p9.theme(axis_title_x = p9.element_blank(),
    #                            panel_background= p9.element_rect(fill = 'gray', color = None),
    #                            # panel_border= p9.element_rect(fill = 'gray'),
    #                            legend_position = 'none'))
    
    # return p_sentinel


def MapTheme():
    map_theme = p9.theme(panel_background = p9.element_rect(fill = None),      
                     panel_border = p9.element_rect(),
                     panel_grid_major=p9.element_blank(),
                     panel_grid_minor=p9.element_blank(),
                     plot_background=p9.element_rect(fill = None))
    return map_theme




def PlotTheme():
    strlit_color = '#0F1116'
    plot_theme = p9.theme(panel_background = p9.element_rect(fill = None),      
                     panel_border = p9.element_rect(color = None),
                     panel_grid_major=p9.element_blank(),
                     panel_grid_minor=p9.element_blank(),
                     axis_text = p9.element_text(color = 'white'),
                     axis_ticks = p9.element_line(color = 'white'),
                     axis_title = p9.element_text(color = 'white'),
                     # plot_background=p9.element_rect(fill = 'black'),
                     plot_background=p9.element_rect(fill = strlit_color, color = strlit_color))
    return plot_theme




# %%
    

# %%