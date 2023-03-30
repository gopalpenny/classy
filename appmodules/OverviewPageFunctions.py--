#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:28:37 2022

@author: gopal
"""

import streamlit as st
import os
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta


# %%

def CreateNewProject():
    new_project_name = st.session_state['new_project_name']
    app_path = st.session_state['app_path']
    new_project_path = os.path.join(app_path, new_project_name)
    
    os.mkdir(new_project_path)
    st.session_state['proj_name_box'] = new_project_name
    st.session_state['proj_name'] = new_project_name
    checkProjStatus()
    st.session_state['new_project_name'] = ''
    
def UpdateProjName():
    print('running UpdateProjName()')
    if not st.session_state['proj_name_box'] == 'Create new project':
        st.session_state['proj_name'] = st.session_state['proj_name_box']
        checkProjStatus()
        
def UpdateAppPath():
    st.session_state['app_path'] = st.session_state['app_path_box']
    if 'proj_name' in st.session_state:
        checkProjStatus()
    
def checkProjStatus():
    print('running checkProjStatus()')
    paths = {}
    app_path = st.session_state['app_path']
    proj_name = st.session_state['proj_name']
    paths['proj_path'] = os.path.join(app_path, proj_name)
    paths['samples_dir_name'] = proj_name + "_sample_locations"
    paths['px_poly_dir_name'] = proj_name + "_px_poly"
    paths['sample_locations_dir_path'] = os.path.join(paths['proj_path'], paths['samples_dir_name'])
    paths['px_poly_dir_path'] = os.path.join(paths['sample_locations_dir_path'], paths['px_poly_dir_name'])
    paths['random_locations_path'] = os.path.join(paths['sample_locations_dir_path'], "random_locations.shp")
    paths['sample_locations_path'] = os.path.join(paths['sample_locations_dir_path'], "sample_locations.shp")
    paths['region_shp_path'] = os.path.join(paths['sample_locations_dir_path'],"region.shp")
    
    paths['timeseries_dir_name'] = proj_name + "_download_timeseries"
    paths['timeseries_dir_path'] = os.path.join(paths['proj_path'], paths['timeseries_dir_name']) 
    
    status = {}
    status['region_status'] = os.path.exists(paths['region_shp_path'])
    status['random_status'] = os.path.exists(paths['random_locations_path'])
    status['sample_status'] = os.path.exists(paths['sample_locations_path'])
    
    classification_dir_name = proj_name + "_classification"
    paths['classification_dir_path'] = os.path.join(paths['proj_path'], classification_dir_name)
    if not os.path.exists(paths['classification_dir_path']): os.mkdir(paths['classification_dir_path'])
    paths['class_path'] = os.path.join(paths['classification_dir_path'], 'location_classification.csv')
    
    st.session_state['paths'] = paths
    st.session_state['status'] = status
    
    st.session_state['proj_vars'] = readProjectVars()
    
    
    if not os.path.exists(paths['sample_locations_dir_path']): 
        os.mkdir(paths['sample_locations_dir_path'])
    
# %%
def saveProjectVars(vars_dict):
    print('running saveProjectVars()')
    proj_path = st.session_state['paths']['proj_path']
    vars_path = os.path.join(proj_path, "project_vars.json")
    json.dump(vars_dict, open(vars_path, 'w', encoding = 'utf-8'), ensure_ascii=True, indent=4)
    
def readProjectVars():
    print('running readProjectVars()')
    proj_path = st.session_state['paths']['proj_path']
    vars_path = os.path.join(proj_path, "project_vars.json")
    
    if os.path.exists(vars_path):
        vars_dict = json.load(open(vars_path))
    else:
        init_year_default = 2014
        init_year_end_default = 2022
        init_month_default = 'June'
        init_day_default = 1
        vars_dict = {'classification_year_default' : init_year_default,
                     'classification_start_month' : init_month_default,
                     'classification_start_day' : init_day_default}
        saveProjectVars(vars_dict)
        setClassificationStartDate(init_year_default, init_month_default, init_day_default)
        
        start_date_str = str(init_year_default) + '-' + init_month_default + '-' +  str(init_day_default)
        start_date = datetime.strftime(datetime.strptime(start_date_str, '%Y-%B-%d'), '%Y-%m-%d')
        end_date_str = str(init_year_end_default) + '-' + init_month_default + '-' +  str(init_day_default)
        end_date = datetime.strftime(datetime.strptime(end_date_str, '%Y-%B-%d'), '%Y-%m-%d')
        
        setProjectTimespan(start_date, end_date)
        vars_dict = json.load(open(vars_path))
    
    return vars_dict

def setClassificationStartDate(year, month, day):
    """
    setClassificationStartDate

    Parameters
    ----------
    year : Int
        Default year to start classification.
    month : Str
        Month as full month name (i.e., 'January', etc).
    day : Int
        Day of month.

    Returns
    -------
    None.

    """
    print('running setClassificationStartDate()')
    st.session_state['proj_vars']['classification_year_default'] = year
    st.session_state['proj_vars']['classification_start_month'] = month
    st.session_state['proj_vars']['classification_start_day'] = day
    vars_dict = st.session_state['proj_vars']
    
    saveProjectVars(vars_dict)
    

def setProjectTimespan(start_date, end_date):
    """
    setProjectTimespan

    Parameters
    ----------
    start_date : Str
        Start date for project as string in YYYY-MM-DD format.
    end_date : Str
        End date for project as string in YYYY-MM-DD format.

    Returns
    -------
    None.

    """
    print('running setProjectTimespan()')
    proj_vars = st.session_state['proj_vars']
    proj_vars['proj_start_date'] = start_date
    proj_vars['proj_end_date'] = end_date
    
    # add years to proj_vars
    start_year = int(datetime.strftime(datetime.strptime(start_date, '%Y-%m-%d'), '%Y'))
    end_year = int(datetime.strftime(datetime.strptime(end_date, '%Y-%m-%d') - relativedelta(months = 6), '%Y'))
    proj_vars['proj_years'] = list(range(start_year, end_year + 1))
    
    st.session_state['proj_vars'] = proj_vars
    saveProjectVars(proj_vars)
