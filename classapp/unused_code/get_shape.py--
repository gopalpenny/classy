#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:48:11 2022

@author: gopal
"""

# %% Initialize
# ee_generate_points

import ee
import os
import sys
import geemap
# import re

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)
from geemod import rs


# %%



# path_to_shp_import = st.text_input('Path to shapefile',
#               value = '/Users/gopal/Projects/ArkavathyTanksProject/arkavathytanks/spatial/CauveryBasin/Cauvery_boundary5.shp')

# shp_path = os.path.join(st.session_state.proj_path,"region")
app_path = '/Users/gopal/Google Drive/_Research projects/ML/manclassify/app_data'
proj_name = 'region1'

# watershed_pt = ee.Geometry.Point([76.9, 13])

# hyd_watershed = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_4') \
#   .filterBounds(watershed_pt) \
#   .union()

# path = 
# %%)

# %%

def GenerateSamples(app_path, proj_name):
    
    ee.Initialize()

    # Generate subdirectories
    proj_path = os.path.join(app_path,proj_name)
    region_path = os.path.join(proj_path,"region")
    region_shp_path = os.path.join(region_path,"region.shp")
    
    
    samples_dir_name = proj_name + '_sample_locations'
    samples_dir_path = os.path.join(proj_path,samples_dir_name)
    
    samples_name = 'sample_locations'
    samples_path = os.path.join(samples_dir_path,samples_name + '.shp')
    
    # %% GENERATE THE SAMPLES IF sample_locations.shp DOES NOT EXIST
    if os.path.exists(samples_path):
        st.write(samples_name + '.shp already exists')
        
    else:
        if not os.path.exists(samples_dir_path): 
            os.mkdir(samples_dir_path)
            
        # Import region.shp
        region_shp = gpd.read_file(region_shp_path)
        
        # Convert region.shp to ee.FeatureCollection
        region_fc_full = geemap.geopandas_to_ee(region_shp)
        region_fc = region_fc_full.union()
        
        # Get Image Collection for sample locations
        ic = ee.ImageCollection('COPERNICUS/S2_SR')
        im = ic.mosaic()
            
        # Generate the sample 
        samp_fc = im.sample(
            region = region_fc,
            scale = 10,
            numPixels = 1000,
            seed = 10,
            geometries = True).map(rs.set_feature_id_func('loc_id')).select('loc_id')
        
        # Export the sample
        task = ee.batch.Export.table.toDrive(
            collection = samp_fc,
            description = 'Generating samples',
            fileNamePrefix = samples_name,
            fileFormat = 'SHP',
            folder = samples_dir_name)
        
        task.start()
        
        st.write("Sent task to Earth Engine")