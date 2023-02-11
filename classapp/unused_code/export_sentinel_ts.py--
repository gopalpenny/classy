#!/usr/bin/env python
# coding: utf-8

# # Title
# 
# And some text below

# %%
import ee
import re
import pandas as pd
from plotnine import *
import sys
import os

gdrive_path = '/Users/gopal/Google Drive'
gdrive_ml_path = os.path.join(gdrive_path, '_Research projects/ML')
sys.path.append(gdrive_ml_path)

out_folder = 'gee_sentinel_ts'
out_path = os.path.join(gdrive_ml_path, 'manclassify/script_output', out_folder)

if not os.path.exists(out_path): os.mkdir(out_path)

# %%
import geemap

# In[3]:


from geemod import eesentinel as ees
from geemod import rs


# In[4]:

import collections
collections.Callable = collections.abc.Callable
ee.Initialize()


# In[19]:


watershed_pt = ee.Geometry.Point([76.9,13])

# s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
#   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',50)) \
#   .filterDate('2019-06-01','2021-05-31')

# s2_clouds = ees.add_cld_shadow_mask_func()


# In[34]:


# s2.aggregate_array('system:index').getInfo()


# In[20]:


# Define a location or area of interest
aoi = watershed_pt

# Define parameters for the cloud and cloud shadow masking
s2params = {
   'START_DATE' : '2019-06-01',
   'END_DATE' : '2021-06-01',
   'CLOUD_FILTER' : 50,
   'CLD_PRB_THRESH' : 55,
   'NIR_DRK_THRESH' : 0.2,
   'CLD_PRJ_DIST' : 2.5, # 1 for low cumulus clouds, 2.5 or greater for high elevation or mountainous regions
   'BUFFER' : 50
}

# Retrieve and filter S2 and S2 clouds
s2_and_clouds_ic = ees.get_s2_sr_cld_col(aoi, s2params) 

# Mask clouds and cloud shadows
s2_clouds_ic = s2_and_clouds_ic.map(ees.add_cld_shadow_mask_func(s2params))
# This is the final result

# Reduce to a single image for mapping
s2_clouds_im = s2_clouds_ic.reduce(ee.Reducer.first())



# In[7]:


# m = geemap.Map()
# m.addLayerControl()
# m.centerObject(aoi, 9)
# m.addLayer(s2_clouds_im, {'bands': ['B8_first','B4_first','B3_first'], 'min':0, 'max': 3000},'S2 FCC')
# m.addLayer(s2_clouds_im, {'bands':['probability_first'],'min': 0, 'max': 100},'probability (cloud)')
# m.addLayer(s2_clouds_im.select('clouds_first').selfMask(),{'bands':['clouds_first'],'palette': ['#e056fd']},'clouds')
# m.addLayer(s2_clouds_im.select('cloud_transform_first').selfMask(),{'bands':['cloud_transform_first'],'min': 0, 'max': 1, 'palette': ['white', 'black']},'cloud_transform')
# m.addLayer(s2_clouds_im.select('dark_pixels_first').selfMask(),{'bands':['dark_pixels_first'],'palette': ['orange']},'dark_pixels')
# m.addLayer(s2_clouds_im.select('shadows_first').selfMask(), {'bands':['shadows_first'], 'palette': ['yellow']},'shadows')
# m.addLayer(s2_clouds_im.select('cloudmask_first').selfMask(), {'bands':['cloudmask_first'], 'palette': ['orange']},'cloudmask')
# m

# %%

id = 1
time_series_fc = rs.get_pixel_ts_allbands(
    pts_fc = ee.FeatureCollection(watershed_pt),
    image_collection = s2_clouds_ic.select(['B8','B4','B3','B2','cloudmask']),
    ic_property_id = 'system:index',
    scale = 10) # for Landsat resolution
# time_series_pd_load = geemap.ee_to_pandas(time_series_fc)
    
task = ee.batch.Export.table.toDrive(
    collection = time_series_fc,
    folder = out_folder,
    fileNamePrefix = 'time_series_' + str(id))

task.start()


# time_series_fc


# In[24]:




