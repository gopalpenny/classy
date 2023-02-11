#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:15:26 2022

@author: gopal
"""

import pandas as pd
import geopandas as gpd

classpath_OLD = "/Users/gopal/Library/CloudStorage/GoogleDrive-gopalpenny@gmail.com/.shortcut-targets-by-id/162lXPvrsBfNPQDY7wSJwDkz0CcB9RmbS/Cambodia/Cambodia_classification/location_classification_OLD.csv"
classpath = "/Users/gopal/Library/CloudStorage/GoogleDrive-gopalpenny@gmail.com/.shortcut-targets-by-id/162lXPvrsBfNPQDY7wSJwDkz0CcB9RmbS/Cambodia/Cambodia_classification/location_classification.csv"

samppath = "/Users/gopal/Library/CloudStorage/GoogleDrive-gopalpenny@gmail.com/.shortcut-targets-by-id/162lXPvrsBfNPQDY7wSJwDkz0CcB9RmbS/Cambodia/Cambodia_sample_locations/sample_locations.shp"

classdf = pd.read_csv(classpath_OLD).rename({'loc_id':'ee_pt_id'}, axis = 1)

print(classdf)

# %%

samp_gpd = gpd.read_file(samppath)

samp_df = pd.DataFrame(samp_gpd)[['loc_id', 'ee_pt_id']]

samp_df

# %%

new_class_df = samp_df.merge(classdf, how = 'left', on = 'ee_pt_id').drop(columns = 'ee_pt_id')

new_class_df
# .to_csv(classpath, index = False)