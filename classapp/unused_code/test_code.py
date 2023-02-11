#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:52:27 2022

@author: gopal
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import math
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotnine as p9

df = pd.DataFrame(
    {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
      'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
      'date': ['2020-01-01', '2017-01-02', '2018-01-01', '2024-01-02', '2023-01-01'],
      'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
      'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})
df['nothing'] = str(np.nan)
df['datetime'] = [datetime.strptime(x, '%Y-%m-%d') for x in df['date']]

df['test'] = (df['Longitude'] > -50) & (df['Latitude'] < 0)
# df.groupby(['test','City'], as_index = False).agg({'Longitude' : 'mean'})
diffs = np.abs(df['datetime'] - datetime.strptime('2018-01-01', '%Y-%m-%d'))
[df['datetime'][i] for i in range(len(diffs)) if np.min(diffs) == diffs[i]]

np.mean(pd.Series([datetime.strptime(x, '%Y-%m-%d') for x in df['date']]))
#%%

gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))

gdf.loc[gdf.Country == "Brazil", 'geometry'] = gdf.geometry.iloc[3]

gdf = gdf.set_crs(4326)
gdf.geometry.crs
#%%

df['datetime'][1]
# %%
a = pd.to_datetime(df['date'])
b = datetime.strptime('2020-03-01', '%Y-%m-%d')

b > a[1]
type(a[1])
# df.columns
# %%

df[df.Latitude >= 0.56]

# %%

# [df.City.iloc[i] == 'Brasilia' for i in range(len(df.City))]

idx = df.index[df.City in ['Brasilia']].tolist()[0] + 1

# df.loc[idx]

# %%


newidx = [idx[0] + 1]

# newidx

df.loc[newidx, 'Latitude'] + 1

# %%

# [i for i in range(len(dfSant.City)) if dfSant.City[i] in ['Santiago']]

idx = [i for i in range(len(df.City)) if df.City[i] in ['Brasilia']]
new = df.City.shift(-1)[idx]

new
# new.iloc[0]
# %%

df.City

# %%

filterargs = {
    'lon' : [-180, 180],
    'lat' : [-90, 90],
    'Class' : 'Any',
    'Subclass' : 'Any',
    'Downloaded' : 'Yes'
    }

# %%

filterargs['lon'][0]

# %%
strlit_color = '#0F1116'
darktheme = p9.theme(panel_background = p9.element_rect(fill = None),      
                 panel_border = p9.element_rect(color = None),
                 panel_grid_major=p9.element_blank(),
                 panel_grid_minor=p9.element_blank(),
                 axis_text = p9.element_text(color = 'white'),
                 axis_ticks = p9.element_line(color = 'red'),
                 axis_title = p9.element_text(color = 'white'),
                 # plot_background=p9.element_rect(fill = 'black'),
                 plot_background=p9.element_rect(fill = strlit_color, color = strlit_color))

# %%
p9.ggplot() + p9.geom_point(data = df, mapping = p9.aes(x = 'Longitude', y = 'Latitude', color = 'nothing'))

# %%

month_start_all = [datetime.strftime(datetime.strptime('2000-' + str(x) + '-01','%Y-%m-%d'), '%b') for x in range(1,13)]

month_start_all

# %%
list1 = [0,1]
list2 = [2,3]

lists = [list1, list2]

lists[0][1]