# %%
import os
import ee
ee.Initialize()

# %%
# Set working directory
wds = ["/Users/gopalpenny/Projects/ml/classy",
       "/Users/gopal/Projects/ml/classy",
       "/home/svu/gpenny/Projects/classy"]
wd_exists = [x for x in wds if os.path.exists(x)][0]
os.chdir(wd_exists)

import sys
sys.path.insert(0, os.path.abspath('appmodules'))

# %%
# print(sys.path)
# sys.path.insert(os.path.abspath('appmodules'))
import DownloadPageFunctions as dpf


sample_pt_xy = ee.Geometry.Point([5.291, 12.132])
loc_id = 5
timeseries_dir_path = '/Users/gopal/Google Drive/_Research/Research projects/ML/classapp/app_data/Test6/Test6_download_timeseries'
date_range = ['2020-01-01', '2020-12-31']

# %%
dpf.DownloadS2pt(sample_pt_xy, loc_id, timeseries_dir_path, date_range, infobox = None)
# %%
