#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 09:37:40 2022

@author: gopal
"""

# ee_test_code.py


import ee
import geemap

# %%

ee.Initialize()

# %%

ee.ImageCollection('COPERNICUS/S2_SR')