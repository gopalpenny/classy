""" scratch

"""


def f1(x):

  def f2():
    return x**2
  
  x = 10
  return f2()


def f3(x):
  return f1(x) ** 2



# import scratch as scr
# import importlib
# # importlib.reload(scr)

# scr.f1(2)
# cd /content/drive/MyDrive/_Research projects/asiaAg/pyscripts

srtm = ee.Image("CGIAR/SRTM90_V4")

m = geemap.Map()
m.addLayerControl()

bbox = ee.Geometry.BBox(65.9, 28.4, 66, 28.5)
m.addLayer(bbox,{},'bbox')
m.centerObject(bbox, 8)

import mapwidgets
mapwidgets.add_coords_click(m)

# https://github.com/google/earthengine-api/issues/87
import rs

importlib.reload(rs)

sp = rs.get_raster_pts(srtm,bbox,90)
sp360 = rs.get_raster_pts(srtm,bbox,3600)

m.addLayer(sp,{}, 'sp')
m.addLayer(sp360.draw(color = 'red', pointRadius = 5),{}, 'sp360')

# Now we can import the library and use the function.
import ees
s2 = ees.myFunc()
print(s2.first().getInfo())