import ipywidgets as widgets
from ipyleaflet import WidgetControl

# Add an output widget to the map
output_widget = widgets.Output(layout={'border': '1px solid black'})
output_control = WidgetControl(widget=output_widget, position='bottomright')


def add_coords_click(m):
  """Display coordinates on click
  To do this, run:
  m = geemap.Map()
  import mapwidgets
  mapwidgets.add_coords_click(m)
  """
  m.add_control(output_control)

  # Capture user interaction with the map
  def handle_interaction(**kwargs):
      latlon = kwargs.get('coordinates')
      if kwargs.get('type') == 'click':
          m.default_style = {'cursor': 'wait'}
          # xy = ee.Geometry.Point(latlon[::-1])

          with output_widget:
              output_widget.clear_output()
              print(latlon)
      m.default_style = {'cursor': 'pointer'}

  m.on_interaction(handle_interaction)