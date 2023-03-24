# install.packages("osmdata")

# https://cran.r-project.org/web/packages/osmdata/vignettes/osmdata.html

library(osmdata)
library(sf)
library(ggplot2)

# q <- 

# btm_water <- opq(bbox = "Battambang, Cambodia") %>%
btm_water <- opq(bbox = "Cambodia") %>%
  add_osm_feature(key = "natural", value = "water") %>%
  osmdata_sf() %>% magrittr::extract2("osm_polygons")

# osm_lines(btm_water)

# btm_waterways<- opq(bbox = "Battambang, Cambodia") %>%
btm_waterways<- opq(bbox = "Battambang, Cambodia") %>%
  add_osm_feature(key = "waterway", value = c("river","canal","stream","ditch","drain")) %>%
  osmdata_sf() %>% magrittr::extract2("osm_lines")
# head(q$available_features())

# ggp::obj_size(btm_water)
# unique(btm_waterways$waterway)

# btm_waterways$osm_lines %>% dplyr::filter(waterway == "flow_control")
names(btm$btm_water)
ggplot() + 
  # geom_sf(data = btm_waterways$osm_points, aes(color = waterway)) +
  geom_sf(data = btm_waterways, aes(color = waterway)) +
  geom_sf(data = btm_water, aes(fill = water), color = NA) +
  coord_sf(xlim = c(102, 104.5), ylim = c(12, 14))

btm_waterways %>% magrittr::extract2("osm_lines")

ggplot() + 
  # geom_sf(data = btm$osm_lines, aes(color = waterway)) +
  # geom_sf(data = btm$osm_multilines, aes(color = waterway)) +
  geom_sf(data = btm$osm_polygons, aes(fill = water, color = water))


btm$osm_polygons #%>% dplyr::select(-geometry)
anames
